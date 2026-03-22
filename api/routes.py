"""
API routes — /query, /ingest, /evaluate endpoints.

All endpoints return structured JSON with timing and metadata.
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.cache_manager import cache_manager as cache
from api.memory import MemoryManager
from config import get_settings
from generation.guardrails import Guardrails
from generation.prompt_templates import ANALYSIS_SYSTEM_PROMPT
from generation.response_generator import ResponseGenerator
from ingestion.chunking import DocumentChunker, SemanticChunker
from ingestion.embedding_pipeline import EmbeddingPipeline
from ingestion.loaders import DocumentLoader
from reranking.cohere_rerank import CohereReranker
from retrieval.agentic_search import AgenticSearcher
from retrieval.hybrid_search import HybridSearcher
from retrieval.hyde import HyDEGenerator
from retrieval.multi_query import MultiQuerySearcher
from retrieval.router import QueryRouter

# Persistent agent instance to avoid re-initializing on every request
_agent_searcher: Optional[AgenticSearcher] = None


def get_agent_searcher():
    global _agent_searcher
    if _agent_searcher is None:
        _agent_searcher = AgenticSearcher()
    return _agent_searcher


logger = structlog.get_logger(__name__)
router = APIRouter()


# ── Request / Response models ─────────────────────────────
class QueryRequest(BaseModel):
    session_id: Optional[str] = None
    question: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(default=30, ge=1, le=100, description="Retrieval candidates")
    top_n: int = Field(default=5, ge=1, le=30, description="After reranking")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="LLM temperature")
    multi_query_enabled: bool = Field(default=True, description="Enable Multi-Query expansion")
    filters: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    model: str
    retrieval_count: int
    latency_ms: float
    tokens_used: int


class IngestRequest(BaseModel):
    directory: str = Field(..., description="Path to document directory")
    extensions: Optional[List[str]] = Field(default=None, description="File extensions to include")


class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_created: int
    vectors_stored: int
    latency_ms: float


class EvaluateRequest(BaseModel):
    dataset_path: str = Field(..., description="Path to evaluation dataset JSON")


class EvaluateResponse(BaseModel):
    scores: Dict[str, float]
    num_samples: int
    passed: bool
    evaluation_time_s: float


# Rebuild models for Pydantic v2 forward reference resolution
QueryRequest.model_rebuild()
QueryResponse.model_rebuild()
IngestRequest.model_rebuild()
IngestResponse.model_rebuild()
EvaluateRequest.model_rebuild()
EvaluateResponse.model_rebuild()


# ── Shared state ──────────────────────────────────────────
_hybrid_searcher: Optional[HybridSearcher] = None
_bm25_built = False


def _get_hybrid_searcher() -> HybridSearcher:
    global _hybrid_searcher, _bm25_built
    if _hybrid_searcher is None:
        _hybrid_searcher = HybridSearcher()
    if not _bm25_built:
        _hybrid_searcher.bm25.build_index()
        _bm25_built = True
    return _hybrid_searcher


# ── /query ────────────────────────────────────────────────
@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Full RAG pipeline: hybrid retrieval → reranking → LLM generation."""
    start = time.perf_counter()

    try:
        # 1. Check cache first for performance
        cache_params = req.dict(exclude={"session_id"})
        cached_resp = cache.get_response(req.question, cache_params)
        if cached_resp:
            logger.info("serving_from_cache", question=req.question)
            return QueryResponse(**cached_resp)

        guardrails = Guardrails()
        is_safe, msg = guardrails.check_query(req.question)
        if not is_safe:
            return QueryResponse(
                answer=msg,
                sources=[],
                model="guardrails",
                retrieval_count=0,
                latency_ms=0.0,
                tokens_used=0,
            )

        searcher = _get_hybrid_searcher()
        multi_searcher = MultiQuerySearcher(searcher)
        reranker = CohereReranker()
        generator = ResponseGenerator()
        router_module = QueryRouter()
        hyde = HyDEGenerator()

        # Context-aware metadata retrieval
        memory = MemoryManager()
        active_filename = None

        # 1. Extract filename from query or request filters
        if req.filters and "filename" in req.filters:
            active_filename = req.filters["filename"]
        else:
            # Robust extraction using regex
            pdf_match = re.search(r"([\w\-. ]+\.pdf)", req.question, re.IGNORECASE)
            if pdf_match:
                active_filename = pdf_match.group(1).strip()

        # 2. Track filename for session
        if active_filename:
            memory.save_last_filename(req.session_id, active_filename)
        else:
            # Try to recover from session history for vague "more details" requests
            vague_terms = [
                "bigger description",
                "more details",
                "detailed overview",
                "summarize the file",
                "about it",
                "of that file",
            ]
            if any(term in req.question.lower() for term in vague_terms):
                active_filename = memory.get_last_filename(req.session_id)

        # 3. Handle description requests with deep analysis fallback
        is_generic_desc = any(
            t in req.question.lower()
            for t in ["description", "overview", "about the file", "what is in the file", "tell me about"]
        )
        is_bigger_request = any(
            t in req.question.lower()
            for t in ["bigger", "more detail", "detailed breakdown", "exhaustive", "detailed explanation"]
        )

        system_prompt_override = None

        if active_filename and is_generic_desc:
            if not is_bigger_request:
                # Return cached short description for speed if it's a simple "what is this"
                metadata = memory.get_document_metadata(active_filename)
                if metadata and metadata.get("short_description"):
                    logger.info("metadata_retrieval_path", filename=active_filename, type="short")
                    return QueryResponse(
                        answer=metadata["short_description"],
                        sources=[{"filename": active_filename, "score": 1.0}],
                        model="metadata-retrieval",
                        retrieval_count=0,
                        latency_ms=0.5,
                        tokens_used=0,
                    )
            else:
                # For "bigger" or "detailed" requests, FORCIBLY trigger a high-depth RAG search
                logger.info("triggering_deep_analysis", filename=active_filename)
                req.top_k = 50  # Get massive context for the breakdown
                system_prompt_override = ANALYSIS_SYSTEM_PROMPT
                # We let it fall through to the RAG path below with the custom prompt

        # 4. Scoped RAG if we have an active filename but it's a specific question
        search_filters = req.filters or {}
        if active_filename and "filename" not in search_filters:
            search_filters["filename"] = active_filename

        # FIX: Define decision and candidates before usage
        decision = router_module.route_query(req.question)
        candidates = []
        reranked = []

        if decision == "RAG":
            # 1. Retrieval (HyDE + Optional Multi-Query)
            search_query = hyde.generate(req.question)

            if req.multi_query_enabled:
                candidates = multi_searcher.search(query=search_query, top_k=req.top_k, filters=search_filters)
            else:
                candidates = searcher.search(query=search_query, top_k=req.top_k, filters=search_filters)

            if not candidates:
                raise HTTPException(status_code=404, detail="No documents found. Ingest documents first.")

            # 2. Reranking (using original question for scoring accuracy)
            reranked = reranker.rerank(req.question, candidates, top_n=req.top_n)

        # 3. Generation
        result = generator.generate(
            query=req.question, contexts=reranked, temperature=req.temperature, system_prompt=system_prompt_override
        )

        # 4. Save history
        memory.save_message(req.session_id, "user", req.question)
        memory.save_message(req.session_id, "assistant", result.answer)

        latency = (time.perf_counter() - start) * 1000

        response_data = {
            "answer": result.answer,
            "sources": result.sources,
            "model": result.model,
            "retrieval_count": len(candidates),
            "latency_ms": round(latency, 2),
            "tokens_used": result.total_tokens,
        }

        # 5. Store in cache
        cache.set_response(req.question, cache_params, response_data)

        return QueryResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("query_error")
        raise HTTPException(status_code=500, detail=str(e))


# ── /agent/query ──────────────────────────────────────────
@router.post("/agent/query", response_model=QueryResponse)
async def agent_query(req: QueryRequest):
    """Agentic RAG pipeline: query decomposition → multi-hop retrieval → synthesis."""
    logger.info("!!! AGENT_QUERY_REQUEST_RECEIVED !!!", payload=req.dict())
    start = time.perf_counter()

    try:
        logger.info("agent_query_start", question=req.question)
        agent = get_agent_searcher()
        answer = await agent.search(req.question)

        latency = (time.perf_counter() - start) * 1000
        logger.info("agent_query_success", latency_ms=latency)

        return QueryResponse(
            answer=answer,
            sources=[],
            model="agent-sub-question",
            retrieval_count=0,
            latency_ms=round(latency, 2),
            tokens_used=0,
        )

    except Exception as e:
        import traceback

        error_msg = traceback.format_exc()
        logger.error("agent_query_error", error=str(e), traceback=error_msg)
        raise HTTPException(status_code=500, detail=f"Agent Error: {str(e)}\n\n{error_msg}")


# ── /chat/stream ──────────────────────────────────────────
@router.post("/chat/stream")
async def chat_stream(req: QueryRequest):
    """Stream response back via SSE."""
    try:
        guardrails = Guardrails()
        is_safe, msg = guardrails.check_query(req.question)
        if not is_safe:

            def error_stream():
                yield f"data: {msg}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(error_stream(), media_type="text/event-stream")

        searcher = _get_hybrid_searcher()
        reranker = CohereReranker()
        generator = ResponseGenerator()
        router_module = QueryRouter()
        hyde = HyDEGenerator()
        memory = MemoryManager()

        # 1. Route Query
        decision = router_module.route_query(req.question)

        # 2. Determine Scope (Active Filename)
        # Sequence: 1. Explicit filter from UI, 2. Regex from query, 3. Session memory
        active_filename = None
        if req.filters and "filename" in req.filters:
            active_filename = req.filters["filename"]

        if not active_filename:
            import re

            match = re.search(r"([\w\-. ]+\.pdf)", req.question, re.IGNORECASE)
            if match:
                active_filename = match.group(1).strip()

        if not active_filename:
            active_filename = memory.get_last_filename(req.session_id)

        if active_filename:
            memory.save_last_filename(req.session_id, active_filename)

        # 3. Detect Specialized Requests
        is_generic_desc = any(
            t in req.question.lower()
            for t in ["description", "overview", "what is this", "tell me about", "about the file"]
        )
        is_analysis_request = any(
            t in req.question.lower()
            for t in [
                "more details",
                "bigger description",
                "exhaustive",
                "technical breakdown",
                "detailed explanation",
                "tell me more",
            ]
        )

        system_prompt_override = None

        # 4. Handle Fast-Path for Simple Descriptions
        if active_filename and is_generic_desc and not is_analysis_request:
            metadata = memory.get_document_metadata(active_filename)
            if metadata and metadata.get("short_description"):
                logger.info("metadata_fast_path", filename=active_filename)

                def meta_stream():
                    yield f"data: {metadata['short_description']}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(meta_stream(), media_type="text/event-stream")

        # 5. Configure RAG Scope
        search_filters = req.filters or {}
        top_k = req.top_k

        if is_analysis_request and active_filename:
            logger.info("deep_analysis_triggered", filename=active_filename)
            top_k = 50
            system_prompt_override = ANALYSIS_SYSTEM_PROMPT
            search_filters["filename"] = active_filename
        elif active_filename and "filename" not in search_filters:
            search_filters["filename"] = active_filename

        candidates = []
        reranked = []

        if decision == "RAG":
            search_query = hyde.generate(req.question)

            multi_searcher = MultiQuerySearcher(searcher)
            logger.info(
                "retrieval_debug",
                searcher_type=type(searcher).__name__,
                client_type=type(searcher.vector.client).__name__,
                has_search=hasattr(searcher.vector.client, "search"),
            )
            if req.multi_query_enabled:
                candidates = multi_searcher.search(query=search_query, top_k=top_k, filters=search_filters)
            else:
                candidates = searcher.search(query=search_query, top_k=top_k, filters=search_filters)

            if candidates:
                reranked = reranker.rerank(req.question, candidates, top_n=req.top_n)

        # 6. Stream generation
        def sse_wrapper():
            full_response = []
            try:
                for chunk in generator.generate_stream(
                    req.question, reranked, temperature=req.temperature, system_prompt=system_prompt_override
                ):
                    full_response.append(chunk)
                    yield f"data: {chunk}\n\n"

                # Save to persistent storage
                memory.save_message(req.session_id, "user", req.question)
                memory.save_message(req.session_id, "assistant", "".join(full_response))

                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error("sse_error", error=str(e))
                yield f"data: Error: {str(e)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(sse_wrapper(), media_type="text/event-stream")

    except Exception as e:
        logger.exception("stream_error")
        raise HTTPException(status_code=500, detail=str(e))


# ── /ingest ───────────────────────────────────────────────
@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """Ingest documents: load → chunk → embed → store in Qdrant."""
    global _bm25_built
    start = time.perf_counter()

    try:
        directory = Path(req.directory)
        if not directory.is_dir():
            raise HTTPException(status_code=400, detail=f"Directory not found: {req.directory}")

        loader = DocumentLoader()
        chunker = SemanticChunker()
        pipeline = EmbeddingPipeline()

        # 1. Load
        documents = loader.load_directory(directory, extensions=req.extensions)
        if not documents:
            raise HTTPException(status_code=400, detail="No supported documents found.")

        # 2. Chunk
        chunks = chunker.chunk_documents(documents)

        # 3. Embed & store
        pipeline.ensure_collection()
        stored = pipeline.ingest(chunks)

        # Reset BM25 index so it's rebuilt on next query
        _bm25_built = False

        latency = (time.perf_counter() - start) * 1000

        return IngestResponse(
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            vectors_stored=stored,
            latency_ms=round(latency, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("ingest_error")
        raise HTTPException(status_code=500, detail=str(e))


# ── /upload ───────────────────────────────────────────────
@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a PDF, save it, summarize it, and ingest it into RAG."""
    start = time.perf_counter()
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        # 1. Save file
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / file.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Extract and Summarize
        loader = DocumentLoader()
        documents = loader.load_file(file_path)
        if not documents:
            raise HTTPException(status_code=400, detail="Failed to extract text from PDF.")

        full_text = documents[0].text
        generator = ResponseGenerator()
        short_desc = generator.generate_summary(full_text, length="short")
        long_desc = generator.generate_summary(full_text, length="long")

        # 3. Store Metadata in Firestore & Track as Last Referened
        memory = MemoryManager()
        metadata_payload = {
            "filename": file.filename,
            "path": str(file_path),
            "short_description": short_desc,
            "long_description": long_desc,
        }
        memory.save_document_metadata(file.filename, metadata_payload)
        memory.save_last_filename("default_session", file.filename)

        # 4. Ingest into RAG (Qdrant)
        chunker = SemanticChunker()
        pipeline = EmbeddingPipeline()
        chunks = chunker.chunk_documents(documents)
        pipeline.ensure_collection()
        stored = pipeline.ingest(chunks)

        # Reset BM25
        global _bm25_built
        _bm25_built = False

        latency = (time.perf_counter() - start) * 1000
        return {
            "filename": file.filename,
            "vectors_stored": stored,
            "short_description": short_desc,
            "latency_ms": round(latency, 2),
        }

    except Exception as e:
        logger.exception("upload_error")
        raise HTTPException(status_code=500, detail=str(e))


# ── /stats ───────────────────────────────────────────────
@router.get("/stats")
async def stats():
    """Return Qdrant collection statistics — verify ingestion worked."""
    try:
        from config.qdrant_client import get_qdrant_client

        settings = get_settings()
        client = get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]

        if settings.qdrant_collection not in collections:
            return {
                "collection": settings.qdrant_collection,
                "status": "not_created",
                "points_count": 0,
                "vectors_count": 0,
                "message": "No collection found. Run ingestion first.",
            }

        info = client.get_collection(settings.qdrant_collection)
        return {
            "collection": settings.qdrant_collection,
            "status": info.status.value,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "vector_dimension": info.config.params.vectors.size,
            "distance_metric": str(info.config.params.vectors.distance),
        }
    except Exception as e:
        logger.exception("stats_error")
        raise HTTPException(status_code=500, detail=str(e))


# ── /evaluate ─────────────────────────────────────────────
@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_endpoint(req: EvaluateRequest):
    """Run RAGAS evaluation on a prepared dataset."""
    try:
        from evaluation.dataset_builder import EvaluationDatasetBuilder
        from evaluation.ragas_evaluator import RAGASEvaluator

        builder = EvaluationDatasetBuilder()
        builder.load(req.dataset_path)
        dataset = builder.to_ragas_dataset()

        evaluator = RAGASEvaluator()
        report = evaluator.evaluate(dataset)

        return EvaluateResponse(
            scores=report["scores"],
            num_samples=report["num_samples"],
            passed=report["passed"],
            evaluation_time_s=report["evaluation_time_s"],
        )

    except Exception as e:
        logger.exception("evaluate_error")
        raise HTTPException(status_code=500, detail=str(e))
