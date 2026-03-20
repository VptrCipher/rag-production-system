# RAG Production System — Architecture Document

> **Version**: 1.0.0  
> **Date**: March 2026  
> **Author**: Senior AI Infrastructure Engineering

---

## Table of Contents

1. [System Architecture](#section-1--system-architecture)
2. [Project Repository Structure](#section-2--project-repository-structure)
3. [Data Ingestion Pipeline](#section-3--data-ingestion-pipeline)
4. [Vector Database Design](#section-4--vector-database-design)
5. [Hybrid Retrieval](#section-5--hybrid-retrieval)
6. [Reranking Layer](#section-6--reranking-layer)
7. [LLM Generation](#section-7--llm-generation)
8. [Evaluation Layer](#section-8--evaluation-layer)
9. [Experimentation Framework](#section-9--experimentation-framework)
10. [Production Optimization](#section-10--production-optimization)
11. [Deployment](#section-11--deployment)
12. [End-to-End Workflow](#section-12--end-to-end-workflow)
13. [Future Improvements](#section-13--future-improvements)

---

## SECTION 1 — System Architecture

### Architecture Explanation

The system is a **modular, pipeline-based RAG architecture** with six distinct layers:

1. **Ingestion Layer** — Loads documents (PDF, HTML, Markdown, TXT), cleans text, splits into semantically meaningful chunks, generates dense embeddings, and stores them in Qdrant with rich metadata payloads.

2. **Retrieval Layer** — Combines two complementary retrieval strategies:
   - *Dense vector search* (semantic similarity via cosine distance in Qdrant)
   - *Sparse BM25 search* (lexical/keyword matching via rank_bm25)
   
   Results are merged using **Reciprocal Rank Fusion (RRF)**, which is distribution-agnostic and robust to score miscalibration between retrievers.

1.  **Ingestion Layer** — Loads documents (PDF, HTML, Markdown, TXT), cleans text, splits into semantically meaningful chunks, generates dense embeddings, and stores them in Qdrant with rich metadata payloads.

2.  **Retrieval Layer** — Combines two complementary retrieval strategies:
    -   *Dense vector search* (semantic similarity via cosine distance in Qdrant)
    -   *Sparse BM25 search* (lexical/keyword matching via rank_bm25)

    Results are merged using **Reciprocal Rank Fusion (RRF)**, which is distribution-agnostic and robust to score miscalibration between retrievers.

3.  **Reranking Layer** — A Cohere cross-encoder reranks the top-30 candidates down to top-5. Cross-encoders jointly attend to query-document pairs, yielding much higher relevance accuracy than bi-encoder retrieval alone.

4.  **Generation Layer** — Feeds the reranked context into an OpenAI GPT model with grounded, citation-aware prompts. The system prompt enforces strict grounding: the LLM must cite `[Source N]` for every claim and refuse to answer if context is insufficient.

5.  **Evaluation Layer** — RAGAS computes four metrics on a curated or synthetic evaluation dataset:
    -   Faithfulness, Answer Correctness, Context Recall, Context Precision
    -   CI/CD threshold assertions halt deployment if quality degrades.

6.  **Agentic Reasoning Layer** — Utilizes LlamaIndex `SubQuestionQueryEngine` to handle multi-hop queries. It decomposes complex questions into logical sub-tasks, retrieves across multiple sources, and synthesizes a final cohesive answer.

7.  **Multimodal Ingestion Layer** — Handles image data (`.png`, `.jpg`, `.jpeg`) using OCR to extract textual context, ensuring visual documentation is fully searchable.

8.  **Observability Layer** — Integrated Arize Phoenix for deep tracing of retrieval spans, latency breakdowns, and token cost attribution.

9.  **Isolation & Scoping Layer** — Implements strict document isolation via metadata filters to prevent cross-document information "spillage".

10. **Thread-Safe Vector Store Access** — Implements a **Thread-Safe Singleton Pattern** with `threading.Lock` for the Qdrant client. This prevents "AlreadyLocked" errors when running in multi-threaded environments or during rapid ingestion/query cycles.

11. **Dynamic Intent Dispatcher** — Analyzes user queries for depth intent. Standard queries use metadata-enriched summaries, while requests for "detailed explanations" trigger a deep RAG path with increased `top_k` and specialized analyst prompts.

12. **Streaming Post-Processor** — A backend-side cleanup layer that corrects Markdown formatting and normalizes citation clusters in real-time as the LLM streams.

### ASCII Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER / CLIENT                                │
│                    POST /api/v1/query                                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      FastAPI Gateway                                 │
│           (CORS · Auth · Rate Limiting · Logging)                    │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
  ┌───────────────────┐     ┌───────────────────┐
  │  Dense Retriever   │     │  BM25 Retriever    │
  │  (Qdrant ANN)      │     │  (rank_bm25)       │
  │  top-30 candidates │     │  top-30 candidates │
  └────────┬──────────┘     └────────┬──────────┘
           │                         │
           └────────┬────────────────┘
                    ▼
          ┌──────────────────┐
          │ Reciprocal Rank  │
          │ Fusion (RRF)     │
          │ merged top-30    │
          └────────┬─────────┘
                   ▼
          ┌──────────────────┐
          │ Cohere Reranker  │
          │ (cross-encoder)  │
          │ top-30 → top-5   │
          └────────┬─────────┘
                   ▼
          ┌──────────────────┐
          │ LLM Generation   │
          │ (OpenAI GPT-4o)  │
          │ grounded + cited  │
          └────────┬─────────┘
                   ▼
          ┌──────────────────┐
          │ Response + Meta  │
          │ answer, sources, │
          │ latency, tokens  │
          └──────────────────┘

════════════════════════════════════════════
         OFFLINE / BATCH PIPELINES
════════════════════════════════════════════

  ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌─────────┐
  │ Document │    │  Text    │    │  Chunking    │    │Embedding│
  │ Loaders  │───▶│ Cleaning │───▶│ (512 tokens) │───▶│ Pipeline│
  │ PDF/HTML │    │          │    │ + overlap 64 │    │ OpenAI  │
  └──────────┘    └──────────┘    └──────────────┘    └────┬────┘
                                                          │
                                                          ▼
                                                   ┌───────────┐
                                                   │  Qdrant   │
                                                   │  Vector   │
                                                   │  Store    │
                                                   └───────────┘

  ┌──────────────┐    ┌──────────┐    ┌───────────────┐
  │ Eval Dataset │───▶│  RAGAS   │───▶│ Metrics Report│
  │ (QA pairs)   │    │ Evaluate │    │ + CI Checks   │
  └──────────────┘    └──────────┘    └───────────────┘
```

### Data Flow

```
Documents (PDF/HTML/MD/TXT/IMG)
    ↓ load (OCR for Images)
Raw Text + Metadata
    ↓ clean
Clean Text
    ↓ chunk (512 tokens, 64 overlap)
Chunks with Metadata
    ↓ embed (OpenAI / Local BGE)
Vectors (1536-dim / 1024-dim)
    ↓ upsert
Qdrant Collection
    ↓ query time
    ├── Standard: Hybrid Search → Rerank → Generate
    └── Agentic: Decompose → Sub-queries → Parallel Retrieval → Aggregate
    ↓ 
Final Answer with [Source N] Citations
    ↓ 
Streaming Post-Processor (Clean Markdown + Citation Normalization)
    ↓ instrument
Arize Phoenix (OTLP Spans)
```

---

## SECTION 2 — Project Repository Structure

```
rag-production-system/
│
├── config/
│   ├── __init__.py
│   └── settings.py              # Pydantic Settings (env-vars)
│
├── ingestion/
│   ├── __init__.py
│   ├── loaders.py               # PDF, HTML, MD, TXT loaders
│   ├── chunking.py              # Sentence-window chunking
│   └── embedding_pipeline.py    # Batch embed + Qdrant upsert
│
├── retrieval/
│   ├── __init__.py
│   ├── vector_search.py         # Qdrant dense retrieval
│   ├── bm25_search.py           # BM25 sparse retrieval
│   └── hybrid_search.py         # RRF fusion
│
├── reranking/
│   ├── __init__.py
│   └── cohere_rerank.py         # Cross-encoder reranking
│
├── generation/
│   ├── __init__.py
│   ├── prompt_templates.py      # Citation-aware prompts
│   └── response_generator.py    # LLM response + metadata
│
├── evaluation/
│   ├── __init__.py
│   ├── ragas_evaluator.py       # RAGAS metrics + CI checks
│   └── dataset_builder.py       # Eval dataset construction
│
├── experiments/
│   └── retrieval_experiments.py  # A/B experiment runner
│
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app
│   └── routes.py                # /query, /ingest, /evaluate
│
├── scripts/
│   └── ingest_documents.py      # CLI bulk ingestion
│
├── docker/
│   ├── Dockerfile               # Python service image
│   └── docker-compose.yml       # rag-api + qdrant
│
├── data/                        # Document storage
├── logs/                        # Application logs
├── .env.example                 # Environment template
├── requirements.txt             # Python dependencies
├── README.md                    # Quick start guide
└── ARCHITECTURE.md              # This document
```

---

## SECTION 3 — Data Ingestion Pipeline

### Chunking Strategy

| Parameter | Value | Rationale |
|---|---|---|
| **Chunk size** | 512 tokens (~2048 chars) | Balances semantic density with retrieval precision. Smaller chunks lose context; larger ones dilute the embedding signal. |
| **Chunk overlap** | 64 tokens (~256 chars) | Preserves boundary context. Prevents information loss when relevant sentences span chunk boundaries. |
| **Splitting method** | Sentence-aware recursive | Avoids mid-sentence breaks which degrade embedding quality and retrieval coherence. Falls back to character-level hard-splitting for oversized single sentences. |

### Metadata Schema

Every chunk carries a rich payload stored alongside its vector in Qdrant:

```json
{
  "source":          "/abs/path/to/document.pdf",
  "filename":        "document.pdf",
  "file_type":       "pdf",
  "file_hash":       "sha256:abc123...",
  "file_size_bytes": 204800,
  "ingested_at":     "2026-03-16T10:00:00Z",
  "chunk_index":     3,
  "total_chunks":    12,
  "chunk_char_len":  1987,
  "text":            "The actual chunk text content..."
}
```

### Pipeline Implementation

See `ingestion/loaders.py`, `ingestion/chunking.py`, `ingestion/embedding_pipeline.py`.

```python
# Usage
from ingestion import DocumentLoader, DocumentChunker, EmbeddingPipeline

loader   = DocumentLoader()
chunker  = DocumentChunker()
pipeline = EmbeddingPipeline()

docs   = loader.load_directory("data/raw")
chunks = chunker.chunk_documents(docs)
pipeline.ensure_collection()
pipeline.ingest(chunks)
```

---

## SECTION 4 — Vector Database Design

### Qdrant Collection Configuration

| Setting | Value | Rationale |
|---|---|---|
| **Embedding size** | 1536 dimensions | OpenAI `text-embedding-3-small` output size |
| **Distance metric** | Cosine | Standard for normalised embeddings; rotation-invariant |
| **Index type** | HNSW (default) | Sub-linear search with high recall at scale |
| **HNSW M** | 16 (default) | Connections per node; balances speed vs. recall |
| **EF construct** | 100 (default) | Build-time quality; higher = better index, slower build |

### Payload Structure

```json
{
  "text":            "chunk text",
  "source":          "path/to/file.pdf",
  "filename":        "file.pdf",
  "file_type":       "pdf",
  "file_hash":       "sha256:...",
  "chunk_index":     0,
  "total_chunks":    10,
  "ingested_at":     "2026-03-16T10:00:00Z"
}
```

### Metadata Filters

Qdrant supports payload filtering during search:

```python
# Filter by file type
results = client.search(
    collection_name="rag_documents",
    query_vector=embedding,
    query_filter=Filter(must=[
        FieldCondition(key="file_type", match=MatchValue(value="pdf"))
    ]),
    limit=30,
)
```

### Scalability

- **Sharding**: Qdrant supports automatic sharding for horizontal scaling.
- **Replicas**: Configure replicas for read-heavy workloads.
- **Quantization**: Enable scalar/product quantization for memory reduction (30-50% savings).
- **Snapshot**: Built-in snapshot/backup support.

---

## SECTION 5 — Hybrid Retrieval

### Retrieval Pipeline

```
User Query
    ├── embed query → Dense Vector Search (Qdrant, cosine, top-30)
    └── tokenise query → BM25 Search (rank_bm25, top-30)
    ↓
Reciprocal Rank Fusion (weighted)
    ↓
Merged & Ranked Candidates (top-30)
```

### Merging Strategy: Reciprocal Rank Fusion (RRF)

RRF is chosen over simple score normalisation because:
- **Distribution-independent**: Works regardless of how each retriever scores.
- **No calibration needed**: BM25 scores and cosine similarity are on different scales.
- **Proven effective**: Used in production by major search engines.

Formula:

```
RRF(d) = Σ  weight_i / (k + rank_i(d))
```

Where `k = 60` is a standard constant that prevents top-ranked documents from dominating.

### Weight Configuration

| Retriever | Default Weight | Rationale |
|---|---|---|
| Vector | 0.7 | Captures semantic similarity (paraphrases, synonyms) |
| BM25 | 0.3 | Captures exact keyword matches, acronyms, proper nouns |

### Implementation

See `retrieval/hybrid_search.py` — the `HybridSearcher` class orchestrates both retrievers and applies RRF fusion.

---

## SECTION 6 — Reranking Layer

### Cross-Encoder vs Bi-Encoder

| Aspect | Bi-Encoder (Stage 1) | Cross-Encoder (Stage 2) |
|---|---|---|
| **Architecture** | Encode query and doc independently | Jointly attend to query + doc |
| **Speed** | Fast (~20-50ms for 30 docs) | Slow (~200-400ms for 30 docs) |
| **Accuracy** | Good | Excellent |
| **Use case** | Candidate generation | Candidate reranking |

### Pipeline

```
Query
  → retrieve top 30 (hybrid, ~50ms)
  → Cohere Rerank (cross-encoder, ~300ms)
  → keep top 5
  → LLM generation
```

### Latency Tradeoffs

| Configuration | Candidates | Rerank Latency | Quality |
|---|---|---|---|
| No reranking | top-5 direct | 0ms | Lower |
| Rerank 15 → 5 | 15 | ~150ms | Good |
| **Rerank 30 → 5** | **30** | **~300ms** | **Best** |
| Rerank 50 → 5 | 50 | ~500ms | Diminishing returns |

### Implementation

See `reranking/cohere_rerank.py` — the `CohereReranker` class wraps the Cohere API.

---

## SECTION 7 — LLM Generation

### Design Requirements

1. **Grounded responses** — Answer ONLY from provided context.
2. **No hallucinations** — Refuse if context is insufficient.
3. **Citation support** — Every claim references `[Source N]`.
4. **Context formatting** — Numbered chunks with source metadata.

### Prompt Template

```
SYSTEM:
You are a precise, helpful research assistant. You answer questions
using ONLY the provided context documents. Follow these rules:
1. Base every claim on sources. Cite using [Source N].
2. If context is insufficient, say so explicitly.
3. Never fabricate information. Never use prior knowledge.
4. Structure your answer in clear Markdown.

USER:
## Context Documents
[Source 1] (file: report.pdf, chunk 3)
The retrieval augmented generation pattern combines ...

[Source 2] (file: guide.md, chunk 1)
Hybrid search merges dense vector and sparse keyword ...

---
## Question
What is retrieval augmented generation?

---
Provide a well-structured answer with citations [Source N].
```

### Example Input/Output

**Input**: `"What is retrieval augmented generation?"`

**Output**:
> Retrieval-Augmented Generation (RAG) is an architecture that combines a retrieval component with a generative language model [Source 1]. The system first retrieves relevant documents from a knowledge base using hybrid search, which merges dense vector similarity with sparse keyword matching [Source 2]. The retrieved documents are then used as context for the language model to generate grounded, accurate responses [Source 1].

### Implementation

See `generation/prompt_templates.py` and `generation/response_generator.py`.

---

## SECTION 8 — Evaluation Layer

### RAGAS Metrics

| Metric | What it Measures | Target |
|---|---|---|
| **Faithfulness** | Is every claim in the answer supported by the context? | ≥ 0.70 |
| **Answer Correctness** | Does the answer match the ground truth? | ≥ 0.60 |
| **Context Recall** | Were all needed context pieces retrieved? | ≥ 0.70 |
| **Context Precision** | Are retrieved contexts actually relevant? | ≥ 0.70 |

### Evaluation Dataset Design

Each sample contains:

```json
{
  "question":     "What chunking strategy works best for RAG?",
  "answer":       "Generated by the pipeline...",
  "contexts":     ["chunk1 text...", "chunk2 text..."],
  "ground_truth": "Human-verified reference answer..."
}
```

**Dataset sources**:
- **Manual**: Domain experts create 50-100 curated QA pairs.
- **Synthetic**: LLM generates questions from ingested document chunks (see `dataset_builder.py`).

### Automated Testing Pipeline

```
1. Load evaluation dataset (JSON)
2. For each question:
   a. Run full RAG pipeline (retrieve → rerank → generate)
   b. Record answer + retrieved contexts
3. Run RAGAS evaluate()
4. Assert all metrics ≥ thresholds
5. Log results to JSON
6. Fail CI if any threshold breached
```

### Implementation

See `evaluation/ragas_evaluator.py` and `evaluation/dataset_builder.py`.

---

## SECTION 9 — Experimentation Framework

### Experiments to Compare

| Dimension | Configurations |
|---|---|
| **Chunk sizes** | 256, 512, 1024 tokens |
| **Embedding models** | text-embedding-3-small, text-embedding-3-large |
| **Retrieval strategies** | Vector-only, BM25-only, Hybrid (70/30), Hybrid (50/50) |
| **Rerankers** | No reranking, top-3, top-5, top-10 |

### Experiment Output Format

Each experiment produces a JSON report:

```json
{
  "name": "chunk_512",
  "config": { "chunk_size": 512, "vector_weight": 0.7, ... },
  "scores": {
    "faithfulness": 0.85,
    "answer_correctness": 0.72,
    "context_recall": 0.88,
    "context_precision": 0.79
  },
  "avg_latency_ms": 1250.3,
  "num_samples": 50
}
```

### Implementation

See `experiments/retrieval_experiments.py` — includes pre-defined experiment sets: `chunk_size_experiments()`, `retrieval_strategy_experiments()`, `reranker_experiments()`.

---

## SECTION 10 — Production Optimization

### Latency Optimization

| Technique | Impact | Complexity |
|---|---|---|
| **Embedding caching** | Avoid re-embedding repeated queries | Low |
| **Qdrant quantization** | ~2x faster search, ~30% less memory | Low |
| **Async retrieval** | Parallel vector + BM25 searches | Medium |
| **Reduced candidate set** | 20 instead of 30 for reranking | Low |
| **Streaming responses** | First token appears faster | Medium |

### Caching Strategy

```
Query Cache (LRU, TTL=1hr):
  - hash(query + filters) → cached response
  - Invalidate on document ingestion
  
Embedding Cache (LRU, TTL=24hr):
  - hash(query_text) → embedding vector
  - Reduces OpenAI API calls by ~60%
```

### Cost Control

| Component | Cost Driver | Mitigation |
|---|---|---|
| OpenAI Embeddings | Per-token pricing | Cache embeddings; batch at ingestion |
| OpenAI GPT | Per-token I/O pricing | Limit context to top-5 chunks; cap max_tokens |
| Cohere Rerank | Per-search pricing | Reduce candidate set; cache frequent queries |
| Qdrant | Compute + storage | Use quantization; archive old collections |

### Observability

```
Structured Logging (structlog):
  - Every pipeline stage logs: operation, latency, counts
  - JSON format for log aggregation (ELK, Loki)

Metrics (Prometheus):
  - query_latency_seconds (histogram)
  - retrieval_candidates_count (gauge)
  - generation_tokens_total (counter)
  - evaluation_scores (gauge per metric)

Health Checks:
  - /health endpoint (FastAPI)
  - Docker HEALTHCHECK
  - Qdrant healthz
```

### Monitoring Dashboard

Key metrics to track:
- **P50/P95/P99 query latency**
- **Retrieval hit rate** (% queries with ≥1 relevant result)
- **Faithfulness score** (rolling 7-day average)
- **Token consumption** (daily/weekly)
- **Error rate** by endpoint

---

## SECTION 11 — Deployment

### Docker Compose Setup

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.12.5
    ports: ["6333:6333", "6334:6334"]
    volumes: [qdrant_data:/qdrant/storage]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]

  rag-api:
    build: { context: .., dockerfile: docker/Dockerfile }
    ports: ["8000:8000"]
    env_file: [../.env]
    environment:
      QDRANT_HOST: qdrant
    depends_on:
      qdrant: { condition: service_healthy }
```

### Deployment Commands

```bash
# Start all services
cd docker && docker compose up -d

# View logs
docker compose logs -f rag-api

# Scale API (for load testing)
docker compose up -d --scale rag-api=3

# Stop
docker compose down
```

---

## SECTION 12 — End-to-End Workflow

```
 ┌─────────────────────────────────────────────────────────┐
 │                 FULL SYSTEM FLOW                        │
 └─────────────────────────────────────────────────────────┘

 1. User submits query via POST /api/v1/query
    │
 2. ├── Dense Vector Search (Qdrant) ──── top-30 candidates
    └── BM25 Keyword Search ────────────── top-30 candidates
    │
 3. Reciprocal Rank Fusion ──────────────── merged top-30
    │
 4. Cohere Reranker (cross-encoder) ─────── top-5 most relevant
    │
 5. Prompt Construction
    │  - Format contexts as [Source 1], [Source 2], etc.
    │  - Inject system prompt (grounding rules)
    │  - Inject user question
    │
 6. LLM Generation (OpenAI GPT-4o)
    │  - Temperature: 0.1 (deterministic)
    │  - Max tokens: 1024
    │
 7. Response Assembly
    │  - Answer text with [Source N] citations
    │  - Source metadata (filename, chunk, score)
    │  - Performance metadata (latency, tokens)
    │
 8. Evaluation Logging
    │  - Log query, answer, contexts for offline evaluation
    │  - Feed into RAGAS pipeline for quality monitoring
    │
 9. Return structured JSON response to user
```

---

## SECTION 13 — Future Improvements

### Query Rewriting

```
User query → LLM rewrites into 3-5 sub-queries → retrieve for each → merge
```
- Improves recall for complex or ambiguous queries.
- Techniques: HyDE (Hypothetical Document Embeddings), Step-Back Prompting.

### Multi-Hop Retrieval

```
Query → retrieve → extract entities → follow-up retrieval → synthesize
```
- For questions requiring information across multiple documents.
- Implement iterative retrieval with entity linking.

### Agentic RAG

```
Query → Agent decides:
  - Which tool to use (search, calculator, API)
  - Whether to retry with different query
  - When enough context is gathered
```
- LlamaIndex agents with tool-use capabilities.
- Self-correcting retrieval with reflection.

### Reinforcement Feedback Loops

```
User feedback (👍/👎) → fine-tune:
  - Retrieval weights
  - Reranking thresholds
  - Prompt templates
```
- RLHF for the generation model.
- Online learning for retrieval weight optimization.
- A/B testing framework for prompt variants.

### Long Context Memory

```
Conversation history → sliding window + summarization → persistent memory
```
- Maintain context across multi-turn conversations.
- Hierarchical memory: recent turns (full), older turns (summarized).
- Vector-indexed conversation memory for retrieval.

### Additional Improvements

| Area | Improvement | Impact |
|---|---|---|
| **Ingestion** | Table/chart extraction from PDFs | Higher recall on structured data |
| **Embeddings** | Fine-tuned domain-specific embeddings | 10-20% relevance improvement |
| **Retrieval** | Qdrant native sparse vectors | Eliminate in-memory BM25 index |
| **Generation** | Streaming responses (SSE) | Better UX, lower perceived latency |
| **Evaluation** | Human-in-the-loop feedback collection | Ground truth quality improvement |
| **Security** | PII detection + redaction in chunks | Compliance with data regulations |
| **Scale** | Kubernetes deployment with HPA | Auto-scaling under load |
