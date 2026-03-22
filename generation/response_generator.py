"""
LLM response generation with context formatting, citation support, and
structured logging.

Supports:
  • Groq (free tier) — llama-3.3-70b-versatile  (set GROQ_API_KEY in .env)
  • OpenAI           — gpt-4o / gpt-4o-mini      (set OPENAI_API_KEY in .env)

Groq is used when GROQ_API_KEY is set. OpenAI is the fallback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import structlog

from config import get_settings
from generation.prompt_templates import RAGPromptTemplates
from retrieval.vector_search import SearchResult

logger = structlog.get_logger(__name__)


@dataclass
class GenerationResult:
    """Container for a generated response with metadata."""

    answer: str
    model: str
    contexts_used: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    sources: List[Dict[str, Any]] = field(default_factory=list)


class ResponseGenerator:
    """Generate grounded LLM responses from retrieved context.

    Usage::

        generator = ResponseGenerator()
        result = generator.generate(query="What is RAG?", contexts=search_results)
        print(result.answer)

    Example Input/Output
    --------------------
    **Input query**: "What is retrieval-augmented generation?"

    **Context** (2 chunks from ingested documents):
      [Source 1] RAG combines retrieval with generative models …
      [Source 2] The architecture includes a retriever and a generator …

    **Output**:
      Retrieval-Augmented Generation (RAG) is an architecture that
      combines a retrieval component with a generative language model
      [Source 1]. The system retrieves relevant documents and feeds
      them as context to the generator [Source 2].
    """

    def __init__(self, model: Optional[str] = None):
        self.settings = get_settings()
        self.templates = RAGPromptTemplates()

        # ── Prefer Groq (free) when a key is configured ───────
        if self.settings.groq_api_key:
            from groq import Groq

            self.client = Groq(api_key=self.settings.groq_api_key)
            self.model = model or (
                "llama-3.3-70b-versatile" if self.settings.llm_model == "gpt-4o" else self.settings.llm_model
            )
            self._backend = "groq"
            logger.info("llm_backend", backend="groq", model=self.model)
        elif self.settings.openai_api_key:
            import openai

            self.client = openai.OpenAI(api_key=self.settings.openai_api_key)
            self.model = model or self.settings.llm_model
            self._backend = "openai"
            logger.info("llm_backend", backend="openai", model=self.model)
        else:
            self.client = None
            self.model = "context-echo"
            self._backend = "echo"
            logger.warning("no_llm_key_configured", hint="Set GROQ_API_KEY or OPENAI_API_KEY in .env")

    def _echo_generate(self, query: str, contexts: List[SearchResult]) -> GenerationResult:
        """Return retrieved context directly when no LLM is configured."""
        lines = [f"**Query**: {query}\n"]
        lines.append("**Retrieved Context** (no LLM key configured — showing raw retrieval results):\n")
        for i, ctx in enumerate(contexts, 1):
            fname = ctx.metadata.get("filename", "unknown")
            lines.append(f"\n[Source {i}] ({fname}):\n{ctx.text}\n")
        answer = "\n".join(lines)
        sources = [
            {
                "source_id": i,
                "filename": ctx.metadata.get("filename", "unknown"),
                "chunk_index": ctx.metadata.get("chunk_index", "?"),
                "relevance_score": ctx.score,
            }
            for i, ctx in enumerate(contexts, 1)
        ]
        return GenerationResult(
            answer=answer,
            model="context-echo (no LLM key)",
            contexts_used=len(contexts),
            sources=sources,
        )

    def _clean_response(self, text: str) -> str:
        """Clean up LLM artifacts like malformed citations or extra spaces."""
        import re

        # 1. Clean up [Source N , Source M] -> [Source N] [Source M]
        text = re.sub(r"\[Source\s*(\d+)\s*,\s*Source\s*(\d+)\]", r"[Source \1] [Source \2]", text)

        # 2. Clean up [Source N , ] or [Source , N]
        text = re.sub(r"\[Source\s*(\d+)\s*,\s*\]", r"[Source \1]", text)
        text = re.sub(r"\[Source\s*,\s*(\d+)\]", r"[Source \1]", text)
        text = re.sub(r"\[,\s*Source\s*(\d+)\]", r"[Source \1]", text)

        # 3. Clean up single empty source [Source , ]
        text = re.sub(r"\[Source\s*,\s*\]", "", text)

        # 4. Ensure space before opening bracket and after closing bracket
        text = re.sub(r"([A-Za-z0-9])(\[Source)", r"\1 \2", text)
        text = re.sub(r"(\])([A-Za-z0-9])", r"\1 \2", text)

        return text.strip()

    def generate(
        self,
        query: str,
        contexts: List[SearchResult],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a grounded answer with citations.

        Parameters
        ----------
        query : str
            User question.
        contexts : list[SearchResult]
            Reranked retrieval results.
        temperature : float
            LLM temperature (low = more deterministic).
        max_tokens : int
            Maximum response length.
        """
        # ── Echo mode: no LLM key configured ─────────────────
        if self._backend == "echo":
            return self._echo_generate(query, contexts)

        # Format context block
        texts = [c.text for c in contexts]
        metadatas = [c.metadata for c in contexts]
        context_block = self.templates.format_contexts(texts, metadatas)

        # Build chat messages
        messages = self.templates.build_messages(query, context_block, system_prompt=kwargs.get("system_prompt"))

        # Call LLM
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        usage = response.usage

        # Build source references
        sources = []
        for i, ctx in enumerate(contexts, start=1):
            sources.append(
                {
                    "source_id": i,
                    "filename": ctx.metadata.get("filename", "unknown"),
                    "chunk_index": ctx.metadata.get("chunk_index", "?"),
                    "relevance_score": ctx.score,
                }
            )

        result = GenerationResult(
            answer=self._clean_response(choice.message.content or ""),
            model=self.model,
            contexts_used=len(contexts),
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            latency_ms=round(latency_ms, 2),
            sources=sources,
        )

        logger.info(
            "generation_complete",
            model=self.model,
            latency_ms=result.latency_ms,
            tokens=result.total_tokens,
            contexts=result.contexts_used,
        )
        return result

    def generate_stream(
        self,
        query: str,
        contexts: List[SearchResult],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream a grounded answer using SSE."""
        if self._backend == "echo":
            yield self._echo_generate(query, contexts).answer
            return

        texts = [c.text for c in contexts]
        metadatas = [c.metadata for c in contexts]
        context_block = self.templates.format_contexts(texts, metadatas)
        messages = self.templates.build_messages(query, context_block, system_prompt=kwargs.get("system_prompt"))

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            logger.error("stream_generation_error", error=str(e))
            yield "An error occurred during generation."

    def generate_summary(self, text: str, length: str = "short") -> str:
        """Generate a short or long summary of the document text."""
        if self._backend == "echo":
            return f"[{length.capitalize()} summary of {len(text)} chars]"

        # Improve coverage for long documents
        if len(text) > 24000:
            # Take chunks from start, middle, and end
            start_chunk = text[:8000]
            mid_point = len(text) // 2
            mid_chunk = text[mid_point - 4000 : mid_point + 4000]
            end_chunk = text[-8000:]
            summarization_content = (
                f"--- START ---\n{start_chunk}\n\n--- MIDDLE ---\n{mid_chunk}\n\n--- END ---\n{end_chunk}"
            )
        else:
            summarization_content = text[:24000]

        prompt = (
            f"Please provide a {length} description of the following document content. "
            f"{'A short description should be about 2 sentences.' if length == 'short' else 'A long description should be a detailed and structured breakdown.'} "
            f"Focus on the main topics, technical specifics, and key takeaways.\n\nContent:\n{summarization_content}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes technical documents."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=300 if length == "short" else 800,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("summary_generation_failed", length=length, error=str(e))
            return "Failed to generate summary."
