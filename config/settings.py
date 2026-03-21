"""
Centralised configuration — loaded once from environment / .env file.

Usage:
    from config import get_settings
    settings = get_settings()
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from llama_index.core import Settings as LlamaSettings
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    """Application-wide configuration.

    Values are read from environment variables first, then from .env.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ───────────────────────────────────────────────
    openai_api_key: str = ""
    llm_model: str = "gpt-4o"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384

    # ── Groq (free LLM fallback) ──────────────────────────
    groq_api_key: str = ""

    # ── Cohere ────────────────────────────────────────────
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-english-v3.0"

    # ── Qdrant ────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "rag_documents"
    qdrant_api_key: Optional[str] = None

    # ── Ingestion ─────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Retrieval ─────────────────────────────────────────
    retrieval_top_k: int = 30
    rerank_top_n: int = 5
    bm25_weight: float = 0.3
    vector_weight: float = 0.7

    # ── API ───────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # ── Paths ─────────────────────────────────────────────
    data_dir: str = "data"
    log_dir: str = "logs"

    def configure_llama_index(self):
        """Configure global LlamaIndex settings with fallbacks."""
        # Embeddings: Use local HuggingFace if OpenAI key is missing
        if not self.openai_api_key:
            LlamaSettings.embed_model = HuggingFaceEmbedding(model_name=self.embedding_model)

        # LLM: Use OpenAI or Groq fallback
        if self.openai_api_key:
            from llama_index.llms.openai import OpenAI

            LlamaSettings.llm = OpenAI(model=self.llm_model, api_key=self.openai_api_key)
        elif self.groq_api_key:
            from llama_index.llms.groq import Groq

            LlamaSettings.llm = Groq(model=self.llm_model, api_key=self.groq_api_key)

        # Templates: Inject our custom structured prompt
        from generation.prompt_templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

        # Create a combined template that mirrors our system reasoning
        full_template = SYSTEM_PROMPT + "\n\n" + USER_PROMPT_TEMPLATE
        LlamaSettings.text_qa_template = PromptTemplate(full_template)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, immutable Settings instance."""
    settings = Settings()
    return settings
