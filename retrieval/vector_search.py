"""
Dense vector search via Qdrant.

Encodes the user query with the same embedding model used at ingestion
time, then performs approximate nearest-neighbour search.

The SentenceTransformer model is loaded once at init to avoid per-query
latency from model loading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import get_settings
from config.qdrant_client import get_qdrant_client

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    """Unified result container used across all retrieval strategies."""

    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""  # "vector" | "bm25" | "hybrid"


class VectorSearcher:
    """Semantic nearest-neighbour search backed by Qdrant.

    The SentenceTransformer model is loaded once at construction time and
    reused for every query — avoiding the O(N) model-load overhead.
    """

    def __init__(self, qdrant_client: Optional[QdrantClient] = None):
        self.settings = get_settings()
        self.client = qdrant_client or get_qdrant_client()
        self.collection_name = self.settings.qdrant_collection

        # Load model once — shared across all embed_query() calls
        logger.info("loading_embedding_model", model=self.settings.embedding_model)
        self.model = SentenceTransformer(self.settings.embedding_model)
        logger.info("embedding_model_loaded")

    def embed_query(self, query: str) -> List[float]:
        """Embed a query string using the cached SentenceTransformer model."""
        embeddings = self.model.encode(
            [query],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings[0].tolist()

    def search(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Return top-k nearest vectors for a query embedding.

        Parameters
        ----------
        query_embedding : list[float]
            Dense embedding of the user query.
        top_k : int, optional
            Number of results (default from settings.retrieval_top_k).
        filters : dict, optional
            Qdrant payload filter (e.g. ``{"file_type": "pdf"}``).
        """
        top_k = top_k or self.settings.retrieval_top_k

        # Build Qdrant filter if provided
        query_filter = None
        if filters:
            from qdrant_client.http.models import FieldCondition, Filter, MatchValue

            conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
            query_filter = Filter(must=conditions)

        # Using query_points as search is missing in this version
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        hits = response.points

        results: List[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            score = 0.0
            if hasattr(hit, "score"):
                score = hit.score

            results.append(
                SearchResult(
                    text=payload.get("text", ""),
                    score=score,
                    metadata={k: v for k, v in payload.items() if k != "text"},
                    source="vector",
                )
            )

        logger.info("vector_search", query_len=len(query_embedding), results=len(results))
        return results
