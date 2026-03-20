"""
BM25 sparse keyword retrieval.

Maintains an in-memory BM25 index over the document corpus stored in
Qdrant.  The index is rebuilt on initialisation by pulling all payloads
from the collection.

For production at scale, consider using Elasticsearch or Qdrant's
built-in sparse vectors instead of an in-memory index.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import structlog
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from config import get_settings
from config.qdrant_client import get_qdrant_client
from retrieval.vector_search import SearchResult

logger = structlog.get_logger(__name__)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokeniser."""
    return re.findall(r"\w+", text.lower())


class BM25Searcher:
    """BM25-based keyword retrieval over the Qdrant corpus."""

    def __init__(self, qdrant_client: Optional[QdrantClient] = None):
        self.settings = get_settings()
        self.client = qdrant_client or get_qdrant_client()

        self.collection_name = self.settings.qdrant_collection
        self._corpus: List[Dict] = []
        self._tokenized_corpus: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    # ── Index management ──────────────────────────────────
    def build_index(self) -> int:
        """Pull all texts from Qdrant and build the BM25 index.

        Returns the corpus size.
        """
        self._corpus = []
        self._tokenized_corpus = []

        # Scroll through all points (paginated)
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                text = payload.get("text", "")
                if text:
                    self._corpus.append(payload)
                    self._tokenized_corpus.append(_tokenize(text))
            if offset is None:
                break

        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info("bm25_index_built", corpus_size=len(self._corpus))
        return len(self._corpus)

    # ── Search ────────────────────────────────────────────
    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Return top-k BM25 results for a query string."""
        if self._bm25 is None:
            logger.warning("bm25_index_empty")
            return []

        top_k = top_k or self.settings.retrieval_top_k
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Get top indices by score
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: List[SearchResult] = []
        for idx in ranked_indices:
            if scores[idx] <= 0:
                continue
            
            payload = self._corpus[idx]
            
            # Simple metadata filtering for in-memory BM25
            if filters:
                match = True
                for k, v in filters.items():
                    if payload.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            results.append(
                SearchResult(
                    text=payload.get("text", ""),
                    score=float(scores[idx]),
                    metadata={k: v for k, v in payload.items() if k != "text"},
                    source="bm25",
                )
            )
            
            if len(results) >= top_k:
                break

        logger.info("bm25_search", query_tokens=len(tokens), results=len(results), filtered=bool(filters))
        return results
