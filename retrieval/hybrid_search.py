"""
Hybrid retrieval — merges dense vector and sparse BM25 results using
**Reciprocal Rank Fusion (RRF)**.

RRF score for a document *d* across *k* ranked lists:

    RRF(d) = Σ  1 / (k_constant + rank_i(d))

This is more robust than raw score normalisation because it is
independent of the score distribution of each retriever.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import structlog

from config import get_settings
from retrieval.vector_search import SearchResult, VectorSearcher
from retrieval.bm25_search import BM25Searcher

logger = structlog.get_logger(__name__)

_RRF_K = 60  # standard RRF constant


class HybridSearcher:
    """Combine vector and BM25 retrieval with Reciprocal Rank Fusion.

    Usage::

        hybrid = HybridSearcher()
        hybrid.bm25.build_index()          # one-time index build
        results = hybrid.search("my query")
    """

    def __init__(
        self,
        vector_searcher: Optional[VectorSearcher] = None,
        bm25_searcher: Optional[BM25Searcher] = None,
    ):
        self.settings = get_settings()
        self.vector = vector_searcher or VectorSearcher()
        self.bm25 = bm25_searcher or BM25Searcher()

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Run hybrid retrieval and return RRF-fused results.

        Parameters
        ----------
        query : str
            User query.
        top_k : int, optional
            Number of final results to return.
        vector_weight : float, optional
            Weight multiplier for vector RRF scores [0-1].
        bm25_weight : float, optional
            Weight multiplier for BM25 RRF scores [0-1].
        filters : dict, optional
            Metadata filters (e.g. {"filename": "doc.pdf"})
        """
        top_k = top_k or self.settings.retrieval_top_k
        vw = vector_weight if vector_weight is not None else self.settings.vector_weight
        bw = bm25_weight if bm25_weight is not None else self.settings.bm25_weight

        # 1. Dense retrieval
        query_embedding = self.vector.embed_query(query)
        vector_results = self.vector.search(query_embedding, top_k=top_k, filters=filters)

        # 2. Sparse retrieval
        bm25_results = self.bm25.search(query, top_k=top_k, filters=filters)

        # 3. Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(
            vector_results, bm25_results, vw, bw
        )

        # 4. Take top_k
        fused = fused[:top_k]

        logger.info(
            "hybrid_search",
            vector_hits=len(vector_results),
            bm25_hits=len(bm25_results),
            fused_hits=len(fused),
            filtered=bool(filters),
        )
        return fused

    # ── RRF implementation ────────────────────────────────
    @staticmethod
    def _reciprocal_rank_fusion(
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
        vector_weight: float,
        bm25_weight: float,
    ) -> List[SearchResult]:
        """Merge two ranked lists via weighted RRF."""

        # Key by text content (could use a doc-id in production)
        score_map: Dict[str, float] = defaultdict(float)
        result_map: Dict[str, SearchResult] = {}

        for rank, res in enumerate(vector_results, start=1):
            key = res.text[:200]  # use first 200 chars as dedup key
            score_map[key] += vector_weight / (_RRF_K + rank)
            if key not in result_map:
                result_map[key] = SearchResult(
                    text=res.text,
                    score=0.0,
                    metadata=res.metadata,
                    source="hybrid",
                )

        for rank, res in enumerate(bm25_results, start=1):
            key = res.text[:200]
            score_map[key] += bm25_weight / (_RRF_K + rank)
            if key not in result_map:
                result_map[key] = SearchResult(
                    text=res.text,
                    score=0.0,
                    metadata=res.metadata,
                    source="hybrid",
                )

        # Assign fused scores and sort
        for key, score in score_map.items():
            result_map[key].score = score

        return sorted(result_map.values(), key=lambda r: r.score, reverse=True)
