"""
Cohere Rerank — cross-encoder reranking layer.

Architecture
------------
1. Bi-encoder (embedding model) produces the initial candidate set quickly.
2. Cross-encoder (Cohere Rerank) scores every (query, document) pair jointly
   for much higher relevance accuracy — but at higher latency.

Pipeline
--------
Query → retrieve top 30 candidates → Cohere rerank → keep top 5.

Latency Tradeoffs
------------------
- Bi-encoder retrieval: ~20-50 ms for 30 candidates.
- Cross-encoder reranking of 30 documents: ~200-400 ms.
- Net improvement in answer quality typically outweighs the latency cost.
- For ultra-low-latency use-cases, reduce candidate set to 15-20.
"""

from __future__ import annotations

from typing import List, Optional

import cohere
import structlog

from config import get_settings
from retrieval.vector_search import SearchResult

logger = structlog.get_logger(__name__)


class CohereReranker:
    """Rerank retrieval candidates with Cohere Rerank API.

    Usage::

        reranker = CohereReranker()
        top_docs = reranker.rerank(query, candidates, top_n=5)
    """

    def __init__(self):
        self.settings = get_settings()
        self.client = cohere.ClientV2(api_key=self.settings.cohere_api_key)
        self.model = self.settings.cohere_rerank_model

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_n: Optional[int] = None,
    ) -> List[SearchResult]:
        """Rerank candidates and return the top-N.

        Parameters
        ----------
        query : str
            The user query.
        candidates : list[SearchResult]
            Retrieval results to rerank (typically 20-30).
        top_n : int, optional
            Number of results to keep after reranking (default: settings.rerank_top_n).
        """
        top_n = top_n or self.settings.rerank_top_n

        if not candidates:
            return []

        documents = [r.text for r in candidates]

        logger.info(
            "reranking_start",
            query_len=len(query),
            candidates=len(candidates),
            top_n=top_n,
        )

        try:
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_n,
            )

            reranked: List[SearchResult] = []
            for result in response.results:
                original = candidates[result.index]
                reranked.append(
                    SearchResult(
                        text=original.text,
                        score=result.relevance_score,
                        metadata={
                            **original.metadata,
                            "original_score": original.score,
                            "rerank_score": result.relevance_score,
                        },
                        source="reranked",
                    )
                )

            logger.info("reranking_complete", reranked_count=len(reranked))
            return reranked
        except Exception as e:
            logger.error("reranking_failed", error=str(e), hint="Falling back to original candidates")
            return candidates[:top_n]
