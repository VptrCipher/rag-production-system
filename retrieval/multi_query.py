"""
Multi-Query Retrieval — expands user queries into multiple variations to improve recall.

This technique is a staple of production RAG systems. It overcomes the limitation of
single-vector similarity by generating 3-5 semantically different versions of the
original question, retrieving for each, and fusing the results.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import structlog

from config import get_settings
from retrieval.hybrid_search import HybridSearcher
from retrieval.vector_search import SearchResult

logger = structlog.get_logger(__name__)


class MultiQuerySearcher:
    """Enhance retrieval by generating multiple query variations.

    Architecture:
    1. LLM generates 3 variations of the user query.
    2. HybridSearcher executes retrieval for each variation.
    3. Results are merged via Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, hybrid_searcher: Optional[HybridSearcher] = None):
        self.settings = get_settings()
        self.hybrid = hybrid_searcher or HybridSearcher()

        # Setup LLM client for expansion
        if self.settings.groq_api_key:
            from groq import Groq

            self.client = Groq(api_key=self.settings.groq_api_key)
            self.model = "llama-3.1-8b-instant"
        elif self.settings.openai_api_key:
            import openai

            self.client = openai.OpenAI(api_key=self.settings.openai_api_key)
            self.model = "gpt-4o-mini"
        else:
            self.client = None

    def _generate_queries(self, original_query: str, count: int = 3) -> List[str]:
        """Generate N variations of the original query."""
        if not self.client:
            return [original_query]

        prompt = f"""You are an AI language model assistant. Your task is to generate {count} different versions of the given user query to retrieve relevant documents from a vector database. 
By generating multiple perspectives on the user query, your goal is to help the user overcome some of the limitations of the distance-based similarity search. 

Provide these alternative queries separated by newlines. Do not include any numbers or conversational filler.

Original query: {original_query}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            queries = [q.strip() for q in response.choices[0].message.content.split("\n") if q.strip()]

            # Ensure we always include the original query
            if original_query not in queries:
                queries.append(original_query)

            return queries[: count + 1]
        except Exception as e:
            logger.error("query_expansion_failed", error=str(e))
            return [original_query]

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Perform multi-query retrieval and fusion."""
        expanded_queries = self._generate_queries(query)
        logger.info("expanded_queries", count=len(expanded_queries), queries=expanded_queries)

        all_results: List[List[SearchResult]] = []
        for q in expanded_queries:
            results = self.hybrid.search(q, top_k=top_k, filters=filters)
            all_results.append(results)

        # Merge all result lists using RRF
        return self._multi_list_rrf(all_results, top_k or self.settings.retrieval_top_k)

    def _multi_list_rrf(self, result_lists: List[List[SearchResult]], top_k: int) -> List[SearchResult]:
        """Fusion for multiple lists of search results."""
        from collections import defaultdict

        score_map: Dict[str, float] = defaultdict(float)
        result_map: Dict[str, SearchResult] = {}
        rrf_k = 60

        for results in result_lists:
            for rank, res in enumerate(results, start=1):
                # Use a combined key for deduplication
                key = res.text[:200]
                score_map[key] += 1.0 / (rrf_k + rank)
                if key not in result_map:
                    result_map[key] = res

        # Sort by fused score
        final_results = sorted(result_map.values(), key=lambda r: score_map[r.text[:200]], reverse=True)

        # Update scores to be the RRF values
        for res in final_results:
            res.score = score_map[res.text[:200]]

        return final_results[:top_k]
