"""
CLI query tool — run a full RAG query without starting the API server.

Usage:
    python scripts/query.py --question "What is RAG?"
    python scripts/query.py --question "What is Qdrant?" --top-k 15 --top-n 3
    python scripts/query.py --question "What is BM25?" --no-rerank
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG CLI — query your document collection"
    )
    parser.add_argument("--question", "-q", required=True, help="Question to ask")
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of retrieval candidates (default: 10)"
    )
    parser.add_argument(
        "--top-n", type=int, default=3, help="Top N after reranking (default: 3)"
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip Cohere reranking (faster, lower quality)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  RAG Production System — Query CLI")
    print("=" * 65)
    print(f"  Question : {args.question}")
    print(f"  top_k    : {args.top_k}")
    print(f"  top_n    : {args.top_n}")
    print(f"  Reranking: {'disabled' if args.no_rerank else 'enabled (Cohere)'}")
    print("=" * 65 + "\n")

    total_start = time.perf_counter()

    # ── 1. Hybrid retrieval ───────────────────────────────────
    print("[1/3] Running hybrid retrieval (vector + BM25)…")
    t0 = time.perf_counter()

    from retrieval.bm25_search import BM25Searcher
    from retrieval.hybrid_search import HybridSearcher
    from retrieval.vector_search import VectorSearcher

    searcher = HybridSearcher(
        vector_searcher=VectorSearcher(),
        bm25_searcher=BM25Searcher(),
    )
    # Build BM25 index from Qdrant payload
    searcher.bm25.build_index()

    candidates = searcher.search(query=args.question, top_k=args.top_k)
    print(f"      Retrieved {len(candidates)} candidates ({(time.perf_counter()-t0)*1000:.0f} ms)\n")

    if not candidates:
        print("❌  No documents found in the vector store.")
        print("    Run:  python scripts/ingest_documents.py --directory data/raw")
        sys.exit(1)

    # ── 2. Reranking ──────────────────────────────────────────
    if args.no_rerank:
        print("[2/3] Skipping reranking (--no-rerank flag set)")
        reranked = candidates[: args.top_n]
    else:
        print(f"[2/3] Reranking {len(candidates)} candidates with Cohere…")
        t0 = time.perf_counter()
        from reranking.cohere_rerank import CohereReranker

        reranker = CohereReranker()
        reranked = reranker.rerank(args.question, candidates, top_n=args.top_n)
        print(f"      Reranked to top {len(reranked)} ({(time.perf_counter()-t0)*1000:.0f} ms)\n")

    # ── 3. Generation ─────────────────────────────────────────
    print("[3/3] Generating answer with LLM…")
    t0 = time.perf_counter()
    from generation.response_generator import ResponseGenerator

    generator = ResponseGenerator()
    result = generator.generate(query=args.question, contexts=reranked)
    gen_ms = (time.perf_counter() - t0) * 1000

    total_ms = (time.perf_counter() - total_start) * 1000

    # ── Output ────────────────────────────────────────────────
    print(f"      Generated in {gen_ms:.0f} ms  |  {result.total_tokens} tokens\n")
    print("=" * 65)
    print("  ANSWER")
    print("=" * 65)
    print(result.answer)
    print()
    print("─" * 65)
    print("  SOURCES")
    print("─" * 65)
    for src in result.sources:
        print(
            f"  [{src['source_id']}] {src['filename']}  "
            f"(chunk {src['chunk_index']}, score {src['relevance_score']:.4f})"
        )
    print()
    print(f"  ⏱  Total latency: {total_ms:.0f} ms  |  Model: {result.model}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
