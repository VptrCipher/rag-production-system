"""
CLI script for batch document ingestion.

Usage:
    python scripts/ingest_documents.py --directory data/raw
    python scripts/ingest_documents.py --directory data/raw --extensions .pdf .md
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings
from ingestion.chunking import DocumentChunker, SemanticChunker
from ingestion.embedding_pipeline import EmbeddingPipeline
from ingestion.loaders import DocumentLoader


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Path to directory containing documents",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=None,
        help="File extensions to include (e.g. .pdf .md .txt)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk size (tokens)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Override chunk overlap (tokens)",
    )
    args = parser.parse_args()

    settings = get_settings()
    start = time.perf_counter()

    print(f"{'='*60}")
    print("  RAG Document Ingestion Pipeline")
    print(f"{'='*60}")
    print(f"  Directory  : {args.directory}")
    print(f"  Extensions : {args.extensions or 'all supported'}")
    print(f"  Chunk size : {args.chunk_size or settings.chunk_size} tokens")
    print(f"  Qdrant     : local_qdrant_storage")
    print(f"  Collection : {settings.qdrant_collection}")
    print(f"{'='*60}\n")

    # 1. Load documents
    print("[1/4] Loading documents...")
    loader = DocumentLoader()
    documents = loader.load_directory(
        args.directory,
        extensions=args.extensions,
    )
    print(f"      Loaded {len(documents)} documents\n")

    if not documents:
        print("No documents found. Exiting.")
        return

    # 2. Chunk documents
    print("[2/4] Chunking documents semantically...")
    chunker = SemanticChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunks = chunker.chunk_documents(documents)
    print(f"      Created {len(chunks)} chunks\n")

    # 3. Ensure collection exists
    print("[3/4] Ensuring Qdrant collection...")
    pipeline = EmbeddingPipeline()
    pipeline.ensure_collection()
    print("      Collection ready\n")

    # 4. Embed and store
    print("[4/4] Generating embeddings and storing...")
    stored = pipeline.ingest(chunks)
    elapsed = time.perf_counter() - start
    print(f"      Stored {stored} vectors\n")

    print(f"{'='*60}")
    print(f"  Ingestion complete in {elapsed:.1f}s")
    print(f"  Documents: {len(documents)} -> Chunks: {len(chunks)} -> Vectors: {stored}")
    info = pipeline.get_collection_info()
    print(f"  Collection '{info['name']}': {info['points_count']} total points")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
