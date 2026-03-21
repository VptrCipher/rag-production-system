"""
Embedding generation + Qdrant upsert pipeline.

Generates dense embeddings via a local SentenceTransformer model and upserts
vectors with full metadata payloads into a Qdrant collection.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from config import get_settings
from config.qdrant_client import get_qdrant_client

logger = structlog.get_logger(__name__)


class EmbeddingPipeline:
    """Generate embeddings and store them in Qdrant.

    The SentenceTransformer model is loaded once at init to avoid repeated
    disk/network access on every ingestion call.

    Usage::

        pipeline = EmbeddingPipeline()
        pipeline.ensure_collection()
        pipeline.ingest(chunks)
    """

    def __init__(self, qdrant_client: Optional[QdrantClient] = None):
        self.settings = get_settings()
        self.client = qdrant_client or get_qdrant_client()
        self.collection_name = self.settings.qdrant_collection
        self.embedding_dim = self.settings.embedding_dimension

        # Load model once at startup — avoids re-loading on every call
        logger.info("loading_embedding_model", model=self.settings.embedding_model)
        self.model = SentenceTransformer(self.settings.embedding_model)
        logger.info("embedding_model_loaded", model=self.settings.embedding_model)

    # ── Embedding ─────────────────────────────────────────
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Encode a list of texts using the cached SentenceTransformer model."""
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 32,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    # ── Collection management ─────────────────────────────
    def ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "collection_created",
                name=self.collection_name,
                dim=self.embedding_dim,
            )
        else:
            logger.info("collection_exists", name=self.collection_name)

    # ── Ingestion ─────────────────────────────────────────
    def ingest(self, documents: list) -> int:
        """Embed and upsert a list of LlamaIndex Documents.

        Returns the number of points upserted.
        """
        texts: List[str] = [doc.text for doc in documents]
        metadatas: List[Dict] = [doc.metadata for doc in documents]

        logger.info("generating_embeddings", count=len(texts))
        embeddings = self.embed_texts(texts)

        points: List[PointStruct] = []
        for text, embedding, meta in zip(texts, embeddings, metadatas):
            point_id = str(uuid.uuid4())
            payload = {
                **meta,
                "text": text,
            }
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

        # Upsert in batches of 256
        batch_size = 256
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        logger.info("ingestion_complete", points_upserted=len(points))
        return len(points)

    def get_collection_info(self) -> dict:
        """Return collection statistics."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value,
        }
