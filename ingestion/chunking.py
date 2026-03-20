"""
Document chunking strategies.

Production chunking uses a **Sentence-Window** approach with recursive
character splitting as a fallback.  Chunk sizes and overlaps are
configurable via `config.settings`.

Chunking Strategy Rationale
---------------------------
- **Chunk size 512 tokens**: Balances context richness with embedding
  quality.  Larger chunks dilute semantic signal; smaller chunks lose
  context.
- **Overlap 64 tokens**: Preserves boundary context and prevents
  information loss at split points.
- **Sentence-aware splitting**: Avoids mid-sentence breaks which
  degrade retrieval quality.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import structlog
from llama_index.core.schema import Document
from sentence_transformers import SentenceTransformer

from config import get_settings

logger = structlog.get_logger(__name__)

# ── Sentence boundary regex ───────────────────────────────
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


class DocumentChunker:
    """Split documents into retrieval-optimised chunks.

    Parameters
    ----------
    chunk_size : int, optional
        Maximum characters per chunk (default from settings).
    chunk_overlap : int, optional
        Overlap in characters between successive chunks.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        settings = get_settings()
        # Settings store *token* counts; we approximate 1 token ≈ 4 chars
        self.chunk_size = (chunk_size or settings.chunk_size) * 4
        self.chunk_overlap = (chunk_overlap or settings.chunk_overlap) * 4

    # ── Public API ────────────────────────────────────────
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk a list of documents, preserving and extending metadata."""
        all_chunks: List[Document] = []

        for doc in documents:
            chunks = self._split_text(doc.text)
            for idx, chunk_text in enumerate(chunks):
                chunk_meta: Dict = {
                    **doc.metadata,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "chunk_char_len": len(chunk_text),
                }
                all_chunks.append(Document(text=chunk_text, metadata=chunk_meta))

        logger.info(
            "chunking_complete",
            input_docs=len(documents),
            output_chunks=len(all_chunks),
        )
        return all_chunks

    # ── Splitting logic ───────────────────────────────────
    def _split_text(self, text: str) -> List[str]:
        """Sentence-aware recursive split."""
        sentences = _SENT_RE.split(text)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sent_len = len(sentence)

            if current_len + sent_len > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep overlap by retaining trailing sentences
                overlap_text = " ".join(current_chunk)
                overlap_sentences = self._tail_overlap(overlap_text)
                current_chunk = overlap_sentences
                current_len = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_len += sent_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Fallback: if a single sentence exceeds chunk_size, hard-split it
        final_chunks: List[str] = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                final_chunks.extend(self._hard_split(chunk))

        return final_chunks

    def _tail_overlap(self, text: str) -> List[str]:
        """Return trailing sentences that fit within the overlap window."""
        sentences = _SENT_RE.split(text)
        tail: List[str] = []
        total = 0
        for sent in reversed(sentences):
            if total + len(sent) > self.chunk_overlap:
                break
            tail.insert(0, sent)
            total += len(sent)
        return tail

    def _hard_split(self, text: str) -> List[str]:
        """Character-level split for oversized blocks."""
        parts: List[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            parts.append(text[start:end])
            start = end - self.chunk_overlap
        return parts


class SemanticChunker(DocumentChunker):
    """Chunk documents semantically by comparing sentence embeddings."""
    def __init__(
        self,
        embed_model_name: Optional[str] = None,
        similarity_threshold: float = 0.5,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        super().__init__(chunk_size, chunk_overlap)
        settings = get_settings()
        self.similarity_threshold = similarity_threshold
        model_name = embed_model_name or settings.embedding_model
        logger.info("loading_semantic_chunker_model", model=model_name)
        self.embed_model = SentenceTransformer(model_name)

    def _split_text(self, text: str) -> List[str]:
        sentences = _SENT_RE.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return []
            
        embeddings = self.embed_model.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            sim = np.dot(embeddings[i-1], embeddings[i]) / (np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i]))
            
            # If similarity drops below threshold or max size is reached, split
            current_len = sum(len(s) for s in current_chunk) + len(sentences[i])
            if sim < self.similarity_threshold or current_len > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

