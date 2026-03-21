"""
Multi-format document loader.

Supports: PDF, HTML, Markdown, plain text.
Each loaded document is returned as a LlamaIndex `Document` with rich metadata.
"""

from __future__ import annotations

import hashlib
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import structlog
from llama_index.core.schema import Document

logger = structlog.get_logger(__name__)

# ── Supported extensions ──────────────────────────────────
_EXTENSION_MAP: dict[str, str] = {
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".md": "markdown",
    ".txt": "text",
    ".text": "text",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
}


def _file_hash(path: Path) -> str:
    """SHA-256 hex-digest of a file (streamed for large files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class DocumentLoader:
    """Load documents from disk in multiple formats.

    Usage::

        loader = DocumentLoader()
        docs = loader.load_directory("data/raw")
    """

    def load_file(self, path: str | Path) -> List[Document]:
        """Load a single file and return LlamaIndex Document(s)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        fmt = _EXTENSION_MAP.get(ext)
        if fmt is None:
            logger.warning("unsupported_extension", path=str(path), ext=ext)
            return []

        logger.info("loading_file", path=str(path), format=fmt)

        text = self._extract_text(path, fmt)
        if not text.strip():
            logger.warning("empty_document", path=str(path))
            return []

        metadata = {
            "source": str(path.resolve()),
            "filename": path.name,
            "file_type": fmt,
            "file_hash": _file_hash(path),
            "file_size_bytes": path.stat().st_size,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }

        return [Document(text=text, metadata=metadata)]

    def load_directory(
        self,
        directory: str | Path,
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[Document]:
        """Load all supported files from a directory."""
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        allowed_exts = set(extensions) if extensions else set(_EXTENSION_MAP.keys())
        pattern = "**/*" if recursive else "*"

        documents: List[Document] = []

        for file_path in sorted(directory.glob(pattern)):
            if file_path.is_file() and file_path.suffix.lower() in allowed_exts:
                try:
                    documents.extend(self.load_file(file_path))
                except Exception:
                    logger.exception("load_error", path=str(file_path))

        logger.info(
            "directory_loaded",
            directory=str(directory),
            total_documents=len(documents),
        )
        return documents

    # ── Private helpers ───────────────────────────────────
    def _extract_text(self, path: Path, fmt: str) -> str:
        if fmt == "pdf":
            return self._read_pdf(path)
        if fmt == "html":
            return self._read_html(path)
        if fmt in ("markdown", "text"):
            return path.read_text(encoding="utf-8", errors="replace")
        if fmt == "image":
            return self._read_image(path)
        return ""

    @staticmethod
    def _read_image(path: Path) -> str:
        """Extract text from images using OCR."""
        try:
            from llama_index.readers.file import ImageReader

            loader = ImageReader(keep_image=False)
            documents = loader.load_data(file=path)
            return "\n".join([d.text for d in documents])
        except Exception as e:
            logger.warning("ocr_failed", path=str(path), error=str(e))
            return f"[OCR Failed for {path.name}]"

    @staticmethod
    def _read_pdf(path: Path) -> str:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages: List[str] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)
        return "\n\n".join(pages)

    @staticmethod
    def _read_html(path: Path) -> str:
        from bs4 import BeautifulSoup

        raw = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "html.parser")
        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
