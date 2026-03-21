"""
Shared Qdrant client singleton.

Using QdrantClient with path= acquires a file lock on the local storage
folder. Opening multiple QdrantClient instances against the same path in
the same process causes an AlreadyLocked error.

Import `get_qdrant_client()` everywhere instead of constructing a new
QdrantClient directly.
"""

from __future__ import annotations

import threading
from typing import Optional

from qdrant_client import QdrantClient

_client: Optional[QdrantClient] = None
_lock = threading.Lock()


def get_qdrant_client() -> QdrantClient:
    """Return the shared QdrantClient instance (local disk mode)."""
    global _client
    if _client is None:
        with _lock:
            # Double-check pattern
            if _client is None:
                _client = QdrantClient(path="local_qdrant_storage")
    return _client
