import hashlib
import json
from typing import Any, Optional

import structlog
from cachetools import TTLCache

logger = structlog.get_logger(__name__)


class CacheManager:
    """
    Thread-safe in-memory cache for LLM responses and embeddings.
    Uses TTL (Time To Live) to expire old entries and LRU for capacity management.
    """

    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        # Default: 1000 items, 1 hour TTL
        self.response_cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.embedding_cache = TTLCache(maxsize=maxsize * 2, ttl=ttl * 24)  # Embeddings last longer

    def _generate_key(self, prefix: str, data: Any) -> str:
        """Create a unique MD5 hash for the data."""
        serialized = json.dumps(data, sort_keys=True)
        return f"{prefix}:{hashlib.md5(serialized.encode()).hexdigest()}"

    def get_response(self, question: str, params: dict) -> Optional[Any]:
        key = self._generate_key("resp", {"q": question, **params})
        val = self.response_cache.get(key)
        if val:
            logger.info("cache_hit", type="response", key=key)
        return val

    def set_response(self, question: str, params: dict, response: Any):
        key = self._generate_key("resp", {"q": question, **params})
        self.response_cache[key] = response
        logger.info("cache_set", type="response", key=key)

    def get_embedding(self, text: str) -> Optional[list[float]]:
        key = self._generate_key("emb", text)
        val = self.embedding_cache.get(key)
        if val:
            logger.info("cache_hit", type="embedding", key=key)
        return val

    def set_embedding(self, text: str, vector: list[float]):
        key = self._generate_key("emb", text)
        self.embedding_cache[key] = vector
        logger.info("cache_set", type="embedding", key=key)


# Singleton instance
cache_manager = CacheManager()
