"""
Redis Cache Manager for Basic RAG Module

Provides async caching for retrieval results and final LLM responses.
Gracefully degrades if Redis is unavailable.

Note: Existing cache entries generated under the BGE-M3 API model are incompatible
with the new local BAAI/bge-small-en-v1.5 model. The config key_prefix has been updated
to intentionally bust the cache and avoid dimension mismatch errors.
"""

import json
import logging
import hashlib
from typing import Any, Dict, Optional

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Async Redis cache wrapper with graceful degradation.
    """

    def __init__(self, config: Dict[str, Any]):
        self._url = config.get("redis_url", "redis://localhost:6379/0")
        self._ttl = config.get("ttl_seconds", 3600)
        self._prefix = config.get("key_prefix", "basic_rag")
        
        self._redis: Optional[redis.Redis] = None
        self._enabled = True
        
        self._stats = {"hits": 0, "misses": 0, "errors": 0}
        
        if redis is None:
            logger.warning("redis package not installed. Caching disabled.")
            self._enabled = False
        else:
            try:
                self._redis = redis.from_url(self._url)
                logger.info(f"CacheManager initialized with Redis at {self._url}")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}. Caching disabled.")
                self._enabled = False
                self._redis = None

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def generate_cache_key(self, prefix_type: str, query: str, config_hash: str) -> str:
        """
        Generate a deterministic cache key.
        
        Args:
            prefix_type: e.g., 'retrieval', 'response', 'embedding'
            query: The user's query or text
            config_hash: Hash of relevant config parameters
            
        Returns:
            A formatted cache key string.
        """
        content = f"{query}:{config_hash}".encode("utf-8")
        hash_str = hashlib.sha256(content).hexdigest()
        return f"{self._prefix}:{prefix_type}:{hash_str}"

    async def get(self, key: str) -> Optional[Any]:
        """Get and deserialize a value from cache."""
        if not self._enabled or not self._redis:
            return None
            
        try:
            val = await self._redis.get(key)
            if val:
                self._stats["hits"] += 1
                return json.loads(val)
            else:
                self._stats["misses"] += 1
                return None
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Redis get error for key {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Serialize and set a value in cache."""
        if not self._enabled or not self._redis:
            return False
            
        try:
            serialized = json.dumps(value)
            await self._redis.set(key, serialized, ex=ttl or self._ttl)
            return True
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Redis set error for key {key}: {e}")
            return False

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis:
            await self._redis.close()
            logger.info("CacheManager Redis connection closed.")
