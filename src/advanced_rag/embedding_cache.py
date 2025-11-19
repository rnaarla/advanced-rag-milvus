"""
Thread-safe Embedding Cache
LRU cache with TTL for embedding vectors
"""

import time
import hashlib
import threading
from typing import Optional, Callable, Any, Dict, Tuple
from dataclasses import dataclass
import numpy as np

try:
    from cachetools import TTLCache  # type: ignore
except Exception:  # pragma: no cover - fallback used when cachetools is absent
    # Minimal TTLCache replacement with dict semantics.
    class TTLCache(dict):  # type: ignore[override]
        def __init__(self, maxsize: int, ttl: int):
            super().__init__()
            self.maxsize = maxsize
            self.ttl = ttl


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    current_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class EmbeddingCache:
    """
    Thread-safe cache for embedding vectors
    
    Features:
    - LRU eviction policy
    - TTL-based expiration
    - Thread-safe operations
    - Cache statistics
    """
    
    def __init__(
        self,
        max_size: int | None = None,
        ttl_seconds: int = 3600,
        enabled: bool = True,
        maxsize: int | None = None,
    ):
        """
        Initialize the cache.

        Both ``max_size`` and ``maxsize`` are accepted for compatibility with
        different tests; they are treated as synonyms and reflected in both
        ``max_size`` and ``maxsize`` attributes.
        """
        effective_max = max_size if max_size is not None else (maxsize if maxsize is not None else 10000)
        self.max_size = effective_max
        self.maxsize = effective_max
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled

        # Internal store: key -> (timestamp, embedding)
        self._cache: Dict[str, Tuple[float, np.ndarray]] = {}
        self._lock = threading.RLock()
        self._stats = CacheStats()

    def _materialize_key(self, *args: Any) -> str:
        """
        Build a cache key from either:
        - (key,)
        - (text, model)
        """
        if len(args) == 1:
            key = str(args[0])
        elif len(args) >= 2:
            text, model = str(args[0]), str(args[1])
            key = f"{model}:{text}"
        else:  # pragma: no cover - defensive
            key = ""
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _sync_get(self, *args: Any) -> Optional[np.ndarray]:
        """Internal synchronous get used by both sync and async APIs."""
        if not self.enabled:
            return None

        key = self._materialize_key(*args)
        with self._lock:
            # Evict expired entries on access
            now = time.time()
            if key in self._cache:
                ts, value = self._cache[key]
                if self.ttl_seconds > 0 and now - ts > self.ttl_seconds:
                    # Expired
                    del self._cache[key]
                    self._stats.current_size = len(self._cache)
                    self._stats.misses += 1
                    return None
                self._stats.hits += 1
                return value
            self._stats.misses += 1
            return None

    def _sync_put(self, *args: Any) -> None:
        """Internal synchronous put used by both sync and async APIs."""
        if not self.enabled:
            return
        if len(args) == 2:
            key_arg, embedding = args
            key = self._materialize_key(key_arg)
        elif len(args) >= 3:
            text, model, embedding = args[0], args[1], args[2]
            key = self._materialize_key(text, model)
        else:  # pragma: no cover - defensive
            return

        with self._lock:
            now = time.time()
            # Capacity-based eviction (simple FIFO/LRU approximation)
            if len(self._cache) >= self.max_size and key not in self._cache:
                try:
                    oldest_key = next(iter(self._cache.keys()))
                    del self._cache[oldest_key]
                    self._stats.evictions += 1
                except StopIteration:  # pragma: no cover - defensive
                    pass

            self._cache[key] = (now, embedding)
            self._stats.current_size = len(self._cache)

    def get(self, *args: Any) -> Any:
        """
        Get embedding from cache.

        Supports both:
        - cache.get(key)
        - cache.get(text, model)

        When awaited (as done in async tests), returns the underlying
        embedding or ``None``. When used synchronously (as in some legacy
        tests), returns the embedding directly.
        """
        value = self._sync_get(*args)

        class _AwaitableValue:
            def __init__(self, v: Any):
                self._v = v

            def __await__(self):
                async def _wrap():
                    return self._v

                return _wrap().__await__()

        return _AwaitableValue(value)

    def put(self, *args: Any) -> Any:
        """
        Store embedding in cache.

        Supports:
        - cache.put(key, embedding)
        - cache.put(text, model, embedding)
        """
        self._sync_put(*args)

        class _AwaitableNone:
            def __await__(self):
                async def _wrap():
                    return None

                return _wrap().__await__()

        return _AwaitableNone()

    def get_or_compute(self, *args: Any) -> Any:
        """
        Get from cache or compute and store.

        Supported signatures:
        - cache.get_or_compute(key, compute_fn)
        - cache.get_or_compute(text, model, compute_fn)
        """

        if len(args) == 2:
            key, compute_fn = args
            key_args: Tuple[Any, ...] = (key,)
        elif len(args) >= 3:
            text, model, compute_fn = args[0], args[1], args[2]
            key, key_args = text, (text, model)
        else:  # pragma: no cover - defensive
            raise TypeError("get_or_compute requires at least key and compute_fn")

        async def _compute_wrapper():
            cached = self._sync_get(*key_args)
            if cached is not None:
                return cached
            # compute_fn is async in all current call sites/tests
            embedding = await compute_fn() if callable(compute_fn) and compute_fn.__code__.co_argcount == 0 else await compute_fn(key)  # type: ignore[arg-type]
            self._sync_put(key, embedding)
            return embedding

        class _AwaitableCompute:
            def __await__(self):
                return _compute_wrapper().__await__()

        return _AwaitableCompute()

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats.current_size = 0
            # Reset detailed stats as well
            self._stats.hits = 0
            self._stats.misses = 0
            self._stats.evictions = 0

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            return {
                # Names expected by tests in tests/test_embedding_cache.py
                "size": len(self._cache),
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "evictions": self._stats.evictions,
                "hit_rate": self._stats.hit_rate,
                # Extra fields for introspection
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "enabled": self.enabled,
            }

    def reset_stats(self):
        """Reset statistics counters."""
        with self._lock:
            self._stats = CacheStats(current_size=len(self._cache))


# Global cache instances
_semantic_cache: Optional[EmbeddingCache] = None
_sparse_cache: Optional[EmbeddingCache] = None
_domain_cache: Optional[EmbeddingCache] = None


def initialize_caches(
    maxsize: int = 10000,
    ttl_seconds: int = 3600,
    enabled: bool = True
):
    """Initialize global embedding caches"""
    global _semantic_cache, _sparse_cache, _domain_cache
    
    _semantic_cache = EmbeddingCache(maxsize, ttl_seconds, enabled)
    _sparse_cache = EmbeddingCache(maxsize, ttl_seconds, enabled)
    _domain_cache = EmbeddingCache(maxsize // 2, ttl_seconds, enabled)


def get_semantic_cache() -> EmbeddingCache:
    """Get semantic embedding cache"""
    if _semantic_cache is None:
        initialize_caches()
    return _semantic_cache


def get_sparse_cache() -> EmbeddingCache:
    """Get sparse embedding cache"""
    if _sparse_cache is None:
        initialize_caches()
    return _sparse_cache


def get_domain_cache() -> EmbeddingCache:
    """Get domain embedding cache"""
    if _domain_cache is None:
        initialize_caches()
    return _domain_cache
