"""
Thread-safe Embedding Cache
LRU cache with TTL for embedding vectors
"""

import time
import hashlib
import threading
from typing import Optional, Callable, Any
from dataclasses import dataclass
import numpy as np

try:
    from cachetools import TTLCache
except ImportError:
    # Fallback simple cache implementation
    class TTLCache(dict):
        def __init__(self, maxsize, ttl):
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
        maxsize: int = 10000,
        ttl_seconds: int = 3600,
        enabled: bool = True
    ):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        
        self._cache = TTLCache(maxsize=maxsize, ttl=ttl_seconds)
        self._lock = threading.RLock()
        self._stats = CacheStats()
    
    def _get_key(self, text: str, model: str) -> str:
        """Generate cache key from text and model"""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def get(self, text: str, model: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        if not self.enabled:
            return None
        
        key = self._get_key(text, model)
        
        with self._lock:
            if key in self._cache:
                self._stats.hits += 1
                return self._cache[key]
            else:
                self._stats.misses += 1
                return None
    
    def put(self, text: str, model: str, embedding: np.ndarray):
        """Store embedding in cache"""
        if not self.enabled:
            return
        
        key = self._get_key(text, model)
        
        with self._lock:
            # Check if we need to evict
            if len(self._cache) >= self.maxsize and key not in self._cache:
                self._stats.evictions += 1
            
            self._cache[key] = embedding
            self._stats.current_size = len(self._cache)
    
    async def get_or_compute(
        self,
        text: str,
        model: str,
        compute_fn: Callable
    ) -> np.ndarray:
        """
        Get from cache or compute and store
        
        Args:
            text: Text to embed
            model: Model identifier
            compute_fn: Async function to compute embedding if not cached
        
        Returns:
            Embedding vector
        """
        # Try to get from cache first
        cached = self.get(text, model)
        if cached is not None:
            return cached
        
        # Compute if not in cache
        embedding = await compute_fn(text)
        
        # Store in cache
        self.put(text, model, embedding)
        
        return embedding
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._stats.current_size = 0
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        with self._lock:
            return {
                "enabled": self.enabled,
                "max_size": self.maxsize,
                "current_size": len(self._cache),
                "ttl_seconds": self.ttl_seconds,
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "evictions": self._stats.evictions,
                "hit_rate": self._stats.hit_rate
            }
    
    def reset_stats(self):
        """Reset statistics counters"""
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
