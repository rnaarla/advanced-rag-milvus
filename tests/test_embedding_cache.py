"""
Comprehensive tests for embedding cache module

Tests cover:
- Cache hit/miss behavior
- LRU eviction
- TTL expiration
- Thread safety
- Statistics tracking
- get_or_compute functionality
- Key generation
"""

import pytest
import asyncio
import time
import numpy as np
import sys
import os
import importlib.util

# Load module directly without triggering __init__.py
module_path = os.path.join(os.path.dirname(__file__), '../src/advanced_rag/embedding_cache.py')
spec = importlib.util.spec_from_file_location("embedding_cache", module_path)
embedding_cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(embedding_cache)

EmbeddingCache = embedding_cache.EmbeddingCache
get_semantic_cache = embedding_cache.get_semantic_cache
get_sparse_cache = embedding_cache.get_sparse_cache
get_domain_cache = embedding_cache.get_domain_cache


class TestEmbeddingCache:
    """Test EmbeddingCache functionality"""
    
    @pytest.mark.asyncio
    async def test_cache_initialization(self):
        """Test cache initializes with correct parameters"""
        cache = EmbeddingCache(max_size=100, ttl_seconds=60)
        
        assert cache.max_size == 100
        assert cache.ttl_seconds == 60
        
        stats = cache.get_stats()
        assert stats['size'] == 0
        assert stats['hits'] == 0
        assert stats['misses'] == 0
    
    @pytest.mark.asyncio
    async def test_put_and_get(self):
        """Test basic put and get operations"""
        cache = EmbeddingCache()
        
        embedding = np.array([1.0, 2.0, 3.0])
        await cache.put("test_key", embedding)
        
        retrieved = await cache.get("test_key")
        
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, embedding)
    
    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss returns None"""
        cache = EmbeddingCache()
        
        result = await cache.get("nonexistent_key")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_hit_statistics(self):
        """Test hit statistics are tracked"""
        cache = EmbeddingCache()
        
        embedding = np.array([1.0, 2.0, 3.0])
        await cache.put("key1", embedding)
        
        # First get - cache hit
        await cache.get("key1")
        
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 0
    
    @pytest.mark.asyncio
    async def test_cache_miss_statistics(self):
        """Test miss statistics are tracked"""
        cache = EmbeddingCache()
        
        # Try to get non-existent key
        await cache.get("nonexistent")
        
        stats = cache.get_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 1
    
    @pytest.mark.asyncio
    async def test_cache_size_tracking(self):
        """Test cache size is tracked correctly"""
        cache = EmbeddingCache()
        
        for i in range(5):
            embedding = np.array([float(i)])
            await cache.put(f"key{i}", embedding)
        
        stats = cache.get_stats()
        assert stats['size'] == 5
    
    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction when max_size is reached"""
        cache = EmbeddingCache(max_size=3)
        
        # Add 3 items
        for i in range(3):
            await cache.put(f"key{i}", np.array([float(i)]))
        
        assert cache.get_stats()['size'] == 3
        
        # Add one more - should evict oldest
        await cache.put("key3", np.array([3.0]))
        
        assert cache.get_stats()['size'] == 3
        
        # key0 should be evicted
        result = await cache.get("key0")
        assert result is None
        
        # Others should still exist
        result = await cache.get("key1")
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test entries expire after TTL"""
        cache = EmbeddingCache(ttl_seconds=1)
        
        embedding = np.array([1.0, 2.0, 3.0])
        await cache.put("test_key", embedding)
        
        # Should be available immediately
        result = await cache.get("test_key")
        assert result is not None
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired
        result = await cache.get("test_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_or_compute_cache_hit(self):
        """Test get_or_compute uses cached value"""
        cache = EmbeddingCache()
        
        embedding = np.array([1.0, 2.0, 3.0])
        await cache.put("key1", embedding)
        
        compute_called = False
        
        async def compute_fn():
            nonlocal compute_called
            compute_called = True
            return np.array([4.0, 5.0, 6.0])
        
        result = await cache.get_or_compute("key1", compute_fn)
        
        # Should return cached value
        np.testing.assert_array_equal(result, embedding)
        # Should not call compute function
        assert not compute_called
    
    @pytest.mark.asyncio
    async def test_get_or_compute_cache_miss(self):
        """Test get_or_compute computes on miss"""
        cache = EmbeddingCache()
        
        computed_embedding = np.array([4.0, 5.0, 6.0])
        
        async def compute_fn():
            return computed_embedding
        
        result = await cache.get_or_compute("new_key", compute_fn)
        
        # Should return computed value
        np.testing.assert_array_equal(result, computed_embedding)
        
        # Should be cached now
        cached = await cache.get("new_key")
        np.testing.assert_array_equal(cached, computed_embedding)
    
    @pytest.mark.asyncio
    async def test_key_generation_consistency(self):
        """Test same text generates same key"""
        cache = EmbeddingCache()
        
        text = "test embedding text"
        embedding = np.array([1.0, 2.0, 3.0])
        
        # Store with text as key
        await cache.put(text, embedding)
        
        # Retrieve with same text
        result = await cache.get(text)
        
        assert result is not None
        np.testing.assert_array_equal(result, embedding)
    
    @pytest.mark.asyncio
    async def test_different_texts_different_keys(self):
        """Test different texts generate different cache keys"""
        cache = EmbeddingCache()
        
        text1 = "first text"
        text2 = "second text"
        
        emb1 = np.array([1.0, 2.0])
        emb2 = np.array([3.0, 4.0])
        
        await cache.put(text1, emb1)
        await cache.put(text2, emb2)
        
        result1 = await cache.get(text1)
        result2 = await cache.get(text2)
        
        np.testing.assert_array_equal(result1, emb1)
        np.testing.assert_array_equal(result2, emb2)
    
    @pytest.mark.asyncio
    async def test_hit_rate_calculation(self):
        """Test hit rate is calculated correctly"""
        cache = EmbeddingCache()
        
        # Add some entries
        await cache.put("key1", np.array([1.0]))
        await cache.put("key2", np.array([2.0]))
        
        # 2 hits
        await cache.get("key1")
        await cache.get("key2")
        
        # 1 miss
        await cache.get("key3")
        
        stats = cache.get_stats()
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['hit_rate'] == pytest.approx(2.0 / 3.0)
    
    @pytest.mark.asyncio
    async def test_hit_rate_no_requests(self):
        """Test hit rate with no requests"""
        cache = EmbeddingCache()
        
        stats = cache.get_stats()
        assert stats['hit_rate'] == 0.0
    
    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Test cache can be cleared"""
        cache = EmbeddingCache()
        
        # Add entries
        for i in range(5):
            await cache.put(f"key{i}", np.array([float(i)]))
        
        assert cache.get_stats()['size'] == 5
        
        cache.clear()
        
        stats = cache.get_stats()
        assert stats['size'] == 0
        assert stats['hits'] == 0
        assert stats['misses'] == 0
    
    @pytest.mark.asyncio
    async def test_thread_safety(self):
        """Test cache operations are thread-safe"""
        cache = EmbeddingCache()
        
        async def worker(worker_id):
            for i in range(10):
                key = f"worker{worker_id}_key{i}"
                embedding = np.array([float(worker_id), float(i)])
                await cache.put(key, embedding)
                
                # Read it back
                result = await cache.get(key)
                assert result is not None
        
        # Run multiple workers concurrently
        tasks = [worker(i) for i in range(5)]
        await asyncio.gather(*tasks)
        
        stats = cache.get_stats()
        assert stats['size'] <= 50  # Some might be evicted
        assert stats['hits'] == 50
        assert stats['misses'] == 0
    
    @pytest.mark.asyncio
    async def test_large_embedding_storage(self):
        """Test storing large embeddings"""
        cache = EmbeddingCache()
        
        # Large embedding (768 dimensions like BERT)
        large_embedding = np.random.randn(768).astype(np.float32)
        
        await cache.put("large_key", large_embedding)
        
        result = await cache.get("large_key")
        
        assert result is not None
        np.testing.assert_array_equal(result, large_embedding)
    
    @pytest.mark.asyncio
    async def test_numpy_array_types(self):
        """Test different numpy array types"""
        cache = EmbeddingCache()
        
        # Different dtypes
        float32_arr = np.array([1.0, 2.0], dtype=np.float32)
        float64_arr = np.array([3.0, 4.0], dtype=np.float64)
        
        await cache.put("float32", float32_arr)
        await cache.put("float64", float64_arr)
        
        result32 = await cache.get("float32")
        result64 = await cache.get("float64")
        
        np.testing.assert_array_equal(result32, float32_arr)
        np.testing.assert_array_equal(result64, float64_arr)
    
    @pytest.mark.asyncio
    async def test_concurrent_get_or_compute(self):
        """Test concurrent get_or_compute calls"""
        cache = EmbeddingCache()
        
        call_count = 0
        
        async def compute_fn():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate work
            return np.array([1.0, 2.0, 3.0])
        
        # Multiple concurrent calls for same key
        tasks = [
            cache.get_or_compute("same_key", compute_fn)
            for _ in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should get the same result
        for result in results:
            np.testing.assert_array_equal(result, results[0])
        
        # Compute function might be called multiple times due to race conditions
        # but all results should be consistent


class TestGlobalCacheFunctions:
    """Test module-level cache accessor functions"""
    
    @pytest.mark.asyncio
    async def test_get_semantic_cache(self):
        """Test get_semantic_cache returns singleton"""
        cache1 = get_semantic_cache()
        cache2 = get_semantic_cache()
        
        assert cache1 is cache2
        
        # Test it works
        embedding = np.array([1.0, 2.0, 3.0])
        await cache1.put("test", embedding)
        
        result = await cache2.get("test")
        np.testing.assert_array_equal(result, embedding)
    
    @pytest.mark.asyncio
    async def test_get_sparse_cache(self):
        """Test get_sparse_cache returns singleton"""
        cache1 = get_sparse_cache()
        cache2 = get_sparse_cache()
        
        assert cache1 is cache2
    
    @pytest.mark.asyncio
    async def test_get_domain_cache(self):
        """Test get_domain_cache returns singleton"""
        cache1 = get_domain_cache()
        cache2 = get_domain_cache()
        
        assert cache1 is cache2
    
    @pytest.mark.asyncio
    async def test_different_cache_types_independent(self):
        """Test different cache types are independent"""
        semantic_cache = get_semantic_cache()
        sparse_cache = get_sparse_cache()
        domain_cache = get_domain_cache()
        
        # All should be different instances
        assert semantic_cache is not sparse_cache
        assert semantic_cache is not domain_cache
        assert sparse_cache is not domain_cache
        
        # Store in semantic cache
        embedding = np.array([1.0, 2.0, 3.0])
        await semantic_cache.put("key1", embedding)
        
        # Should not appear in other caches
        assert await sparse_cache.get("key1") is None
        assert await domain_cache.get("key1") is None
    
    @pytest.mark.asyncio
    async def test_cache_persistence_across_calls(self):
        """Test cache persists across multiple get_cache calls"""
        cache = get_semantic_cache()
        
        embedding = np.array([1.0, 2.0, 3.0])
        await cache.put("persistent_key", embedding)
        
        # Get cache again (should be same instance)
        cache2 = get_semantic_cache()
        
        result = await cache2.get("persistent_key")
        assert result is not None
        np.testing.assert_array_equal(result, embedding)
