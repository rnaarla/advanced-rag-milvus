"""
Comprehensive tests for all new optimization modules
Tests: db_pool, circuit_breaker, embedding_cache, exceptions, constants
"""

import os
import sys
import time
import tempfile
import threading
import importlib.util
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from enum import Enum

# Load modules directly without triggering __init__.py
def load_module(module_name, file_name):
    module_path = os.path.join(os.path.dirname(__file__), f'../advanced_rag/{file_name}')
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load all modules
db_pool = load_module("db_pool", "db_pool.py")
circuit_breaker = load_module("circuit_breaker", "circuit_breaker.py")
embedding_cache = load_module("embedding_cache", "embedding_cache.py")
exceptions = load_module("exceptions", "exceptions.py")
constants = load_module("constants", "constants.py")


class TestDatabasePool:
    """Test database connection pooling"""
    
    def setup_method(self):
        """Clean up before each test"""
        if db_pool._pool_instance:
            db_pool.close_pool()
        # Save original DATABASE_URL and unset it
        self.original_db_url = os.environ.get('DATABASE_URL')
        if 'DATABASE_URL' in os.environ:
            del os.environ['DATABASE_URL']
    
    def teardown_method(self):
        """Restore DATABASE_URL"""
        if self.original_db_url:
            os.environ['DATABASE_URL'] = self.original_db_url
    
    def test_initialize_sqlite_pool(self):
        """Test SQLite pool initialization"""
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            pool = db_pool.initialize_pool(sqlite_path=tf.name)
            assert pool is not None
            assert pool.sqlite_path == tf.name
            db_pool.close_pool()
            os.unlink(tf.name)
    
    def test_get_connection_sqlite(self):
        """Test getting SQLite connection"""
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            pool = db_pool.initialize_pool(sqlite_path=tf.name)
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
                cursor.execute("INSERT INTO test VALUES (1)")
            db_pool.close_pool()
            os.unlink(tf.name)
    
    def test_pool_stats(self):
        """Test pool statistics"""
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            pool = db_pool.initialize_pool(sqlite_path=tf.name)
            stats = pool.get_stats()
            assert "type" in stats
            assert stats["type"] == "sqlite"
            db_pool.close_pool()
            os.unlink(tf.name)
    
    def test_close_pool(self):
        """Test pool closure"""
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            db_pool.initialize_pool(sqlite_path=tf.name)
            db_pool.close_pool()
            with pytest.raises(RuntimeError, match="not initialized"):
                db_pool.get_pool()
            os.unlink(tf.name)
    
    def test_pool_context_manager_error_handling(self):
        """Test connection context manager error handling"""
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            pool = db_pool.initialize_pool(sqlite_path=tf.name)
            
            try:
                with pool.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
                    # Force an error
                    cursor.execute("INSERT INTO test VALUES ('not_an_int')")
            except Exception:
                pass  # Expected
            
            # Pool should still be usable
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
            
            db_pool.close_pool()
            os.unlink(tf.name)
    
    def test_database_pool_direct_instantiation(self):
        """Test DatabasePool direct instantiation"""
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            pool = db_pool.DatabasePool(database_url="", sqlite_path=tf.name)
            assert pool is not None
            assert not pool._is_postgres()
            pool.close_all()
            os.unlink(tf.name)


class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    def test_circuit_breaker_config(self):
        """Test CircuitBreakerConfig initialization"""
        config = circuit_breaker.CircuitBreakerConfig(
            max_failures=5,
            window_seconds=60,
            open_duration_seconds=30
        )
        assert config.max_failures == 5
        assert config.window_seconds == 60
        assert config.open_duration_seconds == 30
    
    def test_circuit_breaker_init(self):
        """Test CircuitBreaker initialization"""
        cb = circuit_breaker.CircuitBreaker()
        assert cb._state == circuit_breaker.CircuitState.CLOSED
    
    def test_circuit_breaker_is_open(self):
        """Test is_open check"""
        cb = circuit_breaker.CircuitBreaker()
        assert not cb.is_open()
    
    def test_circuit_breaker_record_failure(self):
        """Test recording failures"""
        cb = circuit_breaker.CircuitBreaker(
            config=circuit_breaker.CircuitBreakerConfig(max_failures=2)
        )
        
        # First failure
        cb.record_failure()
        assert not cb.is_open()
        
        # Second failure should open circuit
        cb.record_failure()
        assert cb.is_open()
    
    def test_circuit_breaker_record_success(self):
        """Test recording success"""
        cb = circuit_breaker.CircuitBreaker()
        cb.record_success()
        assert not cb.is_open()
    
    def test_circuit_breaker_half_open_to_closed(self):
        """Test half-open to closed transition"""
        cb = circuit_breaker.CircuitBreaker(
            config=circuit_breaker.CircuitBreakerConfig(
                max_failures=1,
                open_duration_seconds=0.1,
                half_open_max_calls=2
            )
        )
        
        # Open the circuit
        cb.record_failure()
        assert cb.is_open()
        
        # Wait for half-open
        time.sleep(0.2)
        assert not cb.is_open()  # Now in half-open
        
        # Record enough successes to close
        cb.record_success()
        cb.record_success()
        assert cb._state == circuit_breaker.CircuitState.CLOSED
    
    def test_circuit_breaker_reset(self):
        """Test manual circuit reset"""
        cb = circuit_breaker.CircuitBreaker(
            config=circuit_breaker.CircuitBreakerConfig(max_failures=1)
        )
        
        cb.record_failure()
        assert cb.is_open()
        
        cb.reset()
        assert not cb.is_open()
        assert cb._state == circuit_breaker.CircuitState.CLOSED
    
    def test_circuit_breaker_get_stats(self):
        """Test getting circuit breaker statistics"""
        cb = circuit_breaker.CircuitBreaker()
        cb.record_failure()
        
        stats = cb.get_stats()
        assert "state" in stats
        assert "failures_in_window" in stats
        assert stats["failures_in_window"] == 1


class TestEmbeddingCache:
    """Test embedding cache with LRU and TTL"""
    
    def test_cache_init(self):
        """Test cache initialization"""
        cache = embedding_cache.EmbeddingCache(maxsize=100, ttl_seconds=300)
        assert cache.maxsize == 100
        assert cache.ttl_seconds == 300
        assert cache._stats.hits == 0
        assert cache._stats.misses == 0
    
    def test_cache_get_put(self):
        """Test cache get and put"""
        cache = embedding_cache.EmbeddingCache(maxsize=10)
        
        emb = np.array([0.1, 0.2, 0.3])
        cache.put("test text", "model1", emb)
        
        result = cache.get("test text", "model1")
        assert result is not None
        assert np.array_equal(result, emb)
        assert cache._stats.hits == 1
    
    def test_cache_miss(self):
        """Test cache miss"""
        cache = embedding_cache.EmbeddingCache(maxsize=10)
        
        result = cache.get("nonexistent", "model1")
        assert result is None
        assert cache._stats.misses == 1
    
    def test_cache_disabled(self):
        """Test disabled cache"""
        cache = embedding_cache.EmbeddingCache(maxsize=10, enabled=False)
        
        emb = np.array([0.1, 0.2, 0.3])
        cache.put("test", "model1", emb)
        
        result = cache.get("test", "model1")
        assert result is None
    
    def test_cache_stats_property(self):
        """Test cache statistics"""
        cache = embedding_cache.EmbeddingCache(maxsize=10)
        
        emb = np.array([1.0])
        cache.put("text1", "model1", emb)
        cache.get("text1", "model1")  # hit
        cache.get("text2", "model1")  # miss
        
        stats = cache._stats
        assert stats.hits == 1
        assert stats.misses == 1
        hit_rate = stats.hit_rate
        assert 0.0 <= hit_rate <= 1.0
    
    def test_cache_eviction_tracking(self):
        """Test cache eviction tracking"""
        cache = embedding_cache.EmbeddingCache(maxsize=2)
        
        # Fill cache
        cache.put("text1", "model1", np.array([1.0]))
        cache.put("text2", "model1", np.array([2.0]))
        
        # This should trigger eviction
        cache.put("text3", "model1", np.array([3.0]))
        
        assert cache._stats.evictions >= 1


class TestExceptions:
    """Test custom exception hierarchy"""
    
    def test_rag_exception(self):
        """Test base RAGException"""
        exc = exceptions.RAGException("Test error")
        assert str(exc) == "Test error"
        assert isinstance(exc, Exception)
    
    def test_embedding_generation_error(self):
        """Test EmbeddingGenerationError"""
        exc = exceptions.EmbeddingGenerationError("Embedding failed")
        assert isinstance(exc, exceptions.RAGException)
        assert str(exc) == "Embedding failed"
    
    def test_retrieval_error(self):
        """Test RetrievalError"""
        exc = exceptions.RetrievalError("Retrieval failed")
        assert isinstance(exc, exceptions.RAGException)
    
    def test_circuit_breaker_open_error(self):
        """Test CircuitBreakerOpenError"""
        exc = exceptions.CircuitBreakerOpenError("Circuit open")
        assert isinstance(exc, exceptions.RAGException)
    
    def test_validation_error(self):
        """Test ValidationError"""
        exc = exceptions.ValidationError("Invalid input")
        assert isinstance(exc, exceptions.RAGException)
    
    def test_database_connection_error(self):
        """Test DatabaseConnectionError"""
        exc = exceptions.DatabaseConnectionError("DB connection failed")
        assert isinstance(exc, exceptions.RAGException)
    
    def test_configuration_error(self):
        """Test ConfigurationError"""
        exc = exceptions.ConfigurationError("Config error")
        assert isinstance(exc, exceptions.RAGException)
    
    def test_exception_raising(self):
        """Test raising exceptions"""
        with pytest.raises(exceptions.RAGException, match="Test"):
            raise exceptions.RAGException("Test")


class TestConstants:
    """Test centralized constants"""
    
    def test_chunking_constants(self):
        """Test ChunkingConstants"""
        assert hasattr(constants.ChunkingConstants, 'DEFAULT_BASE_CHUNK_SIZE')
        assert isinstance(constants.ChunkingConstants.DEFAULT_BASE_CHUNK_SIZE, int)
        assert constants.ChunkingConstants.DEFAULT_BASE_CHUNK_SIZE > 0
        assert hasattr(constants.ChunkingConstants, 'DEFAULT_MAX_CHUNK_SIZE')
        assert hasattr(constants.ChunkingConstants, 'DEFAULT_OVERLAP_RATIO')
    
    def test_retrieval_constants(self):
        """Test RetrievalConstants"""
        assert hasattr(constants.RetrievalConstants, 'DEFAULT_TOP_K')
        assert isinstance(constants.RetrievalConstants.DEFAULT_TOP_K, int)
        assert constants.RetrievalConstants.DEFAULT_TOP_K > 0
    
    def test_circuit_breaker_constants(self):
        """Test CircuitBreakerConstants"""
        assert hasattr(constants.CircuitBreakerConstants, 'DEFAULT_MAX_FAILURES')
        assert isinstance(constants.CircuitBreakerConstants.DEFAULT_MAX_FAILURES, int)
    
    def test_database_constants(self):
        """Test DatabaseConstants"""
        assert hasattr(constants.DatabaseConstants, 'DEFAULT_POOL_MIN_CONNECTIONS')
        assert isinstance(constants.DatabaseConstants.DEFAULT_POOL_MIN_CONNECTIONS, int)
        assert constants.DatabaseConstants.DEFAULT_POOL_MIN_CONNECTIONS > 0
    
    def test_performance_constants(self):
        """Test PerformanceConstants"""
        assert hasattr(constants.PerformanceConstants, 'DEFAULT_CACHE_SIZE')
        assert isinstance(constants.PerformanceConstants.DEFAULT_CACHE_SIZE, int)
    
    def test_api_constants(self):
        """Test APIConstants"""
        assert hasattr(constants.APIConstants, 'MAX_QUERY_LENGTH')
        assert isinstance(constants.APIConstants.MAX_QUERY_LENGTH, (int, float))
    
    def test_milvus_constants(self):
        """Test MilvusConstants"""
        assert hasattr(constants.MilvusConstants, 'DEFAULT_HNSW_M')
        assert isinstance(constants.MilvusConstants.DEFAULT_HNSW_M, int)
    
    def test_constants_values_reasonable(self):
        """Test that constant values are reasonable"""
        # Chunk size should be reasonable (not too small or large)
        assert 100 <= constants.ChunkingConstants.DEFAULT_BASE_CHUNK_SIZE <= 10000
        
        # Top K should be reasonable
        assert 1 <= constants.RetrievalConstants.DEFAULT_TOP_K <= 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
