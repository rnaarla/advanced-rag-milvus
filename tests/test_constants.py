"""
Tests for constants module

Tests cover:
- All constant classes exist
- Values are correct types
- Values are reasonable
- Constants are accessible
"""

import pytest
import sys
import os
import importlib.util

# Load module directly without triggering __init__.py
module_path = os.path.join(os.path.dirname(__file__), '../src/advanced_rag/constants.py')
spec = importlib.util.spec_from_file_location("constants", module_path)
constants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constants)

ChunkingConstants = constants.ChunkingConstants
RetrievalConstants = constants.RetrievalConstants
EvaluationConstants = constants.EvaluationConstants
PerformanceConstants = constants.PerformanceConstants
CircuitBreakerConstants = constants.CircuitBreakerConstants
DatabaseConstants = constants.DatabaseConstants
APIConstants = constants.APIConstants
MilvusConstants = constants.MilvusConstants
ComplianceConstants = constants.ComplianceConstants
LoggingConstants = constants.LoggingConstants

# Additional constant classes introduced for advanced configuration
EmbeddingConstants = constants.EmbeddingConstants
IndexingConstants = constants.IndexingConstants
RateLimitConstants = constants.RateLimitConstants
MetricsConstants = constants.MetricsConstants


class TestChunkingConstants:
    """Test ChunkingConstants class"""
    
    def test_max_chunk_tokens(self):
        """Test MAX_CHUNK_TOKENS is defined and reasonable"""
        assert hasattr(ChunkingConstants, 'MAX_CHUNK_TOKENS')
        assert isinstance(ChunkingConstants.MAX_CHUNK_TOKENS, int)
        assert ChunkingConstants.MAX_CHUNK_TOKENS > 0
        assert ChunkingConstants.MAX_CHUNK_TOKENS <= 10000  # Reasonable upper bound
    
    def test_min_chunk_tokens(self):
        """Test MIN_CHUNK_TOKENS is defined and reasonable"""
        assert hasattr(ChunkingConstants, 'MIN_CHUNK_TOKENS')
        assert isinstance(ChunkingConstants.MIN_CHUNK_TOKENS, int)
        assert ChunkingConstants.MIN_CHUNK_TOKENS > 0
        assert ChunkingConstants.MIN_CHUNK_TOKENS < ChunkingConstants.MAX_CHUNK_TOKENS
    
    def test_overlap_tokens(self):
        """Test OVERLAP_TOKENS is defined and reasonable"""
        assert hasattr(ChunkingConstants, 'OVERLAP_TOKENS')
        assert isinstance(ChunkingConstants.OVERLAP_TOKENS, int)
        assert ChunkingConstants.OVERLAP_TOKENS >= 0
    
    def test_batch_size(self):
        """Test BATCH_SIZE is defined and reasonable"""
        assert hasattr(ChunkingConstants, 'BATCH_SIZE')
        assert isinstance(ChunkingConstants.BATCH_SIZE, int)
        assert ChunkingConstants.BATCH_SIZE > 0


class TestEmbeddingConstants:
    """Test EmbeddingConstants class"""
    
    def test_semantic_dim(self):
        """Test SEMANTIC_DIM is defined and reasonable"""
        assert hasattr(EmbeddingConstants, 'SEMANTIC_DIM')
        assert isinstance(EmbeddingConstants.SEMANTIC_DIM, int)
        assert EmbeddingConstants.SEMANTIC_DIM > 0
        # Common dimensions: 384, 768, 1024, 1536
        assert EmbeddingConstants.SEMANTIC_DIM in [384, 512, 768, 1024, 1536, 4096]
    
    def test_sparse_dim(self):
        """Test SPARSE_DIM is defined and reasonable"""
        assert hasattr(EmbeddingConstants, 'SPARSE_DIM')
        assert isinstance(EmbeddingConstants.SPARSE_DIM, int)
        assert EmbeddingConstants.SPARSE_DIM > 0
    
    def test_domain_dim(self):
        """Test DOMAIN_DIM is defined and reasonable"""
        assert hasattr(EmbeddingConstants, 'DOMAIN_DIM')
        assert isinstance(EmbeddingConstants.DOMAIN_DIM, int)
        assert EmbeddingConstants.DOMAIN_DIM > 0
    
    def test_cache_max_size(self):
        """Test CACHE_MAX_SIZE is defined and reasonable"""
        assert hasattr(EmbeddingConstants, 'CACHE_MAX_SIZE')
        assert isinstance(EmbeddingConstants.CACHE_MAX_SIZE, int)
        assert EmbeddingConstants.CACHE_MAX_SIZE > 0
    
    def test_cache_ttl_seconds(self):
        """Test CACHE_TTL_SECONDS is defined and reasonable"""
        assert hasattr(EmbeddingConstants, 'CACHE_TTL_SECONDS')
        assert isinstance(EmbeddingConstants.CACHE_TTL_SECONDS, int)
        assert EmbeddingConstants.CACHE_TTL_SECONDS > 0


class TestRetrievalConstants:
    """Test RetrievalConstants class"""
    
    def test_default_top_k(self):
        """Test DEFAULT_TOP_K is defined and reasonable"""
        assert hasattr(RetrievalConstants, 'DEFAULT_TOP_K')
        assert isinstance(RetrievalConstants.DEFAULT_TOP_K, int)
        assert RetrievalConstants.DEFAULT_TOP_K > 0
        assert RetrievalConstants.DEFAULT_TOP_K <= 100
    
    def test_max_top_k(self):
        """Test MAX_TOP_K is defined and reasonable"""
        assert hasattr(RetrievalConstants, 'MAX_TOP_K')
        assert isinstance(RetrievalConstants.MAX_TOP_K, int)
        assert RetrievalConstants.MAX_TOP_K >= RetrievalConstants.DEFAULT_TOP_K
    
    def test_timeout_seconds(self):
        """Test TIMEOUT_SECONDS is defined and reasonable"""
        assert hasattr(RetrievalConstants, 'TIMEOUT_SECONDS')
        assert isinstance(RetrievalConstants.TIMEOUT_SECONDS, (int, float))
        assert RetrievalConstants.TIMEOUT_SECONDS > 0
    
    def test_semantic_weight(self):
        """Test SEMANTIC_WEIGHT is defined and reasonable"""
        assert hasattr(RetrievalConstants, 'SEMANTIC_WEIGHT')
        assert isinstance(RetrievalConstants.SEMANTIC_WEIGHT, (int, float))
        assert 0 <= RetrievalConstants.SEMANTIC_WEIGHT <= 1
    
    def test_sparse_weight(self):
        """Test SPARSE_WEIGHT is defined and reasonable"""
        assert hasattr(RetrievalConstants, 'SPARSE_WEIGHT')
        assert isinstance(RetrievalConstants.SPARSE_WEIGHT, (int, float))
        assert 0 <= RetrievalConstants.SPARSE_WEIGHT <= 1
    
    def test_domain_weight(self):
        """Test DOMAIN_WEIGHT is defined and reasonable"""
        assert hasattr(RetrievalConstants, 'DOMAIN_WEIGHT')
        assert isinstance(RetrievalConstants.DOMAIN_WEIGHT, (int, float))
        assert 0 <= RetrievalConstants.DOMAIN_WEIGHT <= 1


class TestIndexingConstants:
    """Test IndexingConstants class"""
    
    def test_batch_size(self):
        """Test BATCH_SIZE is defined and reasonable"""
        assert hasattr(IndexingConstants, 'BATCH_SIZE')
        assert isinstance(IndexingConstants.BATCH_SIZE, int)
        assert IndexingConstants.BATCH_SIZE > 0
    
    def test_retry_attempts(self):
        """Test RETRY_ATTEMPTS is defined and reasonable"""
        assert hasattr(IndexingConstants, 'RETRY_ATTEMPTS')
        assert isinstance(IndexingConstants.RETRY_ATTEMPTS, int)
        assert IndexingConstants.RETRY_ATTEMPTS >= 0
    
    def test_retry_wait_min(self):
        """Test RETRY_WAIT_MIN is defined and reasonable"""
        assert hasattr(IndexingConstants, 'RETRY_WAIT_MIN')
        assert isinstance(IndexingConstants.RETRY_WAIT_MIN, (int, float))
        assert IndexingConstants.RETRY_WAIT_MIN > 0
    
    def test_retry_wait_max(self):
        """Test RETRY_WAIT_MAX is defined and reasonable"""
        assert hasattr(IndexingConstants, 'RETRY_WAIT_MAX')
        assert isinstance(IndexingConstants.RETRY_WAIT_MAX, (int, float))
        assert IndexingConstants.RETRY_WAIT_MAX >= IndexingConstants.RETRY_WAIT_MIN
    
    def test_milvus_timeout_seconds(self):
        """Test MILVUS_TIMEOUT_SECONDS is defined and reasonable"""
        assert hasattr(IndexingConstants, 'MILVUS_TIMEOUT_SECONDS')
        assert isinstance(IndexingConstants.MILVUS_TIMEOUT_SECONDS, (int, float))
        assert IndexingConstants.MILVUS_TIMEOUT_SECONDS > 0
    
    def test_thread_pool_workers(self):
        """Test THREAD_POOL_WORKERS is defined and reasonable"""
        assert hasattr(IndexingConstants, 'THREAD_POOL_WORKERS')
        assert isinstance(IndexingConstants.THREAD_POOL_WORKERS, int)
        assert IndexingConstants.THREAD_POOL_WORKERS > 0
        assert IndexingConstants.THREAD_POOL_WORKERS <= 32  # Reasonable upper bound


class TestCircuitBreakerConstants:
    """Test CircuitBreakerConstants class"""
    
    def test_failure_threshold(self):
        """Test FAILURE_THRESHOLD is defined and reasonable"""
        assert hasattr(CircuitBreakerConstants, 'FAILURE_THRESHOLD')
        assert isinstance(CircuitBreakerConstants.FAILURE_THRESHOLD, int)
        assert CircuitBreakerConstants.FAILURE_THRESHOLD > 0
    
    def test_timeout_seconds(self):
        """Test TIMEOUT_SECONDS is defined and reasonable"""
        assert hasattr(CircuitBreakerConstants, 'TIMEOUT_SECONDS')
        assert isinstance(CircuitBreakerConstants.TIMEOUT_SECONDS, (int, float))
        assert CircuitBreakerConstants.TIMEOUT_SECONDS > 0
    
    def test_success_threshold(self):
        """Test SUCCESS_THRESHOLD is defined and reasonable"""
        assert hasattr(CircuitBreakerConstants, 'SUCCESS_THRESHOLD')
        assert isinstance(CircuitBreakerConstants.SUCCESS_THRESHOLD, int)
        assert CircuitBreakerConstants.SUCCESS_THRESHOLD > 0
    
    def test_half_open_max_calls(self):
        """Test HALF_OPEN_MAX_CALLS is defined and reasonable"""
        assert hasattr(CircuitBreakerConstants, 'HALF_OPEN_MAX_CALLS')
        assert isinstance(CircuitBreakerConstants.HALF_OPEN_MAX_CALLS, int)
        assert CircuitBreakerConstants.HALF_OPEN_MAX_CALLS > 0


class TestDatabaseConstants:
    """Test DatabaseConstants class"""
    
    def test_pool_min_connections(self):
        """Test POOL_MIN_CONNECTIONS is defined and reasonable"""
        assert hasattr(DatabaseConstants, 'POOL_MIN_CONNECTIONS')
        assert isinstance(DatabaseConstants.POOL_MIN_CONNECTIONS, int)
        assert DatabaseConstants.POOL_MIN_CONNECTIONS > 0
    
    def test_pool_max_connections(self):
        """Test POOL_MAX_CONNECTIONS is defined and reasonable"""
        assert hasattr(DatabaseConstants, 'POOL_MAX_CONNECTIONS')
        assert isinstance(DatabaseConstants.POOL_MAX_CONNECTIONS, int)
        assert DatabaseConstants.POOL_MAX_CONNECTIONS >= DatabaseConstants.POOL_MIN_CONNECTIONS
    
    def test_connection_timeout_seconds(self):
        """Test CONNECTION_TIMEOUT_SECONDS is defined and reasonable"""
        assert hasattr(DatabaseConstants, 'CONNECTION_TIMEOUT_SECONDS')
        assert isinstance(DatabaseConstants.CONNECTION_TIMEOUT_SECONDS, (int, float))
        assert DatabaseConstants.CONNECTION_TIMEOUT_SECONDS > 0


class TestRateLimitConstants:
    """Test RateLimitConstants class"""
    
    def test_ingest_per_minute(self):
        """Test INGEST_PER_MINUTE is defined and reasonable"""
        assert hasattr(RateLimitConstants, 'INGEST_PER_MINUTE')
        assert isinstance(RateLimitConstants.INGEST_PER_MINUTE, int)
        assert RateLimitConstants.INGEST_PER_MINUTE > 0
    
    def test_retrieve_per_minute(self):
        """Test RETRIEVE_PER_MINUTE is defined and reasonable"""
        assert hasattr(RateLimitConstants, 'RETRIEVE_PER_MINUTE')
        assert isinstance(RateLimitConstants.RETRIEVE_PER_MINUTE, int)
        assert RateLimitConstants.RETRIEVE_PER_MINUTE > 0
    
    def test_chat_per_minute(self):
        """Test CHAT_PER_MINUTE is defined and reasonable"""
        assert hasattr(RateLimitConstants, 'CHAT_PER_MINUTE')
        assert isinstance(RateLimitConstants.CHAT_PER_MINUTE, int)
        assert RateLimitConstants.CHAT_PER_MINUTE > 0


class TestMetricsConstants:
    """Test MetricsConstants class"""
    
    def test_latency_buckets(self):
        """Test LATENCY_BUCKETS is defined and reasonable"""
        assert hasattr(MetricsConstants, 'LATENCY_BUCKETS')
        assert isinstance(MetricsConstants.LATENCY_BUCKETS, (list, tuple))
        assert len(MetricsConstants.LATENCY_BUCKETS) > 0
        
        # Should be sorted
        assert list(MetricsConstants.LATENCY_BUCKETS) == sorted(MetricsConstants.LATENCY_BUCKETS)
        
        # All should be positive numbers
        for bucket in MetricsConstants.LATENCY_BUCKETS:
            assert isinstance(bucket, (int, float))
            assert bucket > 0
    
    def test_embedding_latency_buckets(self):
        """Test EMBEDDING_LATENCY_BUCKETS is defined and reasonable"""
        assert hasattr(MetricsConstants, 'EMBEDDING_LATENCY_BUCKETS')
        assert isinstance(MetricsConstants.EMBEDDING_LATENCY_BUCKETS, (list, tuple))
        assert len(MetricsConstants.EMBEDDING_LATENCY_BUCKETS) > 0


class TestConstantsIntegrity:
    """Test overall integrity of constants"""
    
    def test_all_classes_exist(self):
        """Test all constant classes are defined"""
        classes = [
            ChunkingConstants,
            EmbeddingConstants,
            RetrievalConstants,
            IndexingConstants,
            CircuitBreakerConstants,
            DatabaseConstants,
            RateLimitConstants,
            MetricsConstants
        ]
        
        for cls in classes:
            assert cls is not None
    
    def test_no_negative_values(self):
        """Test no constants are negative"""
        # Check various numeric constants
        assert ChunkingConstants.MAX_CHUNK_TOKENS >= 0
        assert ChunkingConstants.MIN_CHUNK_TOKENS >= 0
        assert EmbeddingConstants.SEMANTIC_DIM >= 0
        assert RetrievalConstants.DEFAULT_TOP_K >= 0
        assert IndexingConstants.BATCH_SIZE >= 0
        assert CircuitBreakerConstants.FAILURE_THRESHOLD >= 0
        assert DatabaseConstants.POOL_MIN_CONNECTIONS >= 0
        assert RateLimitConstants.INGEST_PER_MINUTE >= 0
    
    def test_logical_relationships(self):
        """Test logical relationships between constants"""
        # Min < Max
        assert ChunkingConstants.MIN_CHUNK_TOKENS < ChunkingConstants.MAX_CHUNK_TOKENS
        assert RetrievalConstants.DEFAULT_TOP_K <= RetrievalConstants.MAX_TOP_K
        assert DatabaseConstants.POOL_MIN_CONNECTIONS <= DatabaseConstants.POOL_MAX_CONNECTIONS
        assert IndexingConstants.RETRY_WAIT_MIN <= IndexingConstants.RETRY_WAIT_MAX
    
    def test_weights_sum_reasonable(self):
        """Test retrieval weights are reasonable"""
        # Weights don't need to sum to 1, but should be in [0, 1]
        assert 0 <= RetrievalConstants.SEMANTIC_WEIGHT <= 1
        assert 0 <= RetrievalConstants.SPARSE_WEIGHT <= 1
        assert 0 <= RetrievalConstants.DOMAIN_WEIGHT <= 1
