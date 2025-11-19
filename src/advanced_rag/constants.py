"""
Constants and Configuration Values
Centralized constants to eliminate magic numbers
"""

class ChunkingConstants:
    """
    Constants for document chunking.

    This class is intentionally aligned with the expectations in tests:
    - MAX_CHUNK_TOKENS / MIN_CHUNK_TOKENS / OVERLAP_TOKENS
    - BATCH_SIZE for batch processing
    """

    # Token-based sizes used by tests
    MAX_CHUNK_TOKENS = 1024
    MIN_CHUNK_TOKENS = 128
    OVERLAP_TOKENS = 64
    BATCH_SIZE = 32

    # Backwards-compatible names used elsewhere in the codebase
    DEFAULT_BASE_CHUNK_SIZE = 512
    DEFAULT_MAX_CHUNK_SIZE = MAX_CHUNK_TOKENS
    DEFAULT_MIN_CHUNK_SIZE = MIN_CHUNK_TOKENS
    DEFAULT_OVERLAP_RATIO = 0.15

    # Adaptive chunking thresholds
    ENTROPY_HIGH_THRESHOLD = 0.8
    ENTROPY_LOW_THRESHOLD = 0.4
    ENTROPY_HIGH_MULTIPLIER = 1.3
    ENTROPY_LOW_MULTIPLIER = 0.8

    REDUNDANCY_HIGH_THRESHOLD = 0.6
    REDUNDANCY_MULTIPLIER = 0.7

    DOMAIN_DENSITY_HIGH_THRESHOLD = 0.3
    DOMAIN_DENSITY_MULTIPLIER = 0.85

    COHERENCE_LOW_THRESHOLD = 0.3
    COHERENCE_MULTIPLIER = 0.75


# Retrieval Configuration
class RetrievalConstants:
    """Constants for retrieval operations."""

    # Top-k configuration
    DEFAULT_TOP_K = 20
    MAX_TOP_K = 100

    DEFAULT_RERANK_TOP_K = 5
    DEFAULT_HYBRID_ALPHA = 0.7

    # End-to-end timeout (seconds) for retrieval operations
    TIMEOUT_SECONDS = 0.3  # 300ms, consistent with PerformanceConstants

    # RRF (Reciprocal Rank Fusion) parameter
    RRF_K_PARAMETER = 60

    # Weight distributions (semantics aligned with tests)
    DEFAULT_DENSE_WEIGHT = 0.7
    DEFAULT_SPARSE_WEIGHT = 0.3
    DEFAULT_DOMAIN_WEIGHT = 0.2

    SEMANTIC_WEIGHT = DEFAULT_DENSE_WEIGHT
    SPARSE_WEIGHT = DEFAULT_SPARSE_WEIGHT
    DOMAIN_WEIGHT = DEFAULT_DOMAIN_WEIGHT

    # MMR parameters
    DEFAULT_MMR_LAMBDA = 0.7  # Balance between relevance and diversity


class EvaluationConstants:
    """Constants for evaluation metrics"""
    # Hallucination risk weights
    HALLUCINATION_SCORE_RISK_WEIGHT = 0.25
    HALLUCINATION_DIVERSITY_RISK_WEIGHT = 0.2
    HALLUCINATION_TOP_SCORE_RISK_WEIGHT = 0.3
    HALLUCINATION_COVERAGE_RISK_WEIGHT = 0.25
    
    # Drift detection thresholds
    DEFAULT_DRIFT_THRESHOLD = 0.15
    CRITICAL_DRIFT_THRESHOLD = 0.3
    EMBEDDING_DIVERGENCE_THRESHOLD = 0.2
    DISTRIBUTION_SHIFT_THRESHOLD = 0.25
    
    # History limits
    MAX_HISTORY_EMBEDDINGS = 1000
    MAX_HISTORY_DISTRIBUTIONS = 1000
    MAX_HISTORY_TIMESTAMPS = 1000


class PerformanceConstants:
    """Constants for performance targets and limits"""
    TARGET_LATENCY_MS = 80.0
    WARNING_LATENCY_MS = 100.0
    CRITICAL_LATENCY_MS = 200.0
    
    DEFAULT_MAX_CONCURRENCY = 64
    DEFAULT_RETRIEVE_TIMEOUT_MS = 300
    DEFAULT_EMBEDDING_TIMEOUT_MS = 5000
    
    # Batch sizes
    DEFAULT_BATCH_SIZE = 32
    MAX_BATCH_SIZE = 128
    
    # Cache settings
    DEFAULT_CACHE_SIZE = 10000
    DEFAULT_CACHE_TTL_SECONDS = 3600


class CircuitBreakerConstants:
    """
    Constants for circuit breaker.

    Exposes both DEFAULT_* names used by the codebase and the shorter
    names used in the tests (FAILURE_THRESHOLD, TIMEOUT_SECONDS, etc.).
    """

    DEFAULT_MAX_FAILURES = 10
    DEFAULT_WINDOW_SECONDS = 30
    DEFAULT_OPEN_DURATION_SECONDS = 15
    DEFAULT_HALF_OPEN_MAX_CALLS = 3

    # Short aliases expected by tests
    FAILURE_THRESHOLD = DEFAULT_MAX_FAILURES
    TIMEOUT_SECONDS = float(DEFAULT_WINDOW_SECONDS)
    SUCCESS_THRESHOLD = 2
    HALF_OPEN_MAX_CALLS = DEFAULT_HALF_OPEN_MAX_CALLS


class DatabaseConstants:
    """Constants for database operations."""

    DEFAULT_POOL_MIN_CONNECTIONS = 5
    DEFAULT_POOL_MAX_CONNECTIONS = 20
    DEFAULT_CONNECTION_TIMEOUT_SECONDS = 30
    DEFAULT_RETENTION_DAYS = 90

    # Short aliases expected by tests
    POOL_MIN_CONNECTIONS = DEFAULT_POOL_MIN_CONNECTIONS
    POOL_MAX_CONNECTIONS = DEFAULT_POOL_MAX_CONNECTIONS
    CONNECTION_TIMEOUT_SECONDS = float(DEFAULT_CONNECTION_TIMEOUT_SECONDS)

    # Schema version
    CURRENT_SCHEMA_VERSION = 1


class APIConstants:
    """Constants for API service"""
    # Rate limits
    DEFAULT_RATE_LIMIT_PER_MINUTE = 100
    DEFAULT_RATE_LIMIT_PER_HOUR = 1000
    RETRIEVE_RATE_LIMIT_PER_MINUTE = 20
    INGEST_RATE_LIMIT_PER_MINUTE = 10
    CHAT_RATE_LIMIT_PER_MINUTE = 30
    
    # Request limits
    MAX_DOCUMENT_TEXT_LENGTH = 1_000_000
    MAX_QUERY_LENGTH = 10_000
    MAX_METADATA_SIZE_BYTES = 10_000
    MAX_DOCUMENTS_PER_REQUEST = 1000
    MIN_DOCUMENTS_PER_REQUEST = 1
    
    # Health check intervals
    HEALTH_CHECK_INTERVAL_SECONDS = 30
    HEALTH_CHECK_TIMEOUT_SECONDS = 5


class MilvusConstants:
    """Constants for Milvus operations"""
    # Index parameters
    DEFAULT_HNSW_M = 16
    DEFAULT_HNSW_EF_CONSTRUCTION = 200
    DEFAULT_HNSW_EF = 64
    
    # Dimensions
    DEFAULT_SEMANTIC_DIM = 1536  # OpenAI ada-002
    DEFAULT_SPARSE_DIM = 10000
    DEFAULT_DOMAIN_DIM = 768
    
    # Collection settings
    DEFAULT_NUM_SHARDS = 4
    MAX_VARCHAR_LENGTH = 65535
    MAX_METADATA_JSON_LENGTH = 10000
    
    # Search parameters
    DEFAULT_SEARCH_TIMEOUT_SECONDS = 5.0
    DEFAULT_INSERT_TIMEOUT_SECONDS = 30.0
    SPARSE_DROP_RATIO_SEARCH = 0.2


class ComplianceConstants:
    """Constants for compliance and auditing"""
    DEFAULT_RETENTION_DAYS = 90
    MAX_AUDIT_LOG_BATCH_SIZE = 1000
    AUDIT_FLUSH_INTERVAL_SECONDS = 60


class LoggingConstants:
    """Constants for logging"""
    DEFAULT_LOG_LEVEL = "INFO"
    MAX_LOG_MESSAGE_LENGTH = 10000
    LOG_QUEUE_SIZE = 10000


class EmbeddingConstants:
    """
    Constants for embedding configuration.

    Mostly a thin wrapper around Milvus / Performance constants, exposed
    with names expected by the tests.
    """

    SEMANTIC_DIM = MilvusConstants.DEFAULT_SEMANTIC_DIM
    SPARSE_DIM = MilvusConstants.DEFAULT_SPARSE_DIM
    DOMAIN_DIM = MilvusConstants.DEFAULT_DOMAIN_DIM

    CACHE_MAX_SIZE = PerformanceConstants.DEFAULT_CACHE_SIZE
    CACHE_TTL_SECONDS = PerformanceConstants.DEFAULT_CACHE_TTL_SECONDS


class IndexingConstants:
    """
    Constants for indexing operations.
    Standalone to keep tests decoupled from Milvus-specific knobs.
    """

    BATCH_SIZE = 64
    RETRY_ATTEMPTS = 3
    RETRY_WAIT_MIN = 0.5
    RETRY_WAIT_MAX = 5.0
    MILVUS_TIMEOUT_SECONDS = 5.0
    THREAD_POOL_WORKERS = 8


class RateLimitConstants:
    """Thin wrapper mapping API rate limit constants to shorter names."""

    INGEST_PER_MINUTE = APIConstants.INGEST_RATE_LIMIT_PER_MINUTE
    RETRIEVE_PER_MINUTE = APIConstants.RETRIEVE_RATE_LIMIT_PER_MINUTE
    CHAT_PER_MINUTE = APIConstants.CHAT_RATE_LIMIT_PER_MINUTE


class MetricsConstants:
    """Buckets and limits for metrics histograms."""

    # Representative latency buckets in milliseconds for request latency
    LATENCY_BUCKETS = [10, 25, 50, 75, 100, 150, 250, 500, 1000]

    # Buckets for embedding latency (typically slower)
    EMBEDDING_LATENCY_BUCKETS = [20, 50, 100, 200, 400, 800, 1600]
