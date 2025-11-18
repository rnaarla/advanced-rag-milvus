"""
Constants and Configuration Values
Centralized constants to eliminate magic numbers
"""

# Chunking Configuration
class ChunkingConstants:
    """Constants for document chunking"""
    DEFAULT_BASE_CHUNK_SIZE = 512
    DEFAULT_MAX_CHUNK_SIZE = 1024
    DEFAULT_MIN_CHUNK_SIZE = 128
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
    """Constants for retrieval operations"""
    DEFAULT_TOP_K = 20
    DEFAULT_RERANK_TOP_K = 5
    DEFAULT_HYBRID_ALPHA = 0.7
    
    # RRF (Reciprocal Rank Fusion) parameter
    RRF_K_PARAMETER = 60
    
    # Weight distributions
    DEFAULT_DENSE_WEIGHT = 0.7
    DEFAULT_SPARSE_WEIGHT = 0.3
    DEFAULT_DOMAIN_WEIGHT = 0.2
    
    # MMR parameters
    DEFAULT_MMR_LAMBDA = 0.7  # Balance between relevance and diversity


# Evaluation Configuration
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


# Performance Configuration
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


# Circuit Breaker Configuration
class CircuitBreakerConstants:
    """Constants for circuit breaker"""
    DEFAULT_MAX_FAILURES = 10
    DEFAULT_WINDOW_SECONDS = 30
    DEFAULT_OPEN_DURATION_SECONDS = 15
    DEFAULT_HALF_OPEN_MAX_CALLS = 3


# Database Configuration
class DatabaseConstants:
    """Constants for database operations"""
    DEFAULT_POOL_MIN_CONNECTIONS = 5
    DEFAULT_POOL_MAX_CONNECTIONS = 20
    DEFAULT_CONNECTION_TIMEOUT_SECONDS = 30
    DEFAULT_RETENTION_DAYS = 90
    
    # Schema version
    CURRENT_SCHEMA_VERSION = 1


# API Configuration
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


# Milvus Configuration
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


# Compliance Configuration
class ComplianceConstants:
    """Constants for compliance and auditing"""
    DEFAULT_RETENTION_DAYS = 90
    MAX_AUDIT_LOG_BATCH_SIZE = 1000
    AUDIT_FLUSH_INTERVAL_SECONDS = 60


# Logging Configuration
class LoggingConstants:
    """Constants for logging"""
    DEFAULT_LOG_LEVEL = "INFO"
    MAX_LOG_MESSAGE_LENGTH = 10000
    LOG_QUEUE_SIZE = 10000
