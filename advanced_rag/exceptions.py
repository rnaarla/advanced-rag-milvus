"""
Custom exceptions for Advanced RAG Pipeline
"""


class RAGException(Exception):
    """Base exception for RAG pipeline"""
    pass


class EmbeddingGenerationError(RAGException):
    """Failed to generate embeddings"""
    pass


class RetrievalError(RAGException):
    """Retrieval operation failed"""
    pass


class MilvusConnectionError(RAGException):
    """Cannot connect to Milvus"""
    pass


class MilvusOperationError(RAGException):
    """Milvus operation failed"""
    pass


class ValidationError(RAGException):
    """Input validation failed"""
    pass


class CircuitBreakerOpenError(RAGException):
    """Circuit breaker is open, service temporarily unavailable"""
    pass


class DatabaseConnectionError(RAGException):
    """Database connection failed"""
    pass


class ConfigurationError(RAGException):
    """Invalid configuration"""
    pass


class ChunkingError(RAGException):
    """Document chunking failed"""
    pass


class IndexingError(RAGException):
    """Indexing operation failed"""
    pass


class EvaluationError(RAGException):
    """Evaluation failed"""
    pass


class ComplianceError(RAGException):
    """Compliance or audit operation failed"""
    pass
