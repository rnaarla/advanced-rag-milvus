"""
Custom exceptions for Advanced RAG Pipeline.

This module exposes a rich hierarchy of exception types used across the
codebase and expected by the test suite. All custom exceptions inherit
from AdvancedRAGException so callers can catch a single base type.
"""


class AdvancedRAGException(Exception):
    """Base exception for all Advanced RAG errors."""


# Backwards-compatible alias used in some modules/tests
RAGException = AdvancedRAGException


class ValidationError(AdvancedRAGException):
    """Input validation failed."""


class EmbeddingError(AdvancedRAGException):
    """Generic embedding-related error."""


class EmbeddingGenerationError(EmbeddingError):
    """Failed to generate embeddings."""


class IndexingError(AdvancedRAGException):
    """Indexing operation failed."""


class RetrievalError(AdvancedRAGException):
    """Retrieval operation failed."""


class DatabaseError(AdvancedRAGException):
    """Generic database error."""


class DatabaseConnectionError(DatabaseError):
    """Database connection failed."""


class CircuitBreakerOpenError(AdvancedRAGException):
    """Circuit breaker is open, service temporarily unavailable."""


class ConfigurationError(AdvancedRAGException):
    """Invalid configuration."""


class TimeoutError(AdvancedRAGException):
    """Operation timed out."""


class CacheError(AdvancedRAGException):
    """Cache-related error."""


class AuthenticationError(AdvancedRAGException):
    """Authentication or authorization error."""


class RateLimitError(AdvancedRAGException):
    """Rate limit exceeded."""


class ChunkingError(AdvancedRAGException):
    """Document chunking failed."""


class EvaluationError(AdvancedRAGException):
    """Evaluation failed."""


class ComplianceError(AdvancedRAGException):
    """Compliance or audit operation failed."""


class MilvusConnectionError(AdvancedRAGException):
    """Cannot connect to Milvus."""


class MilvusOperationError(AdvancedRAGException):
    """Milvus operation failed."""
