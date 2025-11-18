"""
Tests for custom exceptions module

Tests cover:
- All exception types
- Exception hierarchy
- Error messages
- Exception attributes
"""

import pytest
import sys
import os
import importlib.util

# Load module directly without triggering __init__.py
module_path = os.path.join(os.path.dirname(__file__), '../advanced_rag/exceptions.py')
spec = importlib.util.spec_from_file_location("exceptions", module_path)
exceptions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exceptions)

RAGException = exceptions.RAGException
ValidationError = exceptions.ValidationError
EmbeddingGenerationError = exceptions.EmbeddingGenerationError
IndexingError = exceptions.IndexingError
RetrievalError = exceptions.RetrievalError
DatabaseConnectionError = exceptions.DatabaseConnectionError
CircuitBreakerOpenError = exceptions.CircuitBreakerOpenError
ConfigurationError = exceptions.ConfigurationError
ChunkingError = exceptions.ChunkingError
EvaluationError = exceptions.EvaluationError
ComplianceError = exceptions.ComplianceError
MilvusConnectionError = exceptions.MilvusConnectionError
MilvusOperationError = exceptions.MilvusOperationError


class TestExceptionHierarchy:
    """Test exception inheritance structure"""
    
    def test_base_exception(self):
        """Test base exception"""
        exc = AdvancedRAGException("Base error")
        
        assert str(exc) == "Base error"
        assert isinstance(exc, Exception)
    
    def test_validation_error_inheritance(self):
        """Test ValidationError inherits from base"""
        exc = ValidationError("Invalid input")
        
        assert isinstance(exc, AdvancedRAGException)
        assert isinstance(exc, Exception)
    
    def test_embedding_error_inheritance(self):
        """Test EmbeddingError inherits from base"""
        exc = EmbeddingError("Embedding failed")
        
        assert isinstance(exc, AdvancedRAGException)
        assert isinstance(exc, Exception)
    
    def test_indexing_error_inheritance(self):
        """Test IndexingError inherits from base"""
        exc = IndexingError("Indexing failed")
        
        assert isinstance(exc, AdvancedRAGException)
        assert isinstance(exc, Exception)
    
    def test_retrieval_error_inheritance(self):
        """Test RetrievalError inherits from base"""
        exc = RetrievalError("Retrieval failed")
        
        assert isinstance(exc, AdvancedRAGException)
        assert isinstance(exc, Exception)
    
    def test_database_error_inheritance(self):
        """Test DatabaseError inherits from base"""
        exc = DatabaseError("Database error")
        
        assert isinstance(exc, AdvancedRAGException)
        assert isinstance(exc, Exception)
    
    def test_circuit_breaker_open_error_inheritance(self):
        """Test CircuitBreakerOpenError inherits from base"""
        exc = CircuitBreakerOpenError("Circuit open")
        
        assert isinstance(exc, AdvancedRAGException)
        assert isinstance(exc, Exception)
    
    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inherits from base"""
        exc = ConfigurationError("Config error")
        
        assert isinstance(exc, AdvancedRAGException)
        assert isinstance(exc, Exception)
    
    def test_timeout_error_inheritance(self):
        """Test TimeoutError inherits from base"""
        exc = TimeoutError("Timeout occurred")
        
        assert isinstance(exc, AdvancedRAGException)
        assert isinstance(exc, Exception)
    
    def test_cache_error_inheritance(self):
        """Test CacheError inherits from base"""
        exc = CacheError("Cache error")
        
        assert isinstance(exc, AdvancedRAGException)
        assert isinstance(exc, Exception)
    
    def test_authentication_error_inheritance(self):
        """Test AuthenticationError inherits from base"""
        exc = AuthenticationError("Auth failed")
        
        assert isinstance(exc, AdvancedRAGException)
        assert isinstance(exc, Exception)
    
    def test_rate_limit_error_inheritance(self):
        """Test RateLimitError inherits from base"""
        exc = RateLimitError("Rate limit exceeded")
        
        assert isinstance(exc, AdvancedRAGException)
        assert isinstance(exc, Exception)


class TestExceptionRaising:
    """Test exceptions can be raised and caught"""
    
    def test_raise_base_exception(self):
        """Test raising base exception"""
        with pytest.raises(AdvancedRAGException, match="Test error"):
            raise AdvancedRAGException("Test error")
    
    def test_raise_validation_error(self):
        """Test raising ValidationError"""
        with pytest.raises(ValidationError, match="Invalid input"):
            raise ValidationError("Invalid input")
    
    def test_raise_embedding_error(self):
        """Test raising EmbeddingError"""
        with pytest.raises(EmbeddingError, match="Embedding failed"):
            raise EmbeddingError("Embedding failed")
    
    def test_raise_indexing_error(self):
        """Test raising IndexingError"""
        with pytest.raises(IndexingError, match="Index error"):
            raise IndexingError("Index error")
    
    def test_raise_retrieval_error(self):
        """Test raising RetrievalError"""
        with pytest.raises(RetrievalError, match="Retrieval failed"):
            raise RetrievalError("Retrieval failed")
    
    def test_raise_database_error(self):
        """Test raising DatabaseError"""
        with pytest.raises(DatabaseError, match="DB error"):
            raise DatabaseError("DB error")
    
    def test_raise_circuit_breaker_open_error(self):
        """Test raising CircuitBreakerOpenError"""
        with pytest.raises(CircuitBreakerOpenError, match="Circuit is open"):
            raise CircuitBreakerOpenError("Circuit is open")
    
    def test_raise_configuration_error(self):
        """Test raising ConfigurationError"""
        with pytest.raises(ConfigurationError, match="Bad config"):
            raise ConfigurationError("Bad config")
    
    def test_raise_timeout_error(self):
        """Test raising TimeoutError"""
        with pytest.raises(TimeoutError, match="Operation timed out"):
            raise TimeoutError("Operation timed out")
    
    def test_raise_cache_error(self):
        """Test raising CacheError"""
        with pytest.raises(CacheError, match="Cache miss"):
            raise CacheError("Cache miss")
    
    def test_raise_authentication_error(self):
        """Test raising AuthenticationError"""
        with pytest.raises(AuthenticationError, match="Unauthorized"):
            raise AuthenticationError("Unauthorized")
    
    def test_raise_rate_limit_error(self):
        """Test raising RateLimitError"""
        with pytest.raises(RateLimitError, match="Too many requests"):
            raise RateLimitError("Too many requests")


class TestExceptionCatching:
    """Test exception catching behavior"""
    
    def test_catch_specific_exception(self):
        """Test catching specific exception type"""
        try:
            raise ValidationError("Invalid")
        except ValidationError as e:
            assert str(e) == "Invalid"
    
    def test_catch_base_exception(self):
        """Test catching any AdvancedRAG exception"""
        try:
            raise ValidationError("Invalid")
        except AdvancedRAGException as e:
            assert isinstance(e, ValidationError)
    
    def test_catch_generic_exception(self):
        """Test catching as generic Exception"""
        try:
            raise DatabaseError("DB error")
        except Exception as e:
            assert isinstance(e, DatabaseError)
            assert isinstance(e, AdvancedRAGException)
    
    def test_multiple_exception_types(self):
        """Test handling multiple exception types"""
        def raise_different_errors(error_type):
            if error_type == "validation":
                raise ValidationError("Validation failed")
            elif error_type == "database":
                raise DatabaseError("Database failed")
            elif error_type == "timeout":
                raise TimeoutError("Timeout")
        
        # Catch all as base exception
        for error_type in ["validation", "database", "timeout"]:
            try:
                raise_different_errors(error_type)
            except AdvancedRAGException as e:
                assert str(e) in ["Validation failed", "Database failed", "Timeout"]


class TestExceptionMessages:
    """Test exception message handling"""
    
    def test_empty_message(self):
        """Test exception with empty message"""
        exc = ValidationError("")
        assert str(exc) == ""
    
    def test_long_message(self):
        """Test exception with long message"""
        long_msg = "A" * 1000
        exc = ValidationError(long_msg)
        assert str(exc) == long_msg
    
    def test_unicode_message(self):
        """Test exception with unicode characters"""
        exc = ValidationError("Error: 你好 世界")
        assert "你好" in str(exc)
    
    def test_multiline_message(self):
        """Test exception with multiline message"""
        msg = "Line 1\nLine 2\nLine 3"
        exc = DatabaseError(msg)
        assert str(exc) == msg
    
    def test_formatted_message(self):
        """Test exception with formatted message"""
        user_id = 123
        action = "delete"
        exc = ValidationError(f"User {user_id} cannot perform {action}")
        assert "123" in str(exc)
        assert "delete" in str(exc)


class TestExceptionChaining:
    """Test exception chaining with 'from' clause"""
    
    def test_exception_cause(self):
        """Test exception chaining preserves cause"""
        original = ValueError("Original error")
        
        try:
            try:
                raise original
            except ValueError as e:
                raise DatabaseError("Database error") from e
        except DatabaseError as e:
            assert e.__cause__ is original
    
    def test_exception_context(self):
        """Test exception context is preserved"""
        try:
            try:
                raise ValueError("First error")
            except:
                raise DatabaseError("Second error")
        except DatabaseError as e:
            assert e.__context__ is not None


class TestAllExceptionsExist:
    """Test all documented exceptions are defined"""
    
    def test_all_exception_types_defined(self):
        """Test all exception types can be imported and instantiated"""
        exception_types = [
            AdvancedRAGException,
            ValidationError,
            EmbeddingError,
            IndexingError,
            RetrievalError,
            DatabaseError,
            CircuitBreakerOpenError,
            ConfigurationError,
            TimeoutError,
            CacheError,
            AuthenticationError,
            RateLimitError
        ]
        
        for exc_type in exception_types:
            # Should be able to instantiate
            exc = exc_type("Test")
            assert isinstance(exc, Exception)
            assert str(exc) == "Test"
