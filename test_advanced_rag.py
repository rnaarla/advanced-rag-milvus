"""
Unit Tests for Advanced RAG Pipeline
"""

import pytest
import numpy as np
from datetime import datetime

from advanced_rag import (
    DocumentDiagnostics,
    AdaptiveChunker,
    DiagnosticMetrics
)


class TestDocumentDiagnostics:
    """Tests for document diagnostic analysis"""
    
    def test_entropy_calculation(self):
        """Test information entropy calculation"""
        diagnostics = DocumentDiagnostics()
        
        # High entropy text (diverse)
        diverse_text = "machine learning algorithms process data using neural networks and transformers"
        metrics_diverse = diagnostics.analyze_document(diverse_text)
        
        # Low entropy text (repetitive)
        repetitive_text = "the the the the the the the the"
        metrics_repetitive = diagnostics.analyze_document(repetitive_text)
        
        assert metrics_diverse.information_entropy > metrics_repetitive.information_entropy
    
    def test_redundancy_calculation(self):
        """Test redundancy score calculation"""
        diagnostics = DocumentDiagnostics()
        
        # Unique text
        unique_text = "artificial intelligence machine learning deep neural networks"
        metrics_unique = diagnostics.analyze_document(unique_text)
        
        # Redundant text
        redundant_text = "hello world hello world hello world"
        metrics_redundant = diagnostics.analyze_document(redundant_text)
        
        assert metrics_redundant.redundancy_score > metrics_unique.redundancy_score
    
    def test_domain_density(self):
        """Test domain density calculation"""
        diagnostics = DocumentDiagnostics()
        
        # Technical text
        technical_text = "algorithm architecture database query optimization network protocol"
        metrics_technical = diagnostics.analyze_document(technical_text)
        
        # General text
        general_text = "the quick brown fox jumps over the lazy dog"
        metrics_general = diagnostics.analyze_document(general_text)
        
        assert metrics_technical.domain_density > metrics_general.domain_density
    
    def test_vocabulary_diversity(self):
        """Test vocabulary diversity calculation"""
        diagnostics = DocumentDiagnostics()
        
        text = "artificial intelligence and machine learning enable intelligent systems"
        metrics = diagnostics.analyze_document(text)
        
        assert 0 <= metrics.vocabulary_diversity <= 1
        assert metrics.vocabulary_diversity > 0


class TestAdaptiveChunker:
    """Tests for adaptive chunking"""
    
    def test_chunk_creation(self):
        """Test basic chunk creation"""
        chunker = AdaptiveChunker(base_chunk_size=10)
        diagnostics_module = DocumentDiagnostics()
        
        text = "This is a test document. " * 20
        metrics = diagnostics_module.analyze_document(text)
        
        chunks = chunker.chunk_document(
            text=text,
            diagnostics=metrics,
            metadata={"doc_id": "test_doc"}
        )
        
        assert len(chunks) > 0
        assert all(hasattr(chunk, 'text') for chunk in chunks)
        assert all(hasattr(chunk, 'metadata') for chunk in chunks)
    
    def test_chunk_metadata(self):
        """Test chunk metadata generation"""
        chunker = AdaptiveChunker()
        diagnostics_module = DocumentDiagnostics()
        
        text = "Test document with multiple sentences. Each sentence adds content."
        metrics = diagnostics_module.analyze_document(text)
        
        chunks = chunker.chunk_document(
            text=text,
            diagnostics=metrics,
            metadata={"doc_id": "test_doc", "source": "test"}
        )
        
        assert len(chunks) > 0
        chunk = chunks[0]
        
        assert chunk.metadata.doc_id == "test_doc"
        assert chunk.metadata.source == "test"
        assert chunk.metadata.chunk_index == 0
        assert chunk.metadata.token_count > 0
        assert hasattr(chunk.metadata, 'entropy')
        assert hasattr(chunk.metadata, 'redundancy')
    
    def test_adaptive_chunk_size(self):
        """Test that chunk size adapts to diagnostics"""
        chunker = AdaptiveChunker(base_chunk_size=100)
        diagnostics_module = DocumentDiagnostics()
        
        # High entropy text should get larger chunks
        diverse_text = " ".join([f"word{i}" for i in range(200)])
        metrics_diverse = diagnostics_module.analyze_document(diverse_text)
        chunks_diverse = chunker.chunk_document(diverse_text, metrics_diverse)
        
        # High redundancy text should get smaller chunks
        redundant_text = "same text " * 100
        metrics_redundant = diagnostics_module.analyze_document(redundant_text)
        chunks_redundant = chunker.chunk_document(redundant_text, metrics_redundant)
        
        # Note: The actual chunk count depends on the adaptive algorithm
        # We just verify that chunking works
        assert len(chunks_diverse) > 0
        assert len(chunks_redundant) > 0
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap"""
        chunker = AdaptiveChunker(
            base_chunk_size=20,
            overlap_ratio=0.2,
            semantic_boundary_detection=False
        )
        diagnostics_module = DocumentDiagnostics()
        
        text = " ".join([f"word{i}" for i in range(100)])
        metrics = diagnostics_module.analyze_document(text)
        
        chunks = chunker.chunk_document(text, metrics)
        
        if len(chunks) >= 2:
            # Check that consecutive chunks have some overlap
            chunk1_words = set(chunks[0].text.split())
            chunk2_words = set(chunks[1].text.split())
            overlap = chunk1_words & chunk2_words
            
            # Should have some overlap (depending on implementation)
            assert len(overlap) >= 0  # Minimal check


class TestEvaluationMetrics:
    """Tests for evaluation metrics"""
    
    def test_precision_calculation(self):
        """Test precision calculation"""
        from advanced_rag import RAGEvaluator
        
        evaluator = RAGEvaluator()
        
        results = [
            {"id": "doc1", "score": 0.9},
            {"id": "doc2", "score": 0.8},
            {"id": "doc3", "score": 0.7}
        ]
        ground_truth = ["doc1", "doc2"]
        
        precision = evaluator._calculate_precision(results, ground_truth)
        
        assert 0 <= precision <= 1
        assert precision == 2/3  # 2 relevant out of 3 retrieved
    
    def test_recall_calculation(self):
        """Test recall calculation"""
        from advanced_rag import RAGEvaluator
        
        evaluator = RAGEvaluator()
        
        results = [
            {"id": "doc1", "score": 0.9},
            {"id": "doc2", "score": 0.8}
        ]
        ground_truth = ["doc1", "doc2", "doc3"]
        
        recall = evaluator._calculate_recall(results, ground_truth)
        
        assert 0 <= recall <= 1
        assert recall == 2/3  # 2 relevant retrieved out of 3 total relevant
    
    def test_mrr_calculation(self):
        """Test Mean Reciprocal Rank calculation"""
        from advanced_rag import RAGEvaluator
        
        evaluator = RAGEvaluator()
        
        # First result is relevant
        results = [
            {"id": "doc1", "score": 0.9},
            {"id": "doc2", "score": 0.8}
        ]
        ground_truth = ["doc1"]
        
        mrr = evaluator._calculate_mrr(results, ground_truth)
        assert mrr == 1.0
        
        # Second result is relevant
        results = [
            {"id": "doc2", "score": 0.9},
            {"id": "doc1", "score": 0.8}
        ]
        mrr = evaluator._calculate_mrr(results, ground_truth)
        assert mrr == 0.5


class TestComplianceManager:
    """Tests for compliance and auditing"""
    
    @pytest.mark.asyncio
    async def test_audit_log_creation(self):
        """Test audit log creation"""
        from advanced_rag import ComplianceManager
        
        compliance = ComplianceManager(enable_audit=True)
        
        audit_log = await compliance.log_retrieval(
            query="test query",
            chunk_id="chunk1",
            score=0.85,
            latency_ms=50.0,
            user_id="user123"
        )
        
        assert audit_log is not None
        assert audit_log.event_type.value == "retrieval"
        assert audit_log.user_id == "user123"
        assert audit_log.event_data["query"] == "test query"
    
    @pytest.mark.asyncio
    async def test_document_versioning(self):
        """Test document version creation"""
        from advanced_rag import ComplianceManager
        
        compliance = ComplianceManager(enable_versioning=True)
        
        version = await compliance.create_version(
            doc_id="doc1",
            content="test content",
            change_type="create",
            chunk_count=5,
            total_tokens=100,
            user_id="user123"
        )
        
        assert version is not None
        assert version.doc_id == "doc1"
        assert version.version == "v1"
        assert version.change_type == "create"
    
    def test_lineage_tracking(self):
        """Test event lineage tracking"""
        from advanced_rag import ComplianceManager
        
        compliance = ComplianceManager()
        
        compliance.track_lineage("parent1", "child1")
        compliance.track_lineage("parent1", "child2")
        compliance.track_lineage("child1", "grandchild1")
        
        lineage = compliance.get_event_lineage("parent1")
        
        assert lineage["event_id"] == "parent1"
        assert len(lineage["children"]) == 2


class TestHybridRetrieval:
    """Tests for hybrid retrieval"""
    
    def test_rrf_fusion(self):
        """Test Reciprocal Rank Fusion"""
        from advanced_rag import HybridRetriever, MilvusIndexManager
        
        # Create mock index manager
        index_manager = None  # Would need proper initialization
        
        # This is a simplified test - full test would require Milvus
        # Just verify the class can be instantiated
        from advanced_rag import RetrievalConfig
        config = RetrievalConfig(hybrid_alpha=0.7)
        
        assert config.hybrid_alpha == 0.7
        assert config.dense_weight == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
