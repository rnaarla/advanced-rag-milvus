"""
Advanced RAG Pipeline with Milvus
Production-grade retrieval system with compliance and quality guarantees
"""

from .pipeline import (
    AdvancedRAGPipeline,
    PipelineConfig,
    PipelineStage,
    RetrievalResult
)

from .diagnostics import (
    DocumentDiagnostics,
    DiagnosticMetrics
)

from .chunking import (
    AdaptiveChunker,
    ChunkMetadata,
    Chunk
)

from .indexing import (
    MilvusIndexManager,
    IndexType,
    IndexConfig
)

from .retrieval import (
    HybridRetriever,
    RetrievalConfig,
    CrossEncoderReranker
)

from .ranker import (
    LearnedRanker,
    LearnedRankerConfig,
)

from .semantic_enrichment import (
    SemanticEnricher,
    EnrichmentResult,
)

from .decomposition import (
    QueryDecomposer,
    DecompositionResult,
)

from .experiments import (
    ExperimentManager,
)

from .evaluation import (
    RAGEvaluator,
    EvaluationMetrics,
    DriftReport
)

from .compliance import (
    ComplianceManager,
    AuditLog,
    DocumentVersion,
    AuditEventType
)

__version__ = "1.0.0"

__all__ = [
    # Main pipeline
    "AdvancedRAGPipeline",
    "PipelineConfig",
    "PipelineStage",
    "RetrievalResult",
    
    # Diagnostics
    "DocumentDiagnostics",
    "DiagnosticMetrics",
    
    # Chunking
    "AdaptiveChunker",
    "ChunkMetadata",
    "Chunk",
    
    # Indexing
    "MilvusIndexManager",
    "IndexType",
    "IndexConfig",
    
    # Retrieval
    "HybridRetriever",
    "RetrievalConfig",
    "CrossEncoderReranker",
    "LearnedRanker",
    "LearnedRankerConfig",
    "SemanticEnricher",
    "EnrichmentResult",
    "QueryDecomposer",
    "DecompositionResult",
    "ExperimentManager",
    
    # Evaluation
    "RAGEvaluator",
    "EvaluationMetrics",
    "DriftReport",
    
    # Compliance
    "ComplianceManager",
    "AuditLog",
    "DocumentVersion",
    "AuditEventType",
]
