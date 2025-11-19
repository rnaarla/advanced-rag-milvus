"""
Advanced RAG Pipeline with Milvus
Multi-stage information retrieval and semantic enrichment system
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from enum import Enum

from .diagnostics import DocumentDiagnostics, DiagnosticMetrics
from .chunking import AdaptiveChunker, ChunkMetadata
from .indexing import MilvusIndexManager, IndexType
from .retrieval import HybridRetriever, RetrievalConfig
from .ranker import LearnedRanker
from .evaluation import RAGEvaluator, EvaluationMetrics
from .compliance import ComplianceManager, AuditLog
from .semantic_enrichment import SemanticEnricher
from .decomposition import QueryDecomposer, DecompositionResult
from .query_rewriting import QueryRewriter
from .constants import APIConstants


class PipelineStage(Enum):
    """Pipeline execution stages for telemetry"""
    DIAGNOSTICS = "diagnostics"
    CHUNKING = "chunking"
    INDEXING = "indexing"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    EVALUATION = "evaluation"


@dataclass
class PipelineConfig:
    """Configuration for the RAG pipeline"""
    # Performance targets
    target_latency_ms: float = 80.0
    enable_hierarchical_index: bool = True
    enable_sharding: bool = True
    
    # Retrieval configuration
    hybrid_alpha: float = 0.7  # Weight for vector search vs BM25
    top_k: int = 20
    rerank_top_k: int = 5
    enable_reranking: bool = True
    
    # Quality thresholds
    min_relevance_score: float = 0.65
    max_hallucination_risk: float = 0.15
    
    # Compliance
    enable_audit_logging: bool = True
    enable_versioning: bool = True
    retention_days: int = 90


@dataclass
class RetrievalResult:
    """Structured retrieval result with metadata"""
    content: str
    chunk_id: str
    score: float
    metadata: ChunkMetadata
    retrieval_method: str
    latency_ms: float
    audit_trail: Optional[AuditLog] = None


class AdvancedRAGPipeline:
    """
    Production-grade RAG pipeline implementing multi-stage retrieval
    with quality guarantees and compliance tracking
    """
    
    def __init__(
        self,
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        config: Optional[PipelineConfig] = None,
        connect_to_milvus: bool = True,
        query_rewriter: Optional[QueryRewriter] = None,
    ):
        self.config = config or PipelineConfig()
        
        # Initialize core components
        self.diagnostics = DocumentDiagnostics()
        self.chunker = AdaptiveChunker()
        self.enricher = SemanticEnricher()
        self.decomposer = QueryDecomposer()
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.index_manager = MilvusIndexManager(
            host=milvus_host,
            port=milvus_port,
            enable_sharding=self.config.enable_sharding,
            connect=connect_to_milvus
        )
        self.retriever = HybridRetriever(
            index_manager=self.index_manager,
            config=RetrievalConfig(
                hybrid_alpha=self.config.hybrid_alpha,
                top_k=self.config.top_k,
                enable_reranking=self.config.enable_reranking,
            ),
            learned_ranker=LearnedRanker(),
        )
        self.evaluator = RAGEvaluator()
        self.compliance = ComplianceManager(
            enable_audit=self.config.enable_audit_logging,
            enable_versioning=self.config.enable_versioning
        )
        
        # Performance tracking
        self.stage_latencies: Dict[PipelineStage, List[float]] = {
            stage: [] for stage in PipelineStage
        }
        
    async def ingest_documents(
        self,
        documents: List[Dict[str, Any]],
        domain: Optional[str] = None
    ) -> Dict[str, Any]:  # pragma: no cover
        """
        Ingest documents through diagnostic-informed chunking and indexing
        
        Args:
            documents: List of documents with 'text' and optional 'metadata'
            domain: Optional domain identifier for specialized embeddings
            
        Returns:
            Ingestion report with statistics and diagnostics
        """
        start_time = datetime.now()
        ingestion_report = {
            "total_documents": len(documents),
            "chunks_created": 0,
            "diagnostic_metrics": [],
            "indexing_summary": {}
        }
        
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            # Stage 1: Document Diagnostics
            stage_start = datetime.now()
            metrics = self.diagnostics.analyze_document(doc["text"])
            self._record_latency(
                PipelineStage.DIAGNOSTICS,
                (datetime.now() - stage_start).total_seconds() * 1000
            )
            
            ingestion_report["diagnostic_metrics"].append({
                "document_id": doc.get("id", doc_idx),
                "entropy": metrics.information_entropy,
                "redundancy": metrics.redundancy_score,
                "domain_density": metrics.domain_density
            })

            # Data quality assessment per document
            dq = self._assess_data_quality(doc, metrics)
            ingestion_report.setdefault("data_quality", []).append(dq)
            
            # Stage 2: Adaptive Chunking
            stage_start = datetime.now()
            chunks = self.chunker.chunk_document(
                text=doc["text"],
                diagnostics=metrics,
                metadata={
                    "doc_id": doc.get("id", doc_idx),
                    "source": doc.get("metadata", {}).get("source", "unknown"),
                    "timestamp": datetime.now().isoformat(),
                    **doc.get("metadata", {})
                }
            )
            self._record_latency(
                PipelineStage.CHUNKING,
                (datetime.now() - stage_start).total_seconds() * 1000
            )
            
            # Per-chunk semantic enrichment (entities, topics)
            for chunk in chunks:
                enrichment = self.enricher.enrich(chunk.text)
                chunk.metadata.extra.setdefault("entities", enrichment.entities)
                chunk.metadata.extra.setdefault("topics", enrichment.topics)

            all_chunks.extend(chunks)
            ingestion_report["chunks_created"] += len(chunks)
        
        # Stage 3: Multi-layered Indexing
        stage_start = datetime.now()
        indexing_results = await self.index_manager.index_chunks(
            chunks=all_chunks,
            domain=domain
        )
        self._record_latency(
            PipelineStage.INDEXING,
            (datetime.now() - stage_start).total_seconds() * 1000
        )
        
        ingestion_report["indexing_summary"] = indexing_results
        ingestion_report["total_time_ms"] = (
            datetime.now() - start_time
        ).total_seconds() * 1000
        
        # Log compliance data
        if self.config.enable_audit_logging:
            await self.compliance.log_ingestion(
                document_count=len(documents),
                chunk_count=len(all_chunks),
                report=ingestion_report
            )
        
        return ingestion_report
    
    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[RetrievalResult], EvaluationMetrics]:  # pragma: no cover
        """
        Perform hybrid retrieval with quality evaluation
        
        Args:
            query: Search query
            filters: Optional metadata filters
            context: Optional context for evaluation (ground truth, etc.)
            
        Returns:
            Tuple of (retrieval results, evaluation metrics)
        """
        pipeline_start = datetime.now()

        # Apply query rewriting (no-op by default) before retrieval.
        rewritten_query = self.query_rewriter.rewrite(query, context or {})

        # Stage 1: Hybrid Retrieval
        stage_start = datetime.now()
        raw_results = await self.retriever.retrieve(
            query=rewritten_query,
            filters=filters,
            # Allow callers to steer retrieval profile via context hint.
            profile_hint=(context or {}).get("retrieval_profile") if context else None,
        )
        retrieval_latency = (datetime.now() - stage_start).total_seconds() * 1000
        self._record_latency(PipelineStage.RETRIEVAL, retrieval_latency)
        
        # Stage 2: Reranking (if enabled)
        if self.config.enable_reranking:
            stage_start = datetime.now()
            reranked_results = await self.retriever.rerank(
                query=rewritten_query,
                results=raw_results,
                top_k=self.config.rerank_top_k
            )
            self._record_latency(
                PipelineStage.RERANKING,
                (datetime.now() - stage_start).total_seconds() * 1000
            )
        else:
            reranked_results = raw_results[:self.config.rerank_top_k]
        
        # Stage 3: Evaluation
        stage_start = datetime.now()
        eval_metrics = await self.evaluator.evaluate_retrieval(
            query=query,
            results=reranked_results,
            context=context
        )
        self._record_latency(
            PipelineStage.EVALUATION,
            (datetime.now() - stage_start).total_seconds() * 1000
        )
        
        # Check quality thresholds
        if eval_metrics.hallucination_risk > self.config.max_hallucination_risk:
            print(f"WARNING: High hallucination risk detected: {eval_metrics.hallucination_risk:.3f}")
        
        # Format results with audit trail
        total_latency = (datetime.now() - pipeline_start).total_seconds() * 1000
        formatted_results = []
        
        for result in reranked_results:
            audit_log = None
            if self.config.enable_audit_logging:
                audit_log = await self.compliance.log_retrieval(
                    query=query,
                    chunk_id=result["id"],
                    score=result["score"],
                    latency_ms=total_latency
                )
            
            formatted_results.append(RetrievalResult(
                content=result["content"],
                chunk_id=result["id"],
                score=result["score"],
                metadata=result["metadata"],
                retrieval_method=result.get("method", "hybrid"),
                latency_ms=total_latency,
                audit_trail=audit_log
            ))
        
        # Check SLA compliance
        if total_latency > self.config.target_latency_ms:
            print(f"WARNING: SLA violation - latency {total_latency:.2f}ms exceeds target {self.config.target_latency_ms}ms")
        
        return formatted_results, eval_metrics
    
    async def plan_and_execute(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Plan-and-execute flow for complex queries.
        
        - Decompose the query into sub-queries.
        - Run the standard retrieve() flow for each sub-query.
        - Return aggregated results and basic metadata.
        """
        plan: DecompositionResult = self.decomposer.decompose(query)
        subquery_results: List[Dict[str, Any]] = []
        
        for sub_query in plan.sub_queries:
            # Reuse existing retrieve() to keep behaviour consistent.
            results, metrics = await self.retrieve(
                query=sub_query,
                filters=filters,
                context=context,
            )
            subquery_results.append(
                {
                    "query": sub_query,
                    "results": results,
                    "metrics": metrics,
                }
            )
        
        return {
            "decomposition": {
                "sub_queries": plan.sub_queries,
                "strategy": plan.strategy,
            },
            "subqueries": subquery_results,
        }
    
    async def detect_drift(self, sample_queries: List[str]) -> Dict[str, Any]:
        """
        Detect retrieval drift using embedding space divergence
        
        Args:
            sample_queries: Representative query sample for drift analysis
            
        Returns:
            Drift detection report
        """
        return await self.evaluator.detect_drift(
            queries=sample_queries,
            index_manager=self.index_manager
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance telemetry report"""
        report = {
            "stage_latencies": {},
            "sla_compliance": {},
            "throughput_estimate": {}
        }
        
        for stage, latencies in self.stage_latencies.items():
            if latencies:
                report["stage_latencies"][stage.value] = {
                    "p50": np.percentile(latencies, 50),
                    "p95": np.percentile(latencies, 95),
                    "p99": np.percentile(latencies, 99),
                    "mean": np.mean(latencies),
                    "std": np.std(latencies)
                }
        
        # Calculate end-to-end SLA compliance
        if self.stage_latencies[PipelineStage.RETRIEVAL]:
            total_latencies = [
                sum([
                    self.stage_latencies[stage][i]
                    for stage in [PipelineStage.RETRIEVAL, PipelineStage.RERANKING, PipelineStage.EVALUATION]
                    if len(self.stage_latencies[stage]) > i
                ])
                for i in range(len(self.stage_latencies[PipelineStage.RETRIEVAL]))
            ]
            
            sla_compliant = sum(
                1 for lat in total_latencies
                if lat <= self.config.target_latency_ms
            )
            report["sla_compliance"] = {
                "target_ms": self.config.target_latency_ms,
                "compliance_rate": sla_compliant / len(total_latencies) if total_latencies else 0,
                "p95_latency": np.percentile(total_latencies, 95) if total_latencies else 0
            }
        
        return report
    
    def _record_latency(self, stage: PipelineStage, latency_ms: float):
        """Record stage latency for telemetry"""
        self.stage_latencies[stage].append(latency_ms)
        
        # Keep rolling window of last 1000 measurements
        if len(self.stage_latencies[stage]) > 1000:
            self.stage_latencies[stage] = self.stage_latencies[stage][-1000:]

    def _assess_data_quality(
        self,
        doc: Dict[str, Any],
        metrics: DiagnosticMetrics,
    ) -> Dict[str, Any]:
        """
        Lightweight data quality checks for ingestion.

        Flags obvious issues only (empty text, extreme redundancy, etc.) and
        records them in the ingestion report for operators to inspect.
        """
        flags = []
        text = (doc.get("text") or "").strip()
        if not text:
            flags.append("empty_text")

        if len(text) > APIConstants.MAX_DOCUMENT_TEXT_LENGTH:
            flags.append("text_too_long")

        if metrics.redundancy_score > 0.95:
            flags.append("high_redundancy")

        if metrics.information_entropy < 0.05:
            flags.append("very_low_entropy")

        return {
            "document_id": doc.get("id"),
            "flags": flags,
        }

    async def close(self):
        """Cleanup resources"""
        await self.index_manager.close()
        if self.config.enable_audit_logging:
            await self.compliance.close()
