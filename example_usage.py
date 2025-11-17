"""
Example Usage: Advanced RAG Pipeline with Milvus

This example demonstrates:
1. Document ingestion with diagnostics
2. Adaptive chunking
3. Multi-layered indexing
4. Hybrid retrieval
5. Evaluation and quality metrics
6. Drift detection
7. Compliance and auditing
"""

import asyncio
from advanced_rag import (
    AdvancedRAGPipeline,
    PipelineConfig
)


async def main():
    print("=" * 80)
    print("Advanced RAG Pipeline Example")
    print("=" * 80)
    
    # Initialize pipeline with configuration
    config = PipelineConfig(
        target_latency_ms=80.0,
        enable_hierarchical_index=True,
        enable_sharding=True,
        hybrid_alpha=0.7,
        top_k=20,
        rerank_top_k=5,
        enable_reranking=True,
        min_relevance_score=0.65,
        max_hallucination_risk=0.15,
        enable_audit_logging=True,
        enable_versioning=True,
        retention_days=90
    )
    
    pipeline = AdvancedRAGPipeline(
        milvus_host="localhost",
        milvus_port=19530,
        config=config
    )
    
    print("\n✓ Pipeline initialized with configuration:")
    print(f"  - Target latency: {config.target_latency_ms}ms")
    print(f"  - Hybrid retrieval alpha: {config.hybrid_alpha}")
    print(f"  - Reranking: {'enabled' if config.enable_reranking else 'disabled'}")
    print(f"  - Audit logging: {'enabled' if config.enable_audit_logging else 'disabled'}")
    
    # =========================================================================
    # STAGE 1: Document Ingestion
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 1: Document Ingestion with Diagnostics")
    print("=" * 80)
    
    # Sample documents
    documents = [
        {
            "id": "doc_001",
            "text": """
            Machine learning is a subset of artificial intelligence that focuses on 
            developing algorithms that can learn from and make predictions based on data. 
            Deep learning, a specialized form of machine learning, uses neural networks 
            with multiple layers to process complex patterns. Transformers, introduced 
            in the "Attention is All You Need" paper, revolutionized natural language 
            processing by enabling parallel processing of sequential data. Modern large 
            language models like GPT and BERT are built on transformer architectures 
            and have achieved remarkable performance on various NLP tasks.
            """,
            "metadata": {
                "source": "ml_textbook",
                "domain": "technical",
                "author": "Dr. Smith"
            }
        },
        {
            "id": "doc_002",
            "text": """
            Vector databases are specialized database systems optimized for storing 
            and querying high-dimensional vector embeddings. Unlike traditional 
            databases that use exact matching, vector databases use approximate 
            nearest neighbor (ANN) algorithms to find semantically similar items. 
            Milvus is an open-source vector database that supports multiple index 
            types including HNSW, IVF, and DISKANN. It provides features like 
            dynamic schema, hybrid search, and time travel queries. Vector databases 
            are essential infrastructure for retrieval-augmented generation systems.
            """,
            "metadata": {
                "source": "database_guide",
                "domain": "technical",
                "author": "Engineering Team"
            }
        },
        {
            "id": "doc_003",
            "text": """
            Retrieval-augmented generation combines the strengths of retrieval systems 
            and generative models. The retrieval component finds relevant context from 
            a knowledge base, while the generation component produces responses grounded 
            in that context. This approach reduces hallucinations and enables models to 
            access up-to-date information without retraining. Key challenges include 
            chunking strategies, embedding quality, and retrieval precision. Advanced 
            RAG systems employ hybrid retrieval, reranking, and quality evaluation 
            to ensure reliable outputs.
            """,
            "metadata": {
                "source": "rag_whitepaper",
                "domain": "technical",
                "author": "Research Lab"
            }
        }
    ]
    
    # Ingest documents
    print("\nIngesting documents...")
    ingestion_report = await pipeline.ingest_documents(
        documents=documents,
        domain="technical"
    )
    
    print(f"\n✓ Ingestion completed in {ingestion_report['total_time_ms']:.2f}ms")
    print(f"  - Documents processed: {ingestion_report['total_documents']}")
    print(f"  - Chunks created: {ingestion_report['chunks_created']}")
    
    # Show diagnostic metrics
    print("\nDocument Diagnostics:")
    for metric in ingestion_report['diagnostic_metrics']:
        print(f"\n  Document {metric['document_id']}:")
        print(f"    - Information Entropy: {metric['entropy']:.3f}")
        print(f"    - Redundancy Score: {metric['redundancy']:.3f}")
        print(f"    - Domain Density: {metric['domain_density']:.3f}")
    
    # =========================================================================
    # STAGE 2: Hybrid Retrieval
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 2: Hybrid Retrieval with Reranking")
    print("=" * 80)
    
    queries = [
        "What is a transformer architecture in deep learning?",
        "Explain how vector databases work",
        "What are the benefits of retrieval-augmented generation?"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 80)
        
        # Perform retrieval
        results, eval_metrics = await pipeline.retrieve(
            query=query,
            filters={"domain": "technical"}
        )
        
        print(f"\n✓ Retrieved {len(results)} results")
        print(f"  - Latency: {results[0].latency_ms:.2f}ms" if results else "  - No results")
        
        # Show top results
        print("\nTop Results:")
        for i, result in enumerate(results[:3], 1):
            print(f"\n  {i}. Score: {result.score:.4f} | Method: {result.retrieval_method}")
            content_preview = result.content[:150].replace('\n', ' ').strip()
            print(f"     Content: {content_preview}...")
            print(f"     Metadata: doc_id={result.metadata['doc_id']}, "
                  f"entropy={result.metadata.get('entropy', 0):.3f}")
        
        # Show evaluation metrics
        print("\nEvaluation Metrics:")
        print(f"  - Hallucination Risk: {eval_metrics.hallucination_risk:.3f}")
        print(f"  - Faithfulness Score: {eval_metrics.faithfulness_score:.3f}")
        print(f"  - Coverage Score: {eval_metrics.coverage_score:.3f}")
        print(f"  - Diversity Score: {eval_metrics.diversity_score:.3f}")
        print(f"  - Confidence: {eval_metrics.confidence_score:.3f}")
        
        # Check quality thresholds
        if eval_metrics.hallucination_risk > config.max_hallucination_risk:
            print("\n  ⚠️  WARNING: High hallucination risk detected!")
        else:
            print("\n  ✓ Hallucination risk within acceptable threshold")
    
    # =========================================================================
    # STAGE 3: Drift Detection
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 3: Retrieval Drift Detection")
    print("=" * 80)
    
    # Run drift detection
    drift_report = await pipeline.detect_drift(sample_queries=queries)
    
    print(f"\n✓ Drift detection completed")
    print(f"  - Drift detected: {'YES' if drift_report.drift_detected else 'NO'}")
    print(f"  - Drift magnitude: {drift_report.drift_magnitude:.4f}")
    print(f"  - Embedding divergence: {drift_report.embedding_divergence:.4f}")
    print(f"  - Distribution shift: {drift_report.distribution_shift:.4f}")
    
    if drift_report.affected_queries:
        print(f"\n  Affected queries: {len(drift_report.affected_queries)}")
    
    if drift_report.recommendations:
        print("\n  Recommendations:")
        for rec in drift_report.recommendations:
            print(f"    - {rec}")
    
    # =========================================================================
    # STAGE 4: Performance Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 4: Performance Telemetry")
    print("=" * 80)
    
    perf_report = pipeline.get_performance_report()
    
    print("\nStage Latencies:")
    for stage, metrics in perf_report['stage_latencies'].items():
        print(f"\n  {stage}:")
        print(f"    - P50: {metrics['p50']:.2f}ms")
        print(f"    - P95: {metrics['p95']:.2f}ms")
        print(f"    - P99: {metrics['p99']:.2f}ms")
        print(f"    - Mean: {metrics['mean']:.2f}ms ± {metrics['std']:.2f}ms")
    
    if 'sla_compliance' in perf_report:
        sla = perf_report['sla_compliance']
        print(f"\nSLA Compliance:")
        print(f"  - Target: {sla['target_ms']:.2f}ms")
        print(f"  - Compliance rate: {sla['compliance_rate']:.1%}")
        print(f"  - P95 latency: {sla['p95_latency']:.2f}ms")
        
        if sla['compliance_rate'] >= 0.95:
            print("  ✓ Meeting SLA targets")
        else:
            print("  ⚠️  Below SLA targets - optimization needed")
    
    # =========================================================================
    # STAGE 5: Compliance & Auditing
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 5: Compliance & Audit Trail")
    print("=" * 80)
    
    # Query audit logs
    from datetime import datetime
    current_time = datetime.now().isoformat()
    
    compliance_report = pipeline.compliance.generate_compliance_report(
        start_date="2024-01-01",
        end_date=current_time
    )
    
    print(f"\n✓ Compliance report generated")
    print(f"  - Total events: {compliance_report['total_events']}")
    print(f"  - Event breakdown:")
    for event_type, count in compliance_report['event_counts'].items():
        print(f"    - {event_type}: {count}")
    
    # Show document lineage
    if documents:
        doc_id = documents[0]['id']
        lineage = pipeline.compliance.get_document_lineage(doc_id)
        print(f"\n  Document lineage for {doc_id}:")
        for version in lineage:
            print(f"    - {version.version} ({version.created_at})")
            print(f"      Hash: {version.version_hash[:16]}...")
            print(f"      Classification: {version.classification}")
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    print("\n" + "=" * 80)
    print("Cleanup")
    print("=" * 80)
    
    await pipeline.close()
    print("\n✓ Pipeline closed successfully")
    
    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
