# Advanced RAG Pipeline with Milvus

A production-grade Retrieval-Augmented Generation (RAG) system implementing sophisticated information retrieval and semantic enrichment with quality guarantees and compliance tracking.

## 🎯 Overview

This implementation treats RAG as a **multi-stage probabilistic information system** rather than a simple embedding lookup. It provides:

- **Diagnostic-Informed Chunking**: Adaptive chunk granularity based on information entropy, redundancy, and domain density
- **Multi-Layered Indexing**: Semantic embeddings, sparse keyword indexes (BM25), and domain-specific ontological embeddings
- **Hybrid Retrieval**: Combined dense + sparse search with cross-encoder reranking
- **Quality Guarantees**: Hallucination risk quantification, faithfulness scoring, and retrieval drift detection
- **Compliance Framework**: Document lineage, versioning semantics, and audit-grade metadata
- **Performance SLAs**: Sub-80ms latency targets with hierarchical indexing and locality-aware sharding

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AdvancedRAGPipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Diagnostics  │→ │   Chunking   │→ │   Indexing   │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│         │                  │                  │                │
│         ↓                  ↓                  ↓                │
│  ┌──────────────────────────────────────────────────┐        │
│  │         Milvus Multi-Index Manager               │        │
│  ├──────────────┬──────────────┬──────────────┐    │        │
│  │  Semantic    │   Sparse     │   Domain     │    │        │
│  │    Index     │    Index     │    Index     │    │        │
│  │  (HNSW)      │  (BM25)      │  (HNSW)      │    │        │
│  └──────────────┴──────────────┴──────────────┘    │        │
│                         │                            │        │
│                         ↓                            │        │
│  ┌─────────────────────────────────────────────────┐│        │
│  │         Hybrid Retrieval Engine                 ││        │
│  │  • Vector Search + BM25                         ││        │
│  │  • Reciprocal Rank Fusion                       ││        │
│  │  • Cross-Encoder Reranking                      ││        │
│  └─────────────────────────────────────────────────┘│        │
│                         │                            │        │
│                         ↓                            │        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Evaluation  │  │ Drift Detect │  │  Compliance  │        │
│  │  & Quality   │  │   & Metrics  │  │  & Auditing  │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Start Milvus (using Docker)
docker-compose up -d

# Or use Milvus standalone
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker-compose up -d
```

### ⚡ Quick Start (Local)

```bash
# Bring up full stack (API + Milvus + Postgres + OTel-Collector)
docker-compose up -d

# Run DB migrations (uses Postgres in compose)
export DATABASE_URL=postgresql://raguser:ragpass@localhost:5432/ragdb
bash scripts/migrate.sh

# Open the minimal Chat UI
open http://localhost:8000/
# Health & metrics
curl -fsS http://localhost:8000/healthz
curl -fsS http://localhost:8000/metrics | head -n 20
```

Optional flags:
- `API_KEY` – require `x-api-key` on all endpoints
- `ENABLE_MMR=1` – enable diversification
- `ENABLE_ADAPTIVE_WEIGHTS=1` and `EXPERIMENT_ID=exp-001` – adaptive hybrid weighting (A/B ready)

## 💻 Usage

### Basic Example

```python
import asyncio
from advanced_rag import AdvancedRAGPipeline, PipelineConfig

async def main():
    # Initialize pipeline
    config = PipelineConfig(
        target_latency_ms=80.0,
        enable_reranking=True,
        enable_audit_logging=True
    )
    
    pipeline = AdvancedRAGPipeline(
        milvus_host="localhost",
        milvus_port=19530,
        config=config
    )
    
    # Ingest documents
    documents = [
        {
            "id": "doc1",
            "text": "Your document text here...",
            "metadata": {"source": "example"}
        }
    ]
    
    report = await pipeline.ingest_documents(documents)
    print(f"Indexed {report['chunks_created']} chunks")
    
    # Retrieve with quality evaluation
    results, metrics = await pipeline.retrieve(
        query="What is the main topic?",
        filters={"source": "example"}
    )
    
    print(f"Hallucination risk: {metrics.hallucination_risk:.3f}")
    print(f"Top result: {results[0].content[:100]}...")
    
    # Detect drift
    drift_report = await pipeline.detect_drift(
        sample_queries=["query1", "query2"]
    )
    
    if drift_report.drift_detected:
        print("Drift detected! Recommendations:")
        for rec in drift_report.recommendations:
            print(f"  - {rec}")
    
    await pipeline.close()

asyncio.run(main())
```

### Advanced Configuration

```python
config = PipelineConfig(
    # Performance targets
    target_latency_ms=80.0,
    enable_hierarchical_index=True,
    enable_sharding=True,
    
    # Retrieval configuration
    hybrid_alpha=0.7,  # 0=sparse only, 1=dense only
    top_k=20,
    rerank_top_k=5,
    enable_reranking=True,
    
    # Quality thresholds
    min_relevance_score=0.65,
    max_hallucination_risk=0.15,
    
    # Compliance
    enable_audit_logging=True,
    enable_versioning=True,
    retention_days=90
)
```

## 📊 Diagnostic Metrics

```python
from advanced_rag import DocumentDiagnostics

diagnostics = DocumentDiagnostics()
metrics = diagnostics.analyze_document(text)

print(f"Information Entropy: {metrics.information_entropy:.3f}")
print(f"Redundancy Score: {metrics.redundancy_score:.3f}")
print(f"Domain Density: {metrics.domain_density:.3f}")
print(f"Vocabulary Diversity: {metrics.vocabulary_diversity:.3f}")
```

## 🔍 Evaluation Metrics

```python
# Evaluation is automatic during retrieval
results, eval_metrics = await pipeline.retrieve(query)

print(f"Precision: {eval_metrics.retrieval_precision:.3f}")
print(f"Recall: {eval_metrics.retrieval_recall:.3f}")
print(f"MRR: {eval_metrics.mean_reciprocal_rank:.3f}")
print(f"NDCG: {eval_metrics.ndcg_at_k:.3f}")
print(f"Hallucination Risk: {eval_metrics.hallucination_risk:.3f}")
print(f"Faithfulness: {eval_metrics.faithfulness_score:.3f}")
```

## 📈 Performance Monitoring

```python
# Get performance telemetry
perf_report = pipeline.get_performance_report()

for stage, metrics in perf_report['stage_latencies'].items():
    print(f"{stage}:")
    print(f"  P50: {metrics['p50']:.2f}ms")
    print(f"  P95: {metrics['p95']:.2f}ms")
    print(f"  P99: {metrics['p99']:.2f}ms")

# Check SLA compliance
sla = perf_report['sla_compliance']
print(f"SLA Compliance: {sla['compliance_rate']:.1%}")
```

## 🔐 Compliance & Auditing

```python
# Query audit logs
from advanced_rag import AuditEventType

logs = pipeline.compliance.query_audit_logs(
    event_type=AuditEventType.RETRIEVAL,
    start_time="2024-01-01",
    end_time="2024-12-31"
)

# Generate compliance report
report = pipeline.compliance.generate_compliance_report(
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# Get document lineage
lineage = pipeline.compliance.get_document_lineage(doc_id="doc1")
for version in lineage:
    print(f"Version {version.version}: {version.created_at}")
```

## 🎛️ Integration with Embedding Models

```python
# Example: Integrate OpenAI embeddings
from openai import AsyncOpenAI

class EmbeddingGenerator:
    def __init__(self):
        self.client = AsyncOpenAI()
    
    async def encode_semantic(self, text: str):
        response = await self.client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return np.array(response.data[0].embedding)
    
    async def encode_sparse(self, text: str):
        # Use SPLADE or BM25 implementation
        pass
    
    async def encode_domain(self, text: str, domain: str):
        # Use domain-adapted model
        pass

# Set in index manager
pipeline.index_manager.embedding_generator = EmbeddingGenerator()
```

## 🔧 Customization

### Custom Chunking Strategy

```python
from advanced_rag import AdaptiveChunker

chunker = AdaptiveChunker(
    base_chunk_size=512,
    max_chunk_size=1024,
    min_chunk_size=128,
    overlap_ratio=0.15,
    semantic_boundary_detection=True
)

pipeline.chunker = chunker
```

### Custom Reranker

```python
from advanced_rag import CrossEncoderReranker

reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"
)

pipeline.retriever.reranker = reranker
```

## 📊 Performance Characteristics

- **Latency**: P95 < 80ms (with proper hardware)
- **Throughput**: 100+ QPS (with sharding)
- **Scalability**: Horizontal scaling via Milvus sharding
- **Accuracy**: Hallucination risk < 15%
- **SLA Compliance**: > 95% of queries meet latency targets

## 🔬 Research Features

This implementation includes several research-informed features:

1. **Diagnostic-Informed Chunking**: Adapts to document characteristics
2. **Hybrid Retrieval with RRF**: Combines dense and sparse effectively
3. **Hallucination Risk Quantification**: Multi-signal risk assessment
4. **Drift Detection**: Embedding space divergence monitoring
5. **Faithfulness Scoring**: NLI-based content verification

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please submit PRs for:
- New embedding integrations
- Additional evaluation metrics
- Performance optimizations
- Documentation improvements

## 📚 References

- Milvus Documentation: https://milvus.io/docs
- RAG Survey Paper: https://arxiv.org/abs/2312.10997
- Retrieval-Augmented Generation: https://arxiv.org/abs/2005.11401
- HNSW Algorithm: https://arxiv.org/abs/1603.09320

## 🆘 Support

For issues, questions, or feature requests:
- GitHub Issues: [Create an issue]
- Documentation: [Read the docs]
- Community: [Join discussions]

---

**Built with ❤️ for production RAG systems**

---

## 🛡️ Production-Grade Requirements (LL/HT/Scale/Reliability/Security/Cost)

This project is engineered for:

- Low Latency and High Throughput
  - HNSW indexes with tuned `ef` parameters and parallel searches; optional MMR diversification to avoid duplicate results.
  - Streaming responses (SSE) with token-by-token delivery; lazy Milvus connect to minimize startup latency.
  - Batch embedding, efficient tokenization, and minimal serialization overhead in hot paths.
  - Prometheus histogram for retrieve latency; P50/P95 monitoring.

- Scalability
  - Milvus sharding/partitioning per domain/version; collection aliasing for blue/green index swaps.
  - ECS Fargate deployment (Terraform) with ALB; containerized etcd/MinIO/Milvus/API to scale horizontally.
  - Optional EKS Milvus Cluster (recommended for sustained prod) with S3 storage (see DEPLOYMENT.md notes).

- Reliability
  - Idempotent ingestion (content-hash versioning), schema-index validation on start, safe shutdown.
  - CI/CD with linting, tests, coverage gate (≥95%), SAST, Trivy FS & image scans, optional ZAP baseline.
  - Health checks (`/healthz`), smoke tests in CI, and CloudWatch logs on AWS.
  - Local parity: `docker-compose.yml` includes `postgres` and `otel-collector`; API uses `DATABASE_URL` and `OTEL_EXPORTER_OTLP_ENDPOINT`.
  - Import `observability/dashboards/rag_overview.json` into Grafana for latency and request dashboards.

- Security
  - Optional API key enforced via `x-api-key` header; secrets externally configured.
  - Compliance logging with immutable hashes; retention enforcement; document lineage/versioning.
  - Ready for WAF/Cognito integration on ALB; private subnets and Security Groups in Terraform.

- Cost Optimization
  - Choose index params per workload; enable MMR only when needed.
  - Auto disable reranking for short or high-confidence queries (configurable).
  - Container sizes and Fargate sizing fit small-to-medium workloads; easy scale-up.
  - Logs retention minimized; quantization/IVF/PQ can be enabled on cold tiers.

- Efficient Data Querying
  - Hybrid dense + sparse + domain; configurable weighting; optional learned ranker (hook points present).
  - RRF fusion (k=60) + optional MMR; precise filter expressions with escaping to avoid injection.
  - Domain partitions and index aliases for fast targetted retrieval.

---

## 🧪 Quality Gate and Test Coverage

- Unit tests cover critical logic across diagnostics, chunking, retrieval fusion, MMR, evaluation, compliance, and pipeline.
- CI enforces coverage ≥95% via:
  - `python -m coverage run -m pytest`
  - `python -m coverage report -m --fail-under=95`
- To run locally:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
python -m coverage run -m pytest
python -m coverage report -m
```

---

## 🔧 API Endpoints (FastAPI)

- Health/Observability:
  - `GET /healthz` – service status
  - `GET /metrics` – Prometheus metrics
- Chat:
  - `POST /chat/start` → `{session_id}`
  - `GET /chat/history?session_id=...`
  - `POST /chat/clear` with `{session_id}`
  - `GET /chat/stream?session_id=...&q=...` – SSE streaming
  - `POST /chat` with `{session_id, query, ...}` – non-stream
- Feedback:
  - `POST /feedback` with `{message_id, vote: "up"|"down", comment?}`
- ETL/Evals:
  - `POST /etl/run` with `{"source":"fs","config":{"path":"./docs","domain":"technical"}}`
  - `POST /eval/run` with `{"items":[{"query":"...","relevant_doc_ids":["doc1", ...]}]}` – returns aggregated metrics

Auth (optional): set `API_KEY`, then include header `x-api-key: <key>` on all endpoints.

---

## 🖥️ Minimal Chat UI

- Served by API: open `http://localhost:8000/` (local) or `http://<alb-dns>/` (AWS).
- Features:
  - Streaming tokens, conversation history, “New Chat” and “Clear Chat” controls
  - 4 context-aware suggestions; clicking submits a new query
  - Inline citations grouped under the message

---

## ☁️ Cloud Deployment (One Command)

Terraform stack (ECS Fargate + ALB + RDS + ECR):

```bash
bash scripts/deploy_aws.sh us-east-1 v1
# ALB URL printed; UI at http://<alb-dns>/
```

Prereqs:
- AWS credentials configured, Terraform, Docker
- Optional: set `API_KEY` for protected access

Post‑deploy tasks:
```bash
# Run DB migrations against RDS (replace endpoint/password)
export DATABASE_URL="postgresql://raguser:<password>@<rds-endpoint>:5432/ragdb"
alembic upgrade head

# Health & metrics
curl -fsS http://<alb-dns>/healthz
curl -fsS http://<alb-dns>/metrics | head -n 20
```

Observability:
- ECS task ships with ADOT collector (foundation). Configure exporters for X‑Ray (traces) and AMP (metrics) per your environment.
- CloudWatch log groups included. Import `observability/dashboards/rag_overview.json` into Grafana.

---

## ⚙️ Performance Knobs

- RetrievalConfig:
  - `top_k`, `dense_weight`, `sparse_weight`
  - `enable_reranking` (cross-encoder)
  - `enable_mmr`, `mmr_lambda` for diversification
  - `weight_adapter(query) -> (dense_weight, sparse_weight)` hook for adaptive hybrid weighting (A/B friendly)
  - HNSW search params (`ef`) per index
- PipelineConfig:
  - `target_latency_ms`, `hybrid_alpha`, `rerank_top_k`
  - `enable_sharding`, quality thresholds

---

## 🔐 Security Checklist

- API keys or OAuth tokens for API access (header-based).
- Secrets via AWS Secrets Manager / environment variables.
- Input sanitization and filter escaping; content-based version hashing and lineage.
- CI SAST/DAST + Trivy FS and image scans; SBOM and image signing recommended.

---

## 💵 Cost Optimization Guidelines

- Right-size Fargate CPU/RAM; scale ECS tasks with target-tracking ALB requests.
- Use Milvus PQ/IVF for cold data; keep hot shards in memory; adjust `ef`.
- Reduce reranking and `top_k` for easy/high-confidence queries.
- Logging retention policy and exporter sampling for traces; compress embeddings when feasible.

