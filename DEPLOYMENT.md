# Deployment & Operations Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Deployment](#deployment)
5. [Operations](#operations)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)
8. [Performance Tuning](#performance-tuning)

## Prerequisites

### Hardware Requirements
- **Minimum**:
  - 8 GB RAM
  - 4 CPU cores
  - 50 GB storage
  - GPU optional (for embeddings/reranking)

- **Recommended (Production)**:
  - 32 GB RAM
  - 16 CPU cores
  - 500 GB SSD storage
  - GPU with 8GB+ VRAM

### Software Requirements
- Python 3.8+
- Docker & Docker Compose
- Milvus 2.3+

## Installation

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/advanced-rag-milvus.git
cd advanced-rag-milvus

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Or install with extras
pip install -e ".[embeddings,reranking,dev]"
```

### 2. Start Milvus

```bash
# Using Docker Compose
docker-compose up -d

# Verify Milvus is running
docker ps | grep milvus

# Check health
curl http://localhost:9091/healthz
```

### 3. Verify Installation

```bash
# Run tests
pytest test_advanced_rag.py -v

# Run example
python example_usage.py
```

## Configuration

### 1. Create Configuration File

```bash
cp config.template.yaml config.yaml
```

### 2. Set Environment Variables

```bash
# Create .env file
cat > .env << EOF
OPENAI_API_KEY=your_openai_key_here
COHERE_API_KEY=your_cohere_key_here
MILVUS_HOST=localhost
MILVUS_PORT=19530
EOF
```

### 3. Customize Configuration

Edit `config.yaml` to match your requirements:

```yaml
pipeline:
  target_latency_ms: 80.0  # Adjust based on hardware
  hybrid_alpha: 0.7        # Tune dense vs sparse ratio
  top_k: 20                # Initial retrieval size
  rerank_top_k: 5          # Final results count
```

## Deployment

### Development Deployment

```bash
# Start services
docker-compose up -d

# Run pipeline
python example_usage.py
```

### Production Deployment

#### Using Docker

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY advanced_rag/ ./advanced_rag/
COPY example_usage.py .
COPY config.yaml .

CMD ["python", "example_usage.py"]
```

```bash
# Build and run
docker build -t advanced-rag:latest .
docker run -d \
  --name rag-pipeline \
  --network milvus \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  advanced-rag:latest
```

#### Using Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-pipeline
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-pipeline
  template:
    metadata:
      labels:
        app: rag-pipeline
    spec:
      containers:
      - name: rag-pipeline
        image: advanced-rag:latest
        env:
        - name: MILVUS_HOST
          value: "milvus-service"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
```

## Operations

### Daily Operations

#### 1. Health Checks

```bash
# Check Milvus status
curl http://localhost:9091/healthz

# Check collection stats
python -c "
from advanced_rag import MilvusIndexManager
manager = MilvusIndexManager()
stats = manager.get_collection_stats('semantic_index')
print(stats)
"
```

#### 2. Monitoring Queries

```python
# Monitor performance
from advanced_rag import AdvancedRAGPipeline

pipeline = AdvancedRAGPipeline()
report = pipeline.get_performance_report()

print(f"P95 latency: {report['stage_latencies']['retrieval']['p95']:.2f}ms")
print(f"SLA compliance: {report['sla_compliance']['compliance_rate']:.1%}")
```

#### 3. Audit Log Review

```python
# Query audit logs
from advanced_rag import ComplianceManager, AuditEventType

compliance = ComplianceManager()
logs = compliance.query_audit_logs(
    event_type=AuditEventType.RETRIEVAL,
    start_time="2024-01-01"
)

print(f"Total retrievals: {len(logs)}")
```

### Weekly Operations

#### 1. Drift Detection

```python
# Check for retrieval drift
drift_report = await pipeline.detect_drift(sample_queries=[...])

if drift_report.drift_detected:
    print("WARNING: Drift detected!")
    for rec in drift_report.recommendations:
        print(f"  - {rec}")
```

#### 2. Quality Assessment

```python
# Evaluate quality metrics
results, metrics = await pipeline.retrieve(query="test")

if metrics.hallucination_risk > 0.2:
    print("WARNING: High hallucination risk")

if metrics.faithfulness_score < 0.7:
    print("WARNING: Low faithfulness score")
```

#### 3. Index Optimization

```bash
# Compact Milvus collections
python -c "
from pymilvus import Collection
collection = Collection('semantic_index')
collection.compact()
"
```

### Monthly Operations

#### 1. Compliance Reporting

```python
# Generate monthly compliance report
report = compliance.generate_compliance_report(
    start_date="2024-01-01",
    end_date="2024-01-31"
)

print(f"Total events: {report['total_events']}")
print(f"Version updates: {report['version_updates']}")
```

#### 2. Performance Analysis

```bash
# Analyze performance trends
python scripts/analyze_performance.py --period monthly
```

#### 3. Data Retention

```python
# Enforce retention policies
compliance._enforce_retention_policy()
```

## Monitoring

### Metrics to Track

1. **Latency Metrics**
   - P50, P95, P99 latency
   - Per-stage breakdown
   - SLA compliance rate

2. **Quality Metrics**
   - Hallucination risk
   - Faithfulness score
   - Retrieval precision/recall
   - Drift magnitude

3. **System Metrics**
   - CPU/Memory usage
   - Disk I/O
   - Network throughput
   - Error rates

### Prometheus Integration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'rag-pipeline'
    static_configs:
      - targets: ['localhost:9090']
```

### Grafana Dashboards

Import the provided dashboard:
- Latency heatmaps
- Quality score trends
- Drift detection alerts
- System resource usage

## Troubleshooting

### Common Issues

#### 1. High Latency

**Symptoms**: P95 > 100ms

**Solutions**:
- Increase HNSW `ef` parameter for accuracy vs speed trade-off
- Enable sharding if not already enabled
- Add more Milvus replicas
- Optimize chunk sizes

```python
# Adjust search parameters
search_params = {
    "metric_type": "COSINE",
    "params": {"ef": 32}  # Reduce from 64
}
```

#### 2. High Hallucination Risk

**Symptoms**: hallucination_risk > 0.2

**Solutions**:
- Increase rerank_top_k
- Adjust hybrid_alpha
- Improve chunk quality
- Add domain-specific embeddings

```python
config.rerank_top_k = 10  # Increase from 5
config.hybrid_alpha = 0.8  # Favor dense search
```

#### 3. Drift Detected

**Symptoms**: drift_magnitude > 0.15

**Solutions**:
- Retrain embeddings
- Update index with fresh data
- Review query distribution changes
- Consider A/B testing

#### 4. Low Faithfulness

**Symptoms**: faithfulness_score < 0.7

**Solutions**:
- Improve chunk granularity
- Enable NLI-based faithfulness checking
- Review source document quality
- Adjust retrieval thresholds

## Performance Tuning

### 1. Embedding Optimization

```python
# Batch embeddings for throughput
embeddings = []
for batch in batches(texts, batch_size=100):
    emb = await generate_embeddings(batch)
    embeddings.extend(emb)
```

### 2. Index Tuning

```yaml
# For accuracy
index_params:
  M: 32           # Higher = more accurate
  efConstruction: 400

# For speed
index_params:
  M: 8            # Lower = faster
  efConstruction: 100
```

### 3. Sharding Strategy

```python
# Distribute load across shards
MilvusIndexManager(
    enable_sharding=True,
    num_shards=8  # Based on data size
)
```

### 4. Caching

```python
# Implement query caching
from functools import lru_cache

@lru_cache(maxsize=1000)
async def cached_retrieve(query):
    return await pipeline.retrieve(query)
```

### 5. Batch Processing

```python
# Process documents in batches
async def batch_ingest(documents, batch_size=50):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        await pipeline.ingest_documents(batch)
```

## Scaling Guidelines

### Horizontal Scaling

1. **Milvus Cluster**:
   - Deploy distributed Milvus
   - Use shared storage (S3/MinIO)
   - Enable load balancing

2. **Pipeline Replicas**:
   - Run multiple pipeline instances
   - Use message queue for coordination
   - Implement circuit breakers

### Vertical Scaling

1. **Increase Resources**:
   - More CPU cores for parallel processing
   - More RAM for larger indexes
   - GPU for faster embeddings

2. **Optimize Configuration**:
   - Tune HNSW parameters
   - Adjust batch sizes
   - Enable prefetching

## Security Considerations

1. **Authentication**: Enable Milvus authentication
2. **Encryption**: Use TLS for Milvus connections
3. **Access Control**: Implement RBAC for compliance data
4. **Data Privacy**: Anonymize sensitive information in logs
5. **Audit Trail**: Maintain immutable audit logs

## Backup & Recovery

### Backup Strategy

```bash
# Backup Milvus data
docker exec milvus-standalone \
  tar -czf /backup/milvus-$(date +%Y%m%d).tar.gz /var/lib/milvus

# Backup compliance data
tar -czf compliance-$(date +%Y%m%d).tar.gz ./data/compliance
```

### Recovery

```bash
# Restore Milvus
docker exec milvus-standalone \
  tar -xzf /backup/milvus-20240101.tar.gz -C /

# Restart services
docker-compose restart
```

## Support & Resources

- Documentation: [Internal Wiki]
- Issues: [GitHub Issues]
- Monitoring: [Grafana Dashboard]
- Logs: [Log Aggregator]

---

**Last Updated**: 2024-01-01
**Version**: 1.0.0
