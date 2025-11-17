# Quick Start Guide

Get started with the Advanced RAG Pipeline in 5 minutes!

## 🚀 Quick Setup

### 1. Install Prerequisites

```bash
# Python 3.8+ required
python --version

# Install Docker
# Visit: https://docs.docker.com/get-docker/
```

### 2. Clone and Install

```bash
# Navigate to the project directory
cd advanced-rag-milvus

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Start Milvus

```bash
# Start Milvus using Docker Compose
docker-compose up -d

# Wait ~30 seconds for services to initialize

# Verify it's running
curl http://localhost:9091/healthz
# Should return: OK
```

### 4. Run Example

```bash
# Run the example script
python example_usage.py
```

Expected output:
```
================================================================================
Advanced RAG Pipeline Example
================================================================================

✓ Pipeline initialized with configuration:
  - Target latency: 80.0ms
  - Hybrid retrieval alpha: 0.7
  - Reranking: enabled
  - Audit logging: enabled

================================================================================
STAGE 1: Document Ingestion with Diagnostics
================================================================================

Ingesting documents...

✓ Ingestion completed in 1234.56ms
  - Documents processed: 3
  - Chunks created: 12
...
```

## 📝 Your First Query

Create a simple script (`my_first_rag.py`):

```python
import asyncio
from advanced_rag import AdvancedRAGPipeline

async def main():
    # Initialize
    pipeline = AdvancedRAGPipeline()
    
    # Add documents
    docs = [{
        "id": "doc1",
        "text": "Python is a high-level programming language.",
        "metadata": {"source": "tutorial"}
    }]
    
    await pipeline.ingest_documents(docs)
    
    # Query
    results, metrics = await pipeline.retrieve("What is Python?")
    
    print(f"Top result: {results[0].content}")
    print(f"Score: {results[0].score:.3f}")
    print(f"Hallucination risk: {metrics.hallucination_risk:.3f}")
    
    await pipeline.close()

asyncio.run(main())
```

Run it:
```bash
python my_first_rag.py
```

## 🎯 Next Steps

### Customize Configuration

1. **Copy the config template**:
   ```bash
   cp config.template.yaml config.yaml
   ```

2. **Edit settings**:
   ```yaml
   pipeline:
     target_latency_ms: 80.0
     hybrid_alpha: 0.7  # Adjust dense/sparse ratio
     top_k: 20          # Number of results
   ```

3. **Load config in code**:
   ```python
   import yaml
   
   with open('config.yaml') as f:
       config_dict = yaml.safe_load(f)
   
   pipeline = AdvancedRAGPipeline(
       config=PipelineConfig(**config_dict['pipeline'])
   )
   ```

### Add Your Own Embeddings

```python
from openai import AsyncOpenAI
import numpy as np

class MyEmbeddings:
    def __init__(self):
        self.client = AsyncOpenAI()
    
    async def encode_semantic(self, text: str):
        response = await self.client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return np.array(response.data[0].embedding)

# Set in pipeline
pipeline.index_manager.embedding_generator = MyEmbeddings()
```

### Enable Reranking

```python
from advanced_rag import CrossEncoderReranker

reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"
)

pipeline.retriever.reranker = reranker
```

## 🔧 Common Tasks

### Check Pipeline Health

```python
# Get performance report
report = pipeline.get_performance_report()

print(f"P95 latency: {report['stage_latencies']['retrieval']['p95']:.2f}ms")
print(f"SLA compliance: {report['sla_compliance']['compliance_rate']:.1%}")
```

### Monitor Quality

```python
# Detect drift
drift_report = await pipeline.detect_drift(
    sample_queries=["query1", "query2", "query3"]
)

if drift_report.drift_detected:
    print("⚠️  Drift detected!")
    for rec in drift_report.recommendations:
        print(f"  - {rec}")
```

### View Audit Logs

```python
# Query logs
logs = pipeline.compliance.query_audit_logs(
    start_time="2024-01-01",
    end_time="2024-12-31"
)

print(f"Total events: {len(logs)}")
```

### Generate Reports

```python
# Compliance report
report = pipeline.compliance.generate_compliance_report(
    start_date="2024-01-01",
    end_date="2024-01-31"
)

print(f"Total events: {report['total_events']}")
print(f"Event breakdown: {report['event_counts']}")
```

## 🐛 Troubleshooting

### Milvus Connection Error

```
Error: Connection refused to localhost:19530
```

**Solution**:
```bash
# Check if Milvus is running
docker ps | grep milvus

# If not, start it
docker-compose up -d

# Wait 30 seconds, then retry
```

### Import Error

```
ModuleNotFoundError: No module named 'advanced_rag'
```

**Solution**:
```bash
# Install in development mode
pip install -e .

# Or install with extras
pip install -e ".[embeddings,reranking]"
```

### Low Performance

**Symptoms**: Queries taking > 200ms

**Quick Fixes**:
1. Reduce `top_k`: `config.top_k = 10`
2. Disable reranking initially: `config.enable_reranking = False`
3. Lower HNSW `ef`: `search_params["params"]["ef"] = 32`

## 📚 Learn More

- **Full Documentation**: See `README.md`
- **Deployment Guide**: See `DEPLOYMENT.md`
- **Example Code**: See `example_usage.py`
- **Tests**: See `test_advanced_rag.py`

## 💡 Tips

1. **Start Simple**: Begin with default configs, tune later
2. **Monitor Quality**: Check hallucination risk regularly
3. **Use Audit Logs**: Essential for compliance and debugging
4. **Test Locally**: Validate before production deployment
5. **Scale Gradually**: Start with 1 shard, add more as needed

## 🆘 Get Help

If you run into issues:

1. Check the logs: `docker-compose logs milvus-standalone`
2. Review test output: `pytest test_advanced_rag.py -v`
3. Enable debug logging: `config.log_level = "DEBUG"`
4. Check Milvus health: `curl http://localhost:9091/healthz`

---

**Ready to build production RAG systems? Let's go! 🚀**
