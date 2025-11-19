## Runbook: Advanced RAG API

### 1. Overview

The Advanced RAG API exposes retrieval and chat endpoints backed by Milvus and Postgres. It is instrumented with Prometheus metrics and OpenTelemetry traces.

### 2. Key Dashboards

- Grafana dashboard: `observability/dashboards/rag_overview.json`
  - Request rate
  - Latency P50/P95/P99
  - Error rate

### 3. Common Symptoms and Checks

#### a) High latency on `/retrieve`

- Check `rag_retrieve_latency_ms` in Grafana.
- Inspect `rag_active_requests` gauge to see concurrency.
- Verify Milvus health via `/healthz` (Milvus status) and Milvus logs.

#### b) Elevated error rates

- Inspect `rag_errors_total` by `error_type` (timeout, circuit_breaker_open, unknown).
- Check circuit breaker stats in `/healthz` response.
- Confirm database connectivity (Postgres/SQLite) via `/healthz` and DB logs.

#### c) Ingest pipeline failures

- Review application logs around `/ingest` calls.
- Check compliance/audit logs for ingestion events.

### 4. Mitigation Steps

- For sustained high latency:
  - Reduce `RAG_MAX_CONCURRENCY` or raise resources (CPU/memory) on the API service.
  - Tune Milvus index parameters (HNSW/ef, shard counts) in `infra/terraform/aws`.
- For frequent timeouts:
  - Increase `RAG_RETRIEVE_TIMEOUT_MS` cautiously.
  - Verify downstream dependencies (Milvus, Postgres) are healthy and responsive.

### 5. Escalation

- If user-impacting issues persist beyond 15 minutes:
  - Page the on-call engineer for the Advanced RAG service.
  - Escalate to the platform/SRE team if infra-level issues are suspected.


