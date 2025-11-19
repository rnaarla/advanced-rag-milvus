## Service Level Objectives (SLOs) for Advanced RAG API

### Availability

- **SLO**: 99.5% monthly availability for the `/healthz` and `/retrieve` endpoints.
- **Indicator**: HTTP 5xx rate on ALB / API metrics.

### Latency

- **SLO**: P95 end-to-end latency for `/retrieve` under 300 ms for typical queries.
- **Indicator**: Prometheus `rag_retrieve_latency_ms` histogram and Grafana dashboard (`observability/dashboards/rag_overview.json`).

### Error rate

- **SLO**: P99 error rate < 0.5% for `/retrieve` and chat endpoints (excluding client errors).
- **Indicator**: `rag_errors_total` counter partitioned by `error_type`.

### Data quality

- **SLO**: Ingest pipeline completes without fatal errors for ≥ 99% of batches.
- **Indicator**: Ingest logs, compliance/audit events, and evaluation metrics tracking retrieval quality.


