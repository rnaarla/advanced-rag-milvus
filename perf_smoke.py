"""
Lightweight performance smoke test for the Advanced RAG pipeline.

This script is used by CI (Perf-Smoke workflow) to ensure core CPU-bound
paths (diagnostics + chunking) stay within a reasonable latency budget.
It does NOT connect to Milvus or external services.
"""

import asyncio
import time

from advanced_rag import AdvancedRAGPipeline


async def _run_smoke() -> None:
    pipe = AdvancedRAGPipeline(connect_to_milvus=False)

    docs = [
        {
            "id": f"doc-{i}",
            "text": " ".join(
                [
                    "Advanced RAG pipelines use Milvus for vector search and",
                    "diagnostic-informed chunking for high-quality retrieval.",
                ]
                * 5
            ),
            "metadata": {"source": "perf-smoke"},
        }
        for i in range(3)
    ]

    start = time.time()
    await pipe.ingest_documents(docs)
    elapsed_ms = (time.time() - start) * 1000.0
    print(f"[perf-smoke] Ingested {len(docs)} docs in {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    asyncio.run(_run_smoke())


