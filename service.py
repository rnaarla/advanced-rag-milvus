"""
FastAPI service for Advanced RAG with chat features.
Endpoints:
 - GET /healthz
 - POST /ingest
 - POST /retrieve
 - POST /chat/start (new chat)
 - POST /chat/clear (clear session)
 - GET /chat/history?session_id=...
 - POST /chat (non-stream)
 - GET /chat/stream?session_id=...&q=... (SSE streaming)
"""

import os
import asyncio
import sqlite3
import psycopg2
import json
import time
from typing import Any, Dict, List, Optional, Generator
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from advanced_rag import AdvancedRAGPipeline, PipelineConfig
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor


class IngestDocument(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Optional[Dict[str, Any]] = None


class IngestRequest(BaseModel):
    documents: List[IngestDocument]
    domain: Optional[str] = None


class RetrieveRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    use_domain_index: Optional[bool] = False
    domain: Optional[str] = None


app = FastAPI(title="Advanced RAG Service")
app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")


def _get_pipeline() -> AdvancedRAGPipeline:
    host = os.getenv("MILVUS_HOST", "localhost")
    port = int(os.getenv("MILVUS_PORT", "19530"))
    # Do not connect to Milvus on startup to allow container health checking without dependencies
    config = PipelineConfig()
    pipeline = AdvancedRAGPipeline(
        milvus_host=host,
        milvus_port=port,
        config=config,
        connect_to_milvus=False
    )
    return pipeline


pipeline_instance: Optional[AdvancedRAGPipeline] = None
milvus_ready: bool = False
db_path = os.getenv("CHAT_DB_PATH", "./chat.db")
DATABASE_URL = os.getenv("DATABASE_URL", "")
REQUEST_COUNTER = Counter("rag_api_requests_total", "API requests", ["endpoint", "method"])
RETRIEVE_LATENCY = Histogram("rag_retrieve_latency_ms", "Retrieve latency (ms)")
API_KEY = os.getenv("API_KEY", "")
# Concurrency, timeout, and circuit breaker settings
MAX_CONCURRENCY = int(os.getenv("RAG_MAX_CONCURRENCY", "64"))
RETRIEVE_TIMEOUT_MS = int(os.getenv("RAG_RETRIEVE_TIMEOUT_MS", "300"))
CB_FAILS = int(os.getenv("RAG_CB_FAILS", "10"))
CB_WINDOW_SEC = int(os.getenv("RAG_CB_WINDOW_SEC", "30"))
CB_OPEN_SEC = int(os.getenv("RAG_CB_OPEN_SEC", "15"))
_sem = asyncio.Semaphore(MAX_CONCURRENCY)
_fail_timestamps: List[float] = []
_cb_open_until: float = 0.0


def _ensure_milvus_connected() -> None:
    global milvus_ready
    if milvus_ready:
        return
    # Attempt connection and collection initialization lazily
    try:
        pipeline_instance.index_manager._connect()
        pipeline_instance.index_manager._initialize_collections()
        milvus_ready = True
    except Exception:
        # Leave milvus_ready as False; operations requiring Milvus will fail with explicit errors
        pass


def _is_postgres() -> bool:
    return DATABASE_URL.startswith("postgres://") or DATABASE_URL.startswith("postgresql://")


def _db():
    if _is_postgres():
        return psycopg2.connect(DATABASE_URL)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _rows_to_dicts(cur, rows):
    if _is_postgres():
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, r)) for r in rows]
    return rows


def _init_db():
    conn = _db()
    cur = conn.cursor()
    if _is_postgres():
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions(
                id TEXT PRIMARY KEY,
                user_id TEXT,
                created_at DOUBLE PRECISION,
                metadata TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages(
                id SERIAL PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                created_at DOUBLE PRECISION
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback(
                id SERIAL PRIMARY KEY,
                message_id INTEGER,
                vote TEXT,
                comment TEXT,
                created_at DOUBLE PRECISION
            )
        """)
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions(
                id TEXT PRIMARY KEY,
                user_id TEXT,
                created_at REAL,
                metadata TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                created_at REAL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER,
                vote TEXT,
                comment TEXT,
                created_at REAL
            )
        """)
    conn.commit()
    conn.close()


def _auth_or_401(request: Request):
    if not API_KEY:
        return
    key = request.headers.get("x-api-key", "")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _cb_check_open() -> bool:
    return time.time() < _cb_open_until


def _cb_record_failure():
    global _cb_open_until
    now = time.time()
    while _fail_timestamps and now - _fail_timestamps[0] > CB_WINDOW_SEC:
        _fail_timestamps.pop(0)
    _fail_timestamps.append(now)
    if len(_fail_timestamps) >= CB_FAILS:
        _cb_open_until = now + CB_OPEN_SEC


def _cb_record_success():
    if _fail_timestamps:
        del _fail_timestamps[: max(1, len(_fail_timestamps) // 2)]


@app.on_event("startup")
async def on_startup():
    global pipeline_instance
    pipeline_instance = _get_pipeline()
    _init_db()
    try:
      endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")
      resource = Resource.create({"service.name": "rag-api"})
      provider = TracerProvider(resource=resource)
      processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
      provider.add_span_processor(processor)
      trace.set_tracer_provider(provider)
      FastAPIInstrumentor.instrument_app(app)
      RequestsInstrumentor().instrument()
    except Exception:
      # Tracing is best-effort; keep service running even if exporter not available
      pass


@app.get("/healthz")
async def healthz():
    REQUEST_COUNTER.labels(endpoint="/healthz", method="GET").inc()
    return {"status": "ok", "milvus_connected": milvus_ready}

@app.get("/")
async def root_index():
    return FileResponse("static/index.html")


@app.post("/ingest")
async def ingest(req: IngestRequest, request: Request):
    REQUEST_COUNTER.labels(endpoint="/ingest", method="POST").inc()
    _auth_or_401(request)
    _ensure_milvus_connected()
    docs = [{"id": d.id, "text": d.text, "metadata": d.metadata or {}} for d in req.documents]
    report = await pipeline_instance.ingest_documents(documents=docs, domain=req.domain)
    return report


@app.post("/retrieve")
async def retrieve(req: RetrieveRequest, request: Request):
    REQUEST_COUNTER.labels(endpoint="/retrieve", method="POST").inc()
    _auth_or_401(request)
    _ensure_milvus_connected()
    if _cb_check_open():
        raise HTTPException(status_code=503, detail="Temporarily unavailable (circuit open)")
    async with _sem:
        try:
            with RETRIEVE_LATENCY.time():
                results, metrics = await asyncio.wait_for(
                    pipeline_instance.retrieve(
                        query=req.query,
                        filters=req.filters,
                        context=req.context
                    ),
                    timeout=RETRIEVE_TIMEOUT_MS / 1000.0
                )
            _cb_record_success()
        except asyncio.TimeoutError:
            _cb_record_failure()
            raise HTTPException(status_code=504, detail="Retrieve timeout")
        except Exception:
            _cb_record_failure()
            raise
    return {
        "results": [
            {
                "content": r.content,
                "chunk_id": r.chunk_id,
                "score": r.score,
                "metadata": r.metadata,
                "retrieval_method": r.retrieval_method,
                "latency_ms": r.latency_ms,
            }
            for r in results
        ],
        "metrics": metrics.to_dict()
    }


@app.on_event("shutdown")
async def on_shutdown():
    if pipeline_instance:
        await pipeline_instance.close()

class FeedbackRequest(BaseModel):
    message_id: int
    vote: str  # "up" | "down"
    comment: Optional[str] = None

@app.post("/feedback")
async def feedback(req: FeedbackRequest, request: Request):
    REQUEST_COUNTER.labels(endpoint="/feedback", method="POST").inc()
    _auth_or_401(request)
    conn = _db()
    cur = conn.cursor()
    if _is_postgres():
        cur.execute(
            "INSERT INTO feedback(message_id, vote, comment, created_at) VALUES(%s,%s,%s,%s)",
            (req.message_id, req.vote, req.comment, time.time())
        )
    else:
        cur.execute(
            "INSERT INTO feedback(message_id, vote, comment, created_at) VALUES(?,?,?,?)",
            (req.message_id, req.vote, req.comment, time.time())
        )
    conn.commit()
    conn.close()
    return {"ok": True}

@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _new_session_id() -> str:
    raw = f"{time.time()}-{os.urandom(8).hex()}"
    return "sess_" + raw.replace(".", "")


def _save_session(session_id: str, user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
    conn = _db()
    cur = conn.cursor()
    if _is_postgres():
        cur.execute(
            "INSERT INTO sessions(id, user_id, created_at, metadata) VALUES(%s,%s,%s,%s)",
            (session_id, user_id, time.time(), json.dumps(metadata or {}))
        )
    else:
        cur.execute(
            "INSERT INTO sessions(id, user_id, created_at, metadata) VALUES(?,?,?,?)",
            (session_id, user_id, time.time(), json.dumps(metadata or {}))
        )
    conn.commit()
    conn.close()


def _append_message(session_id: str, role: str, content: str) -> int:
    conn = _db()
    cur = conn.cursor()
    if _is_postgres():
        cur.execute(
            "INSERT INTO messages(session_id, role, content, created_at) VALUES(%s,%s,%s,%s) RETURNING id",
            (session_id, role, content, time.time())
        )
        mid = cur.fetchone()[0]
    else:
        cur.execute(
            "INSERT INTO messages(session_id, role, content, created_at) VALUES(?,?,?,?)",
            (session_id, role, content, time.time())
        )
        mid = cur.lastrowid
    conn.commit()
    conn.close()
    return mid


def _get_history(session_id: str) -> List[Dict[str, Any]]:
    conn = _db()
    cur = conn.cursor()
    if _is_postgres():
        cur.execute(
            "SELECT id, role, content, created_at FROM messages WHERE session_id=%s ORDER BY id ASC",
            (session_id,)
        )
        rows = cur.fetchall()
        dicts = _rows_to_dicts(cur, rows)
    else:
        rows = cur.execute(
            "SELECT id, role, content, created_at FROM messages WHERE session_id=? ORDER BY id ASC",
            (session_id,)
        ).fetchall()
        dicts = [{"id": r[0], "role": r[1], "content": r[2], "created_at": r[3]} for r in rows]
    conn.close()
    return dicts


def _clear_session(session_id: str):
    conn = _db()
    cur = conn.cursor()
    if _is_postgres():
        cur.execute("DELETE FROM messages WHERE session_id=%s", (session_id,))
    else:
        cur.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    conn.commit()
    conn.close()


class ChatStartRequest(BaseModel):
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    session_id: str
    query: str
    filters: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    use_domain_index: Optional[bool] = False
    domain: Optional[str] = None


class ClearRequest(BaseModel):
    session_id: str
    
class ETLRunRequest(BaseModel):
    source: str  # "fs"
    config: Dict[str, Any]  # e.g., { "path": "./docs", "domain": "technical" }
    
class EvalItem(BaseModel):
    query: str
    relevant_doc_ids: List[str]
    
class EvalRunRequest(BaseModel):
    items: List[EvalItem]


@app.post("/chat/start")
async def chat_start(req: ChatStartRequest, request: Request):
    REQUEST_COUNTER.labels(endpoint="/chat/start", method="POST").inc()
    _auth_or_401(request)
    sid = _new_session_id()
    _save_session(sid, req.user_id, req.metadata)
    return {"session_id": sid}


@app.post("/chat/clear")
async def chat_clear(req: ClearRequest, request: Request):
    REQUEST_COUNTER.labels(endpoint="/chat/clear", method="POST").inc()
    _auth_or_401(request)
    _clear_session(req.session_id)
    return {"session_id": req.session_id, "cleared": True}


@app.get("/chat/history")
async def chat_history(session_id: str, request: Request):
    REQUEST_COUNTER.labels(endpoint="/chat/history", method="GET").inc()
    _auth_or_401(request)
    return {"session_id": session_id, "messages": _get_history(session_id)}


def _make_answer_from_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"text": "No relevant information found.", "citations": [], "context_text": ""}
    top = results[:3]
    answer_parts = []
    citations = []
    context_acc = []
    for r in top:
        snippet = r["content"][:400].replace("\n", " ")
        answer_parts.append(f"- {snippet}")
        citations.append({"chunk_id": r["id"], "score": r["score"], "doc_id": r["metadata"].get("doc_id")})
        context_acc.append(snippet)
    text = "Here is what I found:\n" + "\n".join(answer_parts)
    return {"text": text, "citations": citations, "context_text": " ".join(context_acc)}


def _generate_suggestions(context_text: str, limit: int = 4) -> List[str]:
    tokens = [t for t in context_text.lower().split() if t.isalpha() and len(t) > 3]
    uniq = list(dict.fromkeys(tokens))[:10]
    base = uniq[:4] if len(uniq) >= 4 else uniq + ["details", "examples", "implications", "limitations"]
    patterns = [
        "Give more details on {}",
        "Provide examples about {}",
        "What are the implications of {}?",
        "How does {} compare to alternatives?"
    ]
    suggestions = []
    for i in range(4):
        term = base[i % len(base)]
        suggestions.append(patterns[i].format(term))
    return suggestions[:limit]


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    REQUEST_COUNTER.labels(endpoint="/chat", method="POST").inc()
    _auth_or_401(request)
    _ensure_milvus_connected()
    if not milvus_ready:
        raise HTTPException(status_code=503, detail="Milvus not connected")
    _append_message(req.session_id, "user", req.query)
    if _cb_check_open():
        raise HTTPException(status_code=503, detail="Temporarily unavailable (circuit open)")
    async with _sem:
        try:
            with RETRIEVE_LATENCY.time():
                results, metrics = await asyncio.wait_for(
                    pipeline_instance.retrieve(
                        query=req.query,
                        filters=req.filters,
                        context=req.context
                    ),
                    timeout=RETRIEVE_TIMEOUT_MS / 1000.0
                )
            _cb_record_success()
        except asyncio.TimeoutError:
            _cb_record_failure()
            raise HTTPException(status_code=504, detail="Retrieve timeout")
        except Exception:
            _cb_record_failure()
            raise
    ans = _make_answer_from_results([{
        "id": r.chunk_id,
        "content": r.content,
        "score": r.score,
        "metadata": r.metadata
    } for r in results])
    suggestions = _generate_suggestions(ans["context_text"])
    _append_message(req.session_id, "assistant", ans["text"])
    # Ensure metrics are JSON-serializable (convert NumPy types to native floats)
    try:
        metrics_dict = metrics.to_dict()
        safe_metrics = {k: float(v) for k, v in metrics_dict.items()}
    except Exception:
        safe_metrics = {}
    return {
        "session_id": req.session_id,
        "message": ans["text"],
        "citations": ans["citations"],
        "suggestions": suggestions,
        "metrics": safe_metrics
    }


def _sse_event(data: Dict[str, Any]) -> bytes:
    return (f"data: {json.dumps(data)}\n\n").encode("utf-8")


@app.get("/chat/stream")
async def chat_stream(request: Request, session_id: str, q: str):
    REQUEST_COUNTER.labels(endpoint="/chat/stream", method="GET").inc()
    _auth_or_401(request)
    _ensure_milvus_connected()
    if not milvus_ready:
        raise HTTPException(status_code=503, detail="Milvus not connected")
    _append_message(session_id, "user", q)
    if _cb_check_open():
        raise HTTPException(status_code=503, detail="Temporarily unavailable (circuit open)")
    async with _sem:
        try:
            with RETRIEVE_LATENCY.time():
                results, metrics = await asyncio.wait_for(
                    pipeline_instance.retrieve(query=q),
                    timeout=RETRIEVE_TIMEOUT_MS / 1000.0
                )
            _cb_record_success()
        except asyncio.TimeoutError:
            _cb_record_failure()
            raise HTTPException(status_code=504, detail="Retrieve timeout")
        except Exception:
            _cb_record_failure()
            raise
    ans = _make_answer_from_results([{
        "id": r.chunk_id,
        "content": r.content,
        "score": r.score,
        "metadata": r.metadata
    } for r in results])
    suggestions = _generate_suggestions(ans["context_text"])
    # Ensure metrics are JSON-serializable for SSE payload
    try:
        metrics_dict = metrics.to_dict()
        safe_metrics = {k: float(v) for k, v in metrics_dict.items()}
    except Exception:
        safe_metrics = {}

    async def event_gen() -> Generator[bytes, None, None]:
        text = ans["text"]
        for token in text.split():
            if await request.is_disconnected():
                break
            yield _sse_event({"token": token + " "})
            await asyncio.sleep(0.01)
        yield _sse_event({"done": True, "citations": ans["citations"], "suggestions": suggestions, "metrics": safe_metrics})
        _append_message(session_id, "assistant", text)

    return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.post("/etl/run")
async def etl_run(req: ETLRunRequest, request: Request):
    REQUEST_COUNTER.labels(endpoint="/etl/run", method="POST").inc()
    _auth_or_401(request)
    _ensure_milvus_connected()
    if req.source != "fs":
        raise HTTPException(status_code=400, detail="Only 'fs' source supported in this endpoint")
    path = req.config.get("path")
    if not path or not os.path.isdir(path):
        raise HTTPException(status_code=400, detail="Invalid path")
    domain = req.config.get("domain")
    docs = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith((".txt", ".md")):
                fp = os.path.join(root, f)
                try:
                    with open(fp, "r", encoding="utf-8") as fh:
                        text = fh.read()
                    docs.append({"id": os.path.relpath(fp, path), "text": text, "metadata": {"source": "fs", "path": fp}})
                except Exception:
                    continue
    if not docs:
        return {"ingested": 0, "chunks": 0}
    report = await pipeline_instance.ingest_documents(docs, domain=domain)
    return {"ingested": report.get("total_documents", 0), "chunks": report.get("chunks_created", 0)}

@app.post("/eval/run")
async def eval_run(req: EvalRunRequest, request: Request):
    REQUEST_COUNTER.labels(endpoint="/eval/run", method="POST").inc()
    _auth_or_401(request)
    _ensure_milvus_connected()
    from advanced_rag import RAGEvaluator
    evaluator = RAGEvaluator()
    per_query = []
    for item in req.items:
        results, metrics = await pipeline_instance.retrieve(
            query=item.query,
            context={"relevant_doc_ids": item.relevant_doc_ids}
        )
        per_query.append(metrics.to_dict())
    if not per_query:
        return {"count": 0, "metrics": {}}
    keys = per_query[0].keys()
    agg = {k: float(sum(d[k] for d in per_query) / len(per_query)) for k in keys}
    return {"count": len(per_query), "metrics": agg}

