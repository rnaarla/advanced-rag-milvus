FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install -U pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install fastapi uvicorn[standard] pydantic

COPY advanced_rag/ ./advanced_rag/
COPY service.py ./service.py
COPY static/ ./static/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
 CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]


