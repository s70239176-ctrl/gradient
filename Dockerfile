# ── Stage 1: Build React frontend ──────────────────────────────────────
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --legacy-peer-deps
COPY frontend/ .
RUN npm run build

# ── Stage 2: Python backend ─────────────────────────────────────────────
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.3.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt .
RUN grep -v "^torch" requirements.txt > /tmp/req_notorch.txt \
    && pip install --no-cache-dir -r /tmp/req_notorch.txt

COPY backend/ .
COPY --from=frontend-builder /frontend/dist /app/frontend/dist
RUN mkdir -p /app/model
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 8000

# Shell form so $PORT is evaluated at runtime by Railway
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2
