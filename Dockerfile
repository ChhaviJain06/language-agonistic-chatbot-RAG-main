# BHASHA application image: FastAPI backend + the built React UI. Open-source base only.
FROM node:22-slim AS web
WORKDIR /web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY server ./server
RUN pip install -e ".[rag,graph,voice]"
COPY scripts ./scripts
COPY data ./data
COPY --from=web /web/dist ./web/dist
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
