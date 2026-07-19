FROM python:3.11-slim

# System deps: libpq for psycopg2, build tools for faiss/torch wheels if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev gcc curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x entrypoint.sh

# Persisted across container restarts so the first-run check in
# entrypoint.sh can tell "already trained" from "fresh start".
VOLUME ["/app/state"]

ENV PYTHONUNBUFFERED=1 \
    STATE_DIR=/app/state \
    MOVIELENS_DATA_DIR=/data/movielens \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
