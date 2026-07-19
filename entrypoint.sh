#!/usr/bin/env bash
# Entrypoint for the MovieLens RecSys container.
#
# First run (no marker file yet, i.e. no trained weights/optimizations
# persisted from a previous container lifetime):
#   startup -> ingest -> validate -> featurize -> split -> tune -> train
#           -> cross_distill -> evaluate -> precompute -> serve
#
# Subsequent runs (marker present -> models/checkpoints already exist in
# MinIO/Postgres from a prior run of this same persisted volume):
#   startup -> serve   (retraining is handled separately by Airflow DAGs)
set -euo pipefail

STATE_DIR="${STATE_DIR:-/app/state}"
MARKER_FILE="${STATE_DIR}/.pipeline_trained"
mkdir -p "${STATE_DIR}"

echo "[entrypoint] Waiting for PostgreSQL..."
python - <<'PY'
import os, time
from db.connection import check_connection
for _ in range(60):
    if check_connection():
        break
    time.sleep(2)
else:
    raise SystemExit("[entrypoint] PostgreSQL never became reachable.")
PY

if [ ! -f "${MARKER_FILE}" ]; then
    echo "[entrypoint] No marker at ${MARKER_FILE} — first run detected."
    echo "[entrypoint] Running full pipeline (ingest through precompute, with hparam tuning)."

    python main.py \
        --stages startup ingest validate featurize split tune train cross_distill evaluate precompute \
        --data-dir "${MOVIELENS_DATA_DIR:-/data/movielens}" \
        --ingest-mode "${INGEST_MODE:-upsert}"

    touch "${MARKER_FILE}"
    echo "[entrypoint] Full pipeline complete — marker written, weights/optimizations persisted."
else
    echo "[entrypoint] Marker found at ${MARKER_FILE} — skipping ingest/tune/train."
    echo "[entrypoint] Re-running startup checks only, then serving."
    python main.py --stages startup
fi

echo "[entrypoint] Starting API server."
exec python main.py --stages serve --host 0.0.0.0 --port "${PORT:-8000}"
