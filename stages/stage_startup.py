"""Stage 0 — Startup: provision all infrastructure before the pipeline runs.

What this stage does
--------------------
1.  Verify PostgreSQL is reachable.
2.  Create the four PostgreSQL schemas (raw, features, serving, pipeline)
    if they do not already exist.
3.  Create all tables from SQLAlchemy ORM models (Base.metadata.create_all).
4.  Verify MinIO is reachable and create the four buckets if missing.
5.  Start the MLflow tracking server if it is not already running, then
    verify it is healthy.  The subprocess is stored in StartupResult so
    callers can manage it.  No external daemon needed.
6.  Print a full configuration summary so the operator can confirm the
    correct values were picked up.

The stage is idempotent — running it multiple times is safe.
MLflow is started as a Python subprocess — no manual server launch required.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# How long (seconds) to wait for MLflow to become healthy after launch
_MLFLOW_STARTUP_TIMEOUT = 30
_MLFLOW_POLL_INTERVAL   = 1

# PostgreSQL schemas that must exist before any table is created
_REQUIRED_SCHEMAS = ["raw", "features", "serving", "pipeline"]

# MinIO bucket names (must match config/registry.yaml)
_REQUIRED_BUCKETS = [
    "model-checkpoints",
    "faiss-indices",
    "teacher-snapshots",
    "cross-distill",
]


@dataclass
class StartupResult:
    db_ok: bool = False
    schemas_created: list[str] = field(default_factory=list)
    migrations_ok: bool = False
    minio_ok: bool = False
    buckets_created: list[str] = field(default_factory=list)
    mlflow_ok: bool = False
    mlflow_process: Optional[subprocess.Popen] = field(default=None, repr=False)
    mlflow_launched: bool = False   # True when we started it, False if it was already up
    all_ok: bool = False
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------

def _check_postgres(result: StartupResult) -> None:
    from db.connection import check_connection
    logger.info("[startup] Checking PostgreSQL connection...")
    if not check_connection():
        msg = "Cannot connect to PostgreSQL. Check RECSYS_DB_URL."
        result.errors.append(msg)
        logger.error("[startup] %s", msg)
        return
    result.db_ok = True
    logger.info("[startup] PostgreSQL: OK")


def _create_schemas(result: StartupResult) -> None:
    from sqlalchemy import text
    from db.connection import get_engine
    logger.info("[startup] Creating PostgreSQL schemas if missing...")
    engine = get_engine()
    with engine.connect() as conn:
        for schema in _REQUIRED_SCHEMAS:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
            conn.commit()
            logger.info("[startup]   schema '%s': OK", schema)
            result.schemas_created.append(schema)


def _create_tables(result: StartupResult) -> None:
    """Create all tables directly from SQLAlchemy ORM models.

    Uses Base.metadata.create_all() — self-contained, no migration scripts
    needed, and fully idempotent (skips tables that already exist).
    After creation, stamps Alembic if scripts exist so future manual
    alembic runs do not try to re-create everything.
    """
    from sqlalchemy import inspect
    from db.connection import get_engine
    from db.models import Base

    logger.info("[startup] Creating tables from ORM models (create_all)...")
    engine = get_engine()

    # Idempotent: skips tables that already exist
    Base.metadata.create_all(engine)

    # Log every table that now exists across all managed schemas
    inspector = inspect(engine)
    for schema in _REQUIRED_SCHEMAS:
        tables = inspector.get_table_names(schema=schema)
        logger.info("[startup]   %s: %s", schema, tables)

    result.migrations_ok = True
    logger.info("[startup] Tables: OK")

    # Stamp Alembic so manual alembic runs later see the DB as up-to-date
    _stamp_alembic_if_available()


def _stamp_alembic_if_available() -> None:
    """Mark current schema as head in Alembic version table (no-op if not set up)."""
    try:
        import os
        from pathlib import Path
        from alembic.config import Config
        from alembic import command

        versions_dir = Path(__file__).parent.parent / "db" / "migrations" / "versions"
        if not versions_dir.exists() or not any(versions_dir.iterdir()):
            return  # no migration scripts — nothing to stamp

        cfg = Config()
        cfg.set_main_option(
            "script_location",
            str(Path(__file__).parent.parent / "db" / "migrations"),
        )
        cfg.set_main_option(
            "sqlalchemy.url",
            os.environ.get(
                "RECSYS_DB_URL",
                "postgresql+psycopg2://postgres:postgres@localhost:5432/movielens",
            ),
        )
        command.stamp(cfg, "head")
        logger.info("[startup] Alembic stamped at head.")
    except Exception as exc:
        logger.debug("[startup] Alembic stamp skipped: %s", exc)


# ---------------------------------------------------------------------------
# MinIO
# ---------------------------------------------------------------------------

def _check_minio(result: StartupResult) -> None:
    logger.info("[startup] Checking MinIO connection...")
    try:
        from config.settings import settings
        from minio import Minio
        from minio.error import S3Error

        client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        # Verify connection by listing buckets
        existing = {b.name for b in client.list_buckets()}
        result.minio_ok = True
        logger.info("[startup] MinIO: OK  (existing buckets: %s)", existing)

        for bucket in _REQUIRED_BUCKETS:
            if bucket not in existing:
                client.make_bucket(bucket)
                result.buckets_created.append(bucket)
                logger.info("[startup]   created bucket: %s", bucket)
            else:
                logger.info("[startup]   bucket '%s': already exists", bucket)

    except Exception as exc:
        msg = f"Cannot connect to MinIO at {__import__('config.settings', fromlist=['settings']).settings.minio_endpoint}: {exc}"
        result.errors.append(msg)
        logger.error("[startup] %s", msg)


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

def _mlflow_is_healthy(uri: str) -> bool:
    """Return True if the MLflow server at *uri* responds to /health."""
    import urllib.request
    import urllib.error
    try:
        host = uri.rstrip("/")
        with urllib.request.urlopen(f"{host}/health", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _launch_mlflow(result: StartupResult, settings) -> subprocess.Popen:
    """Start the MLflow server as a subprocess of this Python process.

    Uses the same Python interpreter that is running main.py so there is no
    dependency on a globally installed `mlflow` binary.
    """
    import os

    cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", settings.mlflow_backend_store_uri,
        "--default-artifact-root", settings.mlflow_artifact_root,
        "--host", "0.0.0.0",
        "--port", str(settings.mlflow_port),
    ]

    env = os.environ.copy()
    env["MLFLOW_S3_ENDPOINT_URL"]  = settings.mlflow_s3_endpoint_url
    env["AWS_ACCESS_KEY_ID"]       = settings.aws_access_key_id
    env["AWS_SECRET_ACCESS_KEY"]   = settings.aws_secret_access_key

    logger.info("[startup] Launching MLflow server: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    result.mlflow_process  = proc
    result.mlflow_launched = True
    return proc


def _ensure_mlflow(result: StartupResult) -> None:
    """Start MLflow if not already running, then verify it is healthy."""
    from config.settings import settings

    uri = settings.mlflow_uri
    logger.info("[startup] Checking MLflow at %s ...", uri)

    if _mlflow_is_healthy(uri):
        logger.info("[startup] MLflow: already running — OK")
        result.mlflow_ok = True
        return

    # Not running — launch it
    logger.info("[startup] MLflow not running. Starting automatically...")
    try:
        proc = _launch_mlflow(result, settings)
    except Exception as exc:
        msg = f"Failed to launch MLflow subprocess: {exc}"
        result.errors.append(msg)
        logger.error("[startup] %s", msg)
        return

    # Wait until healthy or timeout
    deadline = time.time() + _MLFLOW_STARTUP_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            # Process exited prematurely — read any output for diagnostics
            output = ""
            if proc.stdout:
                try:
                    output = proc.stdout.read(2000)
                except Exception:
                    pass
            msg = (
                f"MLflow server process exited unexpectedly "
                f"(return code {proc.returncode}). Output: {output!r}"
            )
            result.errors.append(msg)
            logger.error("[startup] %s", msg)
            return

        if _mlflow_is_healthy(uri):
            result.mlflow_ok = True
            logger.info(
                "[startup] MLflow: started successfully (PID %d, URI: %s)",
                proc.pid, uri,
            )
            return

        logger.debug(
            "[startup] Waiting for MLflow to become healthy... (%.0fs remaining)",
            deadline - time.time(),
        )
        time.sleep(_MLFLOW_POLL_INTERVAL)

    # Timed out
    msg = (
        f"MLflow did not become healthy within {_MLFLOW_STARTUP_TIMEOUT}s. "
        f"Check logs or try increasing _MLFLOW_STARTUP_TIMEOUT."
    )
    result.errors.append(msg)
    logger.error("[startup] %s", msg)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    skip_minio: bool = False,
    skip_mlflow: bool = False,
    skip_migrations: bool = False,
    strict: bool = True,
) -> StartupResult:
    """Provision all infrastructure for the pipeline.

    Args:
        skip_minio:       skip MinIO checks (useful when running ETL only).
        skip_mlflow:      skip MLflow checks (useful when running ETL only).
        skip_migrations:  skip Alembic migrations (if you manage schema externally).
        strict:           raise RuntimeError if any critical service is unavailable.

    Returns:
        StartupResult describing what was created and what succeeded.
    """
    from config.settings import settings

    logger.info("[startup] Configuration: %s", settings.summary())
    logger.info("[startup] Applying settings to environment...")
    settings.apply_to_environment()

    result = StartupResult()

    # PostgreSQL — always required
    _check_postgres(result)
    if not result.db_ok:
        if strict:
            raise RuntimeError(
                "PostgreSQL is not reachable. Cannot proceed.\n"
                f"  DB URL: {settings.effective_db_url()}\n"
                "Fix the connection and re-run."
            )
        return result

    _create_schemas(result)
    if not skip_migrations:
        _create_tables(result)
    else:
        result.migrations_ok = True
        logger.info("[startup] Table creation: SKIPPED")

    # MinIO — required for model storage
    if not skip_minio:
        _check_minio(result)
    else:
        result.minio_ok = True
        logger.info("[startup] MinIO check: SKIPPED")

    # MLflow — required for experiment tracking
    if not skip_mlflow:
        _ensure_mlflow(result)
    else:
        result.mlflow_ok = True
        logger.info("[startup] MLflow check: SKIPPED")

    result.all_ok = (
        result.db_ok
        and result.migrations_ok
        and result.minio_ok
        and result.mlflow_ok
    )

    if result.all_ok:
        logger.info("[startup] All systems ready.")
    else:
        logger.warning("[startup] Some systems are not ready: %s", result.errors)
        if strict:
            raise RuntimeError(
                f"Startup checks failed:\n"
                + "\n".join(f"  - {e}" for e in result.errors)
            )

    return result
