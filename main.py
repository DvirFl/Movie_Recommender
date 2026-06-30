#!/usr/bin/env python3
"""main.py — single entry point for the MovieLens RecSys pipeline.

Startup sequence
----------------
1.  Load .env file (if present) into os.environ.
2.  Apply CLI flag overrides to os.environ.
3.  Build Settings from the now-populated environment.
4.  Run the 'startup' stage (DB schemas, migrations, MinIO buckets, service checks).
5.  Run the requested pipeline stages in order.

Environment variables
---------------------
All configuration is resolved from environment variables.  The full list is in
.env.example.  Key variables:

    RECSYS_DB_URL         PostgreSQL connection URL
    MOVIELENS_DATA_DIR    path to MovieLens CSV files
    MLFLOW_TRACKING_URI   MLflow server URI
    MINIO_ENDPOINT        MinIO host:port
    RECSYS_ENV            development | test | production

Resolution order (highest priority first):
    1. CLI flags (--db-url, --data-dir, --mlflow-uri, --minio-endpoint)
    2. Shell environment variables
    3. .env file (in project root or path given by --env-file)
    4. Defaults in config/settings.py

Usage examples
--------------
    # Full pipeline:
    python main.py

    # ETL only, pointing at a custom data directory:
    python main.py --stages ingest validate featurize split --data-dir ./data/ml-small

    # Train one architecture, skip tuning, no MinIO:
    python main.py --stages train --architectures TwoTower --skip-tune --no-minio

    # From featurize to evaluate (skip ingest/validate):
    python main.py --from-stage featurize --to-stage evaluate

    # Re-run startup checks only (verify services are up):
    python main.py --stages startup

    # Serve the API:
    python main.py --stages serve --port 8000 --reload

    # Dry run — print what would execute without doing it:
    python main.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging — configured before any import that might log
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Stage registry — insertion order = default execution order
# ---------------------------------------------------------------------------
ALL_STAGES = [
    "startup",
    "ingest",
    "validate",
    "featurize",
    "split",
    "tune",
    "train",
    "cross_distill",
    "evaluate",
    "precompute",
    "serve",
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python main.py",
        description="MovieLens RecSys pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Stage selection ---------------------------------------------------
    p.add_argument(
        "--stages", nargs="+", choices=ALL_STAGES, metavar="STAGE", default=None,
        help=f"Explicit list of stages to run. Choices: {', '.join(ALL_STAGES)}",
    )
    p.add_argument(
        "--from-stage", choices=ALL_STAGES, metavar="STAGE", default=None,
        help="Run from this stage to the end (or --to-stage).",
    )
    p.add_argument(
        "--to-stage", choices=ALL_STAGES, metavar="STAGE", default=None,
        help="Stop after this stage (inclusive).",
    )
    p.add_argument(
        "--skip-startup", action="store_true",
        help="Skip the startup stage (assumes services are already running).",
    )

    # --- Environment / config overrides -----------------------------------
    p.add_argument(
        "--env-file", default=None, metavar="PATH",
        help="Path to a .env file (default: looks for .env in project root).",
    )
    p.add_argument(
        "--db-url", default=None, metavar="URL",
        help="PostgreSQL connection URL. Overrides RECSYS_DB_URL.",
    )
    p.add_argument(
        "--data-dir", default=None, metavar="PATH",
        help="Path to MovieLens CSV directory. Overrides MOVIELENS_DATA_DIR.",
    )
    p.add_argument(
        "--mlflow-uri", default=None, metavar="URI",
        help="MLflow tracking server URI. Overrides MLFLOW_TRACKING_URI.",
    )
    p.add_argument(
        "--minio-endpoint", default=None, metavar="HOST:PORT",
        help="MinIO endpoint. Overrides MINIO_ENDPOINT.",
    )

    # --- ETL params -------------------------------------------------------
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac",   type=float, default=0.1)
    p.add_argument("--decay-lambda", type=float, default=0.1)
    p.add_argument(
        "--no-strict-validate", action="store_true",
        help="Log validation failures instead of stopping the pipeline.",
    )
    p.add_argument(
        "--ingest-mode",
        choices=["upsert", "skip", "replace"],
        default="upsert",
        dest="ingest_mode",
        help=(
            "How to handle rows that already exist in the DB: "
            "upsert=update existing (default, safe to re-run), "
            "skip=keep existing/ignore dupes, "
            "replace=delete all rows first then reload."
        ),
    )

    # --- Model filtering --------------------------------------------------
    p.add_argument(
        "--losses", nargs="+", default=["all"], metavar="LOSS",
        help="Loss names to include, or 'all' (default).",
    )
    p.add_argument(
        "--architectures", nargs="+", default=["all"], metavar="ARCH",
        help="Architecture names to include, or 'all' (default).",
    )

    # --- Tuning -----------------------------------------------------------
    p.add_argument("--n-trials", type=int, default=None,
                   help="Override Optuna n_trials.")
    p.add_argument(
        "--skip-tune", action="store_true",
        help="Skip tuning and use default hparams for training.",
    )

    # --- Training ---------------------------------------------------------
    p.add_argument(
        "--no-minio", action="store_true",
        help="Disable MinIO checkpoint uploads.",
    )
    p.add_argument("--cross-distill-epochs", type=int, default=2)

    # --- Precompute -------------------------------------------------------
    p.add_argument("--top-n", type=int, default=20)

    # --- Startup ----------------------------------------------------------
    p.add_argument(
        "--skip-migrations", action="store_true",
        help="Skip Alembic migrations during startup.",
    )
    p.add_argument(
        "--no-strict-startup", action="store_true",
        help="Warn instead of raising when a service is unreachable at startup.",
    )

    # --- Serving ----------------------------------------------------------
    p.add_argument("--host",    default="0.0.0.0")
    p.add_argument("--port",    type=int, default=8000)
    p.add_argument("--reload",  action="store_true")
    p.add_argument("--workers", type=int, default=1)

    # --- Misc -------------------------------------------------------------
    p.add_argument("--dry-run", action="store_true",
                   help="Print stages that would run without executing.")
    p.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )

    return p


# ---------------------------------------------------------------------------
# Environment bootstrap
# ---------------------------------------------------------------------------

def bootstrap_environment(args: argparse.Namespace) -> None:
    """Load .env then apply CLI overrides into os.environ.

    This must run before config.settings is imported so the Settings
    dataclass reads the final values.
    """
    # Step 1 — load .env file
    from config.env_loader import load_env
    env_path = load_env(path=args.env_file)
    if env_path:
        log.info("Loaded environment from: %s", env_path)
    else:
        log.debug("No .env file found — using shell environment only.")

    # Step 2 — apply CLI overrides (highest priority)
    overrides = {
        "RECSYS_DB_URL":         args.db_url,
        "MOVIELENS_DATA_DIR":    args.data_dir,
        "MLFLOW_TRACKING_URI":   args.mlflow_uri,
        "MINIO_ENDPOINT":        args.minio_endpoint,
    }
    for key, value in overrides.items():
        if value is not None:
            os.environ[key] = value
            log.info("CLI override: %s=%s", key, value)

    # Step 3 — reload Settings so it picks up the updated environment
    from config.settings import get_settings
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Stage resolution
# ---------------------------------------------------------------------------

def resolve_stages(args: argparse.Namespace) -> list[str]:
    if args.stages:
        stages = list(args.stages)
    elif args.from_stage or args.to_stage:
        start = ALL_STAGES.index(args.from_stage) if args.from_stage else 0
        stop  = ALL_STAGES.index(args.to_stage) + 1 if args.to_stage else len(ALL_STAGES)
        stages = ALL_STAGES[start:stop]
    else:
        stages = list(ALL_STAGES)

    # 'startup' is always first unless explicitly skipped
    if args.skip_startup and "startup" in stages:
        stages = [s for s in stages if s != "startup"]
        log.info("--skip-startup: removed 'startup' from stage list.")
    elif "startup" not in stages and not args.skip_startup:
        # If stages were selected explicitly but don't include startup,
        # prepend it so services are always verified first.
        # (Unless user explicitly used --skip-startup.)
        stages = ["startup"] + stages

    if args.skip_tune and "tune" in stages:
        stages = [s for s in stages if s != "tune"]
        log.info("--skip-tune: removed 'tune' from stage list.")

    return stages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _losses_arg(args: argparse.Namespace) -> list[str] | str:
    raw = args.losses
    return "all" if raw == ["all"] else raw


def _archs_arg(args: argparse.Namespace) -> list[str] | str:
    raw = args.architectures
    return "all" if raw == ["all"] else raw


# ---------------------------------------------------------------------------
# Stage dispatcher
# ---------------------------------------------------------------------------

def run_stage(stage: str, args: argparse.Namespace, state: dict) -> None:
    t0 = time.time()

    if stage == "startup":
        from stages.stage_startup import run
        state["startup"] = run(
            skip_minio=args.no_minio,
            skip_mlflow=False,
            skip_migrations=args.skip_migrations,
            strict=not args.no_strict_startup,
        )

    elif stage == "ingest":
        from stages.stage_ingest import run
        from config.settings import get_settings
        data_dir = args.data_dir or get_settings().data_dir
        state["ingest"] = run(data_dir=data_dir, mode=args.ingest_mode)

    elif stage == "validate":
        from stages.stage_validate import run
        state["validate"] = run(strict=not args.no_strict_validate)

    elif stage == "featurize":
        from stages.stage_featurize import run
        state["featurize"] = run(decay_lambda=args.decay_lambda)

    elif stage == "split":
        from stages.stage_split import run
        state["split"] = run(train_frac=args.train_frac, val_frac=args.val_frac)

    elif stage == "tune":
        from stages.stage_tune import run
        from config.settings import get_settings
        state["tune"] = run(
            losses=_losses_arg(args),
            architectures=_archs_arg(args),
            n_trials=args.n_trials,
            mlflow_tracking_uri=args.mlflow_uri or get_settings().mlflow_uri,
        )

    elif stage == "train":
        from stages.stage_train import run
        from config.settings import get_settings
        best_hparams = getattr(state.get("tune"), "best_hparams", None)
        state["train"] = run(
            best_hparams=best_hparams,
            losses=_losses_arg(args),
            architectures=_archs_arg(args),
            save_to_minio=not args.no_minio,
            mlflow_tracking_uri=args.mlflow_uri or get_settings().mlflow_uri,
            trigger_type="manual",
        )

    elif stage == "cross_distill":
        from stages.stage_cross_distill import run
        from config.settings import get_settings
        state["cross_distill"] = run(
            mlflow_tracking_uri=args.mlflow_uri or get_settings().mlflow_uri,
            n_epochs=args.cross_distill_epochs,
        )

    elif stage == "evaluate":
        from stages.stage_evaluate import run
        from config.settings import get_settings
        state["evaluate"] = run(
            losses=_losses_arg(args),
            architectures=_archs_arg(args),
            mlflow_tracking_uri=args.mlflow_uri or get_settings().mlflow_uri,
        )

    elif stage == "precompute":
        from stages.stage_precompute import run
        state["precompute"] = run(
            losses=_losses_arg(args),
            architectures=_archs_arg(args),
            top_n=args.top_n,
        )

    elif stage == "serve":
        from stages.stage_serve import run
        state["serve"] = run(
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers,
        )

    log.info("Stage '%s' completed in %.1fs.", stage, time.time() - t0)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(stages_run: list[str], state: dict, total_time: float) -> None:
    log.info("")
    log.info("=" * 60)
    log.info("Pipeline summary — %.1fs total", total_time)
    log.info("=" * 60)

    for stage in stages_run:
        result = state.get(stage)
        if result is None:
            log.info("  %-16s  (no result)", stage)
            continue

        if stage == "startup":
            status = "OK" if result.all_ok else "PARTIAL"
            log.info("  %-16s  %s  schemas=%s  buckets_created=%s",
                     stage, status, result.schemas_created, result.buckets_created)
            if result.errors:
                for e in result.errors:
                    log.warning("               ↳ %s", e)

        elif stage == "ingest":
            log.info("  %-16s  %s", stage, result.counts)

        elif stage == "validate":
            log.info("  %-16s  passed=%s  stats=%s", stage, result.passed, result.stats)

        elif stage == "featurize":
            log.info("  %-16s  users=%d  items=%d", stage, result.n_users, result.n_items)

        elif stage == "split":
            log.info("  %-16s  train=%d  val=%d  test=%d",
                     stage, result.train, result.val, result.test)

        elif stage == "tune":
            log.info("  %-16s  combinations=%d", stage, len(result.best_hparams))

        elif stage == "train":
            for key, run_id in result.run_ids.items():
                log.info("  %-16s  %s → %s", stage, key, run_id)

        elif stage == "cross_distill":
            if result.skipped:
                log.info("  %-16s  SKIPPED (%s)", stage, result.reason)
            else:
                log.info("  %-16s  pairs=%s", stage, result.pairs_run)

        elif stage == "evaluate":
            for key, loss in result.test_losses.items():
                log.info("  %-16s  %s → test_loss=%.6f", stage, key, loss)

        elif stage == "precompute":
            for key, counts in result.counts.items():
                log.info("  %-16s  %s → %s", stage, key, counts)

        elif stage == "serve":
            log.info("  %-16s  %s:%d", stage, result.host, result.port)

    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args   = parser.parse_args(argv)

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # ---- Bootstrap: .env → CLI overrides → Settings ---------------------
    bootstrap_environment(args)

    # ---- Log final settings so the operator can verify ------------------
    from config.settings import get_settings
    s = get_settings()
    log.info("Effective configuration: %s", s.summary())

    # ---- Resolve stages -------------------------------------------------
    stages = resolve_stages(args)
    if not stages:
        log.error("No stages selected.")
        return 1

    log.info("Stages to run: %s", " → ".join(stages))

    if args.dry_run:
        log.info("[dry-run] Would run: %s", " → ".join(stages))
        log.info("[dry-run] Configuration: %s", s.summary())
        return 0

    # ---- Execute stages -------------------------------------------------
    state: dict        = {}
    pipeline_start     = time.time()
    failed_stage: str | None = None

    for stage in stages:
        log.info("")
        log.info("─" * 60)
        log.info("▶  Stage: %s", stage.upper())
        log.info("─" * 60)
        try:
            run_stage(stage, args, state)
        except KeyboardInterrupt:
            log.warning("Interrupted during stage '%s'.", stage)
            failed_stage = stage
            break
        except Exception as exc:
            log.exception("Stage '%s' failed: %s", stage, exc)
            failed_stage = stage
            break

    total_time        = time.time() - pipeline_start
    completed         = stages[: stages.index(failed_stage)] if failed_stage else stages
    print_summary(completed, state, total_time)

    if failed_stage:
        log.error("Pipeline stopped at stage '%s'.", failed_stage)
        return 1

    log.info("Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
