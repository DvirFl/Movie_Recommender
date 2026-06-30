"""Pure pipeline logic — no Airflow imports anywhere in this file.

All task callables and helpers that need unit testing live here.
The DAG files (common.py, dag_*.py) import from this module so the
logic is testable without an Airflow runtime.
"""
from __future__ import annotations

import datetime as dt
import importlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def get_active_combinations(conf: dict[str, Any]):
    """Return registry combinations filtered by the trigger conf payload.

    Args:
        conf: Airflow dag_run.conf dict.
              Keys: 'losses' (list | 'all'), 'architectures' (list | 'all').

    Returns:
        List of (ArchEntry, LossEntry) tuples that are both enabled and match
        the requested filters.
    """
    from training.registry import ComponentRegistry
    reg = ComponentRegistry()

    req_archs = conf.get("architectures", "all")
    req_losses = conf.get("losses", "all")

    arch_filter = None if req_archs == "all" else (
        req_archs if isinstance(req_archs, list) else [req_archs]
    )
    loss_filter = None if req_losses == "all" else (
        req_losses if isinstance(req_losses, list) else [req_losses]
    )
    return reg.filter_combinations(
        architecture_names=arch_filter,
        loss_names=loss_filter,
    )


def should_run_combination(arch_name: str, loss_name: str, conf: dict) -> bool:
    """Return True if this arch×loss pair is included in the trigger conf."""
    active = get_active_combinations(conf)
    return (arch_name, loss_name) in {(a.name, l.name) for a, l in active}


def resolve_trainer(arch_name: str):
    """Return the train() callable for the given architecture name.

    Derives the trainer module path from the architecture's registry module:
        training.two_tower.towers  →  training.two_tower.trainer
    """
    from training.registry import ComponentRegistry
    arch_entry = ComponentRegistry().get_arch_entry(arch_name)
    parts = arch_entry.cls.__module__.split(".")   # e.g. ['training','two_tower','towers']
    trainer_path = ".".join(parts[:-1]) + ".trainer"
    return importlib.import_module(trainer_path).train


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_datasets():
    """Return (dataset_dict, user_features, item_features, n_users, n_items)."""
    from etl.utils import load_split_dataframes, load_user_features, load_item_features
    from training.dataset import MovieLensDataset

    splits_raw = load_split_dataframes()
    uf = load_user_features()
    itf = load_item_features()
    n_users = max(uf.keys()) + 1
    n_items = max(itf.keys()) + 1
    datasets = {
        split: MovieLensDataset(df, uf, itf, split=split)
        for split, df in splits_raw.items()
    }
    return datasets, uf, itf, n_users, n_items


def mlflow_uri() -> str:
    from config import get_mlflow_config
    return get_mlflow_config()["tracking_uri"]


# ---------------------------------------------------------------------------
# Trigger log
# ---------------------------------------------------------------------------

def update_trigger_log(trigger_id: str | None, status: str) -> None:
    """Update pipeline.trigger_log — no-op when trigger_id is None."""
    if not trigger_id:
        return
    try:
        from db.connection import get_session
        from db.models import TriggerLog
        with get_session() as session:
            entry = session.get(TriggerLog, trigger_id)
            if entry:
                entry.status = status
                if status in ("success", "failed"):
                    entry.completed_at = dt.datetime.now(tz=dt.timezone.utc)
    except Exception as exc:
        logger.warning("Could not update trigger log: %s", exc)


# ---------------------------------------------------------------------------
# Watermark sensor logic
# ---------------------------------------------------------------------------

RAW_TABLES = [
    "raw.ratings",
    "raw.movies",
    "raw.tags",
    "raw.links",
    "raw.genome_scores",
]


def check_watermark(tables: list[str]) -> tuple[bool, str | None]:
    """Return (has_new_data, table_name_that_fired | None).

    Queries pipeline.trigger_watermarks for each table and returns True as
    soon as any table has rows inserted after the last recorded watermark.
    """
    from sqlalchemy import text
    from db.connection import get_session

    with get_session() as session:
        for table in tables:
            schema, tbl = table.split(".")
            new_rows = session.execute(
                text(f"""
                    SELECT COUNT(*)
                    FROM {schema}.{tbl} t
                    LEFT JOIN pipeline.trigger_watermarks w
                           ON w.table_name = :tname
                    WHERE w.last_inserted_at IS NULL
                       OR t.inserted_at > w.last_inserted_at
                """),
                {"tname": table},
            ).scalar() or 0

            if new_rows > 0:
                logger.info(
                    "Watermark check: %d new row(s) in %s.", new_rows, table
                )
                return True, table

    return False, None


# ---------------------------------------------------------------------------
# Task callables — each independently callable without Airflow
# ---------------------------------------------------------------------------

def run_ingest(**context) -> dict:
    from etl.ingest import ingest_all
    data_dir = os.environ.get("MOVIELENS_DATA_DIR", "/data/movielens")
    counts = ingest_all(data_dir)
    logger.info("Ingest complete: %s", counts)
    return counts


def run_validate(**context) -> dict:
    from etl.validate import validate_all
    report = validate_all()
    if not report.passed:
        raise ValueError(f"Validation failed: {report.issues}")
    return report.stats


def run_featurize(**context) -> dict:
    from etl.featurize import featurize_all
    counts = featurize_all()
    logger.info("Featurize complete: %s", counts)
    return counts


def run_split(**context) -> dict:
    from etl.split import split_ratings
    counts = split_ratings()
    logger.info("Split complete: %s", counts)
    return counts


def run_tune(arch_name: str, loss_name: str, **context) -> dict:
    """Optuna sweep for one arch×loss. Returns best hparams dict."""
    import mlflow
    from training.registry import ComponentRegistry
    from training.hparam.tuner import run_sweep

    mlflow.set_tracking_uri(mlflow_uri())

    registry = ComponentRegistry()
    arch_entry = registry.get_arch_entry(arch_name)
    loss_entry = registry.get_loss_entry(loss_name)

    datasets, _, _, n_users, n_items = load_datasets()
    train_fn = resolve_trainer(arch_name)

    def objective(hparams: dict) -> float:
        arch = arch_entry.cls(n_users=n_users, n_items=n_items)
        loss = loss_entry.cls()
        # Halve epochs during sweep for speed
        sweep_hparams = {**hparams, "n_epochs": max(1, hparams.get("n_epochs", 2) // 2)}
        run_id = train_fn(
            arch, loss,
            datasets["train"], datasets["val"],
            hparams=sweep_hparams,
            experiment_name=f"hparam/{arch_name}_{loss_name}",
            save_to_minio=False,
            mlflow_tags={"sweep_trial": "true"},
        )
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        return float(run.data.metrics.get("best_val_loss", 9999.0))

    best = run_sweep(
        arch_cls=arch_entry.cls,
        arch_kwargs={"n_users": n_users, "n_items": n_items},
        loss_cls=loss_entry.cls,
        objective_fn=objective,
        experiment_name=f"hparam/{arch_name}_{loss_name}",
        study_name=f"{arch_name}_{loss_name}",
    )
    logger.info("Best hparams for %s×%s: %s", arch_name, loss_name, best)
    return best


def run_train(arch_name: str, loss_name: str, **context) -> str:
    """Train arch×loss with best hparams from XCom. Returns MLflow run_id."""
    import mlflow
    from training.registry import ComponentRegistry

    dag_run = context.get("dag_run") or type("DR", (), {"conf": {}, "run_id": "local"})()
    ti = context.get("ti")
    conf = getattr(dag_run, "conf", {}) or {}

    best_hparams: dict = {}
    if ti is not None:
        best_hparams = ti.xcom_pull(task_ids=f"tune_{arch_name}_{loss_name}") or {}
    if not best_hparams:
        logger.warning("No hparams from tune — using defaults for %s×%s.", arch_name, loss_name)
        best_hparams = {
            "lr": 1e-3, "n_epochs": 5, "batch_size": 512,
            "sdft_weight": 0.1, "ema_alpha": 0.02,
        }

    mlflow.set_tracking_uri(mlflow_uri())

    registry = ComponentRegistry()
    arch_entry = registry.get_arch_entry(arch_name)
    loss_entry = registry.get_loss_entry(loss_name)

    datasets, _, _, n_users, n_items = load_datasets()
    arch = arch_entry.cls(n_users=n_users, n_items=n_items)
    loss = loss_entry.cls()
    train_fn = resolve_trainer(arch_name)

    run_id = train_fn(
        arch, loss,
        datasets["train"], datasets["val"],
        hparams=best_hparams,
        run_name=f"{arch_name}_{loss_name}_{getattr(dag_run, 'run_id', 'local')}",
        experiment_name=f"train/{arch_name}_{loss_name}",
        mlflow_tags={
            "trigger_type": conf.get("trigger_type", "unknown"),
            "trigger_id":   conf.get("trigger_id", ""),
            "dag_run_id":   getattr(dag_run, "run_id", ""),
        },
    )
    logger.info("Training complete — run_id: %s", run_id)
    return run_id


def run_cross_distill(**context) -> None:
    """Cross-architecture SDFT for every ordered pair of trained models."""
    import mlflow
    from torch.utils.data import DataLoader
    from training.registry import ComponentRegistry
    from training.distillation.cross_distill import cross_distill

    mlflow.set_tracking_uri(mlflow_uri())
    mlflow.set_experiment("distillation/cross")

    registry = ComponentRegistry()
    combos = registry.get_enabled_combinations()
    if len(combos) < 2:
        logger.info("Fewer than 2 architectures — skipping cross-distillation.")
        return

    datasets, _, _, n_users, n_items = load_datasets()
    loader = DataLoader(datasets["train"], batch_size=512, shuffle=True, num_workers=2)

    arch_models: dict[str, Any] = {}
    for arch_entry, _ in combos:
        if arch_entry.name not in arch_models:
            arch_models[arch_entry.name] = arch_entry.cls(n_users=n_users, n_items=n_items)

    for student_name in arch_models:
        for teacher_name in arch_models:
            if student_name == teacher_name:
                continue
            logger.info("Cross-distilling: %s → %s", teacher_name, student_name)
            cross_distill(
                student=arch_models[student_name],
                teacher_model=arch_models[teacher_name],
                dataloader=loader,
                student_name=student_name,
                teacher_name=teacher_name,
            )


def run_evaluate(**context) -> dict:
    """Evaluate active models on test split; log metrics to MLflow."""
    import mlflow
    import torch
    from torch.utils.data import DataLoader
    from training.device_utils import get_device, move_batch

    conf = getattr(context.get("dag_run"), "conf", {}) or {}
    mlflow.set_tracking_uri(mlflow_uri())

    combos = get_active_combinations(conf)
    datasets, _, _, n_users, n_items = load_datasets()
    test_loader = DataLoader(datasets["test"], batch_size=512, num_workers=2)
    device = get_device()

    results: dict[str, float] = {}
    with mlflow.start_run(run_name="evaluate", nested=False):
        mlflow.set_tag("stage", "evaluate")
        for arch_entry, loss_entry in combos:
            arch    = arch_entry.cls(n_users=n_users, n_items=n_items).to(device)
            loss_fn = loss_entry.cls().to(device)
            arch.eval()
            total = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    batch = move_batch(batch, device)
                    u = arch.encode_user(batch)
                    i = arch.encode_item(batch)
                    total += loss_fn(u, i, batch["rating"], batch["weight"]).item()
            avg = total / max(len(test_loader), 1)
            key = f"{arch_entry.name}_{loss_entry.name}"
            results[key] = avg
            mlflow.log_metric(f"test_loss_{key}", avg)
            logger.info("Test loss [%s]: %.6f", key, avg)
    return results


def run_precompute(**context) -> dict:
    """Pre-compute Top-N per user×genre and cold-start for active models."""
    from training.device_utils import get_device
    from precompute.recommend import precompute_recommendations

    conf = getattr(context.get("dag_run"), "conf", {}) or {}
    combos = get_active_combinations(conf)
    device = get_device()
    _, _, _, n_users, n_items = load_datasets()

    all_counts: dict[str, int] = {}
    for arch_entry, loss_entry in combos:
        arch = arch_entry.cls(n_users=n_users, n_items=n_items).to(device)
        model_name = f"{arch_entry.name}_{loss_entry.name}"
        counts = precompute_recommendations(arch, model_name)
        all_counts[model_name] = counts.get("top_n_user_genre", 0)
        logger.info("Precomputed %s: %s", model_name, counts)
    return all_counts


def run_visualize(**context) -> None:
    logger.info(
        "Visualize stage complete. "
        "Metrics available via GET /viz/runs, /viz/pipeline_sizes, /viz/hparam_sweep."
    )


def run_finalize(**context) -> None:
    conf = getattr(context.get("dag_run"), "conf", {}) or {}
    trigger_id = conf.get("trigger_id")
    update_trigger_log(trigger_id, "success")
    logger.info(
        "Pipeline run complete. dag_run_id=%s trigger_id=%s",
        getattr(context.get("dag_run"), "run_id", "local"),
        trigger_id,
    )


def check_new_data_for_daily() -> bool:
    """Used by the daily DAG's ShortCircuitOperator.

    Returns True if any raw table has new rows → run ingest.
    Returns False → skip ingest, continue from featurize.
    """
    has_new, table = check_watermark(RAW_TABLES)
    if has_new:
        logger.info("Daily run: new data in %s — will ingest.", table)
    else:
        logger.info("Daily run: no new data — skipping ingest.")
    return has_new
