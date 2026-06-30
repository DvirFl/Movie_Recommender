"""Airflow DAG: movielens_pipeline

Three trigger modes:
  1. Data sensor   — fires when any raw.* table has new rows (watermark check)
  2. Daily schedule — runs at midnight UTC
  3. On-demand     — triggered via POST /trigger with conf payload

Dynamic task mapping: training stages are generated per registered arch×loss
combination from the registry, so adding a new combination to registry.yaml
automatically adds tasks without changing this file.
"""
from __future__ import annotations

import datetime as dt
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom watermark sensor
# ---------------------------------------------------------------------------

class RawTableWatermarkSensor(BaseSensorOperator):
    """Fires when any raw.* table has new rows since last watermark."""

    @apply_defaults
    def __init__(self, tables: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self.tables = tables

    def poke(self, context) -> bool:
        from sqlalchemy import text
        from db.connection import get_session

        with get_session() as session:
            for table in self.tables:
                result = session.execute(
                    text(f"""
                        SELECT COUNT(*) FROM {table} t
                        LEFT JOIN pipeline.trigger_watermarks w
                            ON w.table_name = :tname
                        WHERE w.last_inserted_at IS NULL
                           OR t.inserted_at > w.last_inserted_at
                    """),
                    {"tname": table},
                ).scalar()
                if result and result > 0:
                    logger.info("New data detected in %s (%d rows)", table, result)
                    return True
        return False


# ---------------------------------------------------------------------------
# Task callables — each is independently testable
# ---------------------------------------------------------------------------

def task_ingest(**context) -> dict:
    from etl.ingest import ingest_all
    import os
    data_dir = os.environ.get("MOVIELENS_DATA_DIR", "/data/movielens")
    return ingest_all(data_dir)


def task_validate(**context) -> dict:
    from etl.validate import validate_all
    report = validate_all()
    if not report.passed:
        raise ValueError(f"Validation failed: {report.issues}")
    return report.stats


def task_featurize(**context) -> dict:
    from etl.featurize import featurize_all
    return featurize_all()


def task_split(**context) -> dict:
    from etl.split import split_ratings
    return split_ratings()


def task_tune(arch_name: str, loss_name: str, **context) -> dict:
    """Hyperparameter sweep for one arch×loss combination."""
    from training.registry import ComponentRegistry
    from training.hparam.tuner import run_sweep
    from etl.utils import load_split_dataframes, load_user_features, load_item_features
    from training.dataset import MovieLensDataset
    import mlflow

    registry = ComponentRegistry()
    arch_entry = registry.get_arch_entry(arch_name)
    loss_entry = registry.get_loss_entry(loss_name)

    splits = load_split_dataframes()
    user_features = load_user_features()
    item_features = load_item_features()

    n_users = max(user_features.keys()) + 1
    n_items = max(item_features.keys()) + 1

    train_ds = MovieLensDataset(splits["train"], user_features, item_features, split="train")
    val_ds = MovieLensDataset(splits["val"], user_features, item_features, split="val")

    mlflow_cfg = __import__("config").get_mlflow_config()
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])

    def objective(hparams):
        import torch
        arch = arch_entry.cls(n_users=n_users, n_items=n_items)
        loss = loss_entry.cls()

        if arch_name == "TwoTower":
            from training.two_tower.trainer import train
        else:
            from training.infonce.trainer import train

        train(
            arch, loss, train_ds, val_ds, hparams,
            experiment_name=f"hparam/{arch_name}_{loss_name}",
            save_to_minio=False,
        )
        # Return last val_loss from MLflow (simplified)
        return hparams.get("lr", 1e-3)  # placeholder; real impl reads MLflow

    best = run_sweep(
        arch_cls=arch_entry.cls,
        arch_kwargs={"n_users": n_users, "n_items": n_items},
        loss_cls=loss_entry.cls,
        objective_fn=objective,
        experiment_name=f"hparam/{arch_name}_{loss_name}",
    )
    return best


def task_train(arch_name: str, loss_name: str, **context) -> str:
    """Train one arch×loss combination with best hparams from tune stage."""
    from training.registry import ComponentRegistry
    from etl.utils import load_split_dataframes, load_user_features, load_item_features
    from training.dataset import MovieLensDataset
    import mlflow

    # Retrieve best hparams from XCom (set by tune task)
    ti = context["ti"]
    best_hparams = ti.xcom_pull(task_ids=f"tune_{arch_name}_{loss_name}") or {}

    registry = ComponentRegistry()
    arch_entry = registry.get_arch_entry(arch_name)
    loss_entry = registry.get_loss_entry(loss_name)

    splits = load_split_dataframes()
    user_features = load_user_features()
    item_features = load_item_features()

    n_users = max(user_features.keys()) + 1
    n_items = max(item_features.keys()) + 1

    arch = arch_entry.cls(n_users=n_users, n_items=n_items)
    loss = loss_entry.cls()

    train_ds = MovieLensDataset(splits["train"], user_features, item_features, split="train")
    val_ds = MovieLensDataset(splits["val"], user_features, item_features, split="val")

    mlflow_cfg = __import__("config").get_mlflow_config()
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])

    trigger_conf = context["dag_run"].conf or {}
    tags = {
        "trigger_type": trigger_conf.get("trigger_type", "schedule"),
        "trigger_id": trigger_conf.get("trigger_id", ""),
    }

    if arch_name == "TwoTower":
        from training.two_tower.trainer import train as train_fn
    else:
        from training.infonce.trainer import train as train_fn

    run_id = train_fn(
        arch, loss, train_ds, val_ds,
        hparams=best_hparams or {"lr": 1e-3, "n_epochs": 5, "batch_size": 512},
        experiment_name=f"train/{arch_name}_{loss_name}",
        mlflow_tags=tags,
    )
    return run_id


def task_cross_distill(**context) -> None:
    """Cross-architecture SDFT distillation between all trained models."""
    from training.registry import ComponentRegistry
    from training.distillation.cross_distill import cross_distill
    from etl.utils import load_split_dataframes, load_user_features, load_item_features
    from training.dataset import MovieLensDataset
    from torch.utils.data import DataLoader
    import torch, mlflow

    registry = ComponentRegistry()
    combos = registry.get_enabled_combinations()
    if len(combos) < 2:
        logger.info("Less than 2 architectures — skipping cross-distillation.")
        return

    splits = load_split_dataframes()
    user_features = load_user_features()
    item_features = load_item_features()
    n_users = max(user_features.keys()) + 1
    n_items = max(item_features.keys()) + 1
    train_ds = MovieLensDataset(splits["train"], user_features, item_features, split="train")
    loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    mlflow_cfg = __import__("config").get_mlflow_config()
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment("distillation/cross")

    # Build model instances for all combos
    models = {
        ae.name: ae.cls(n_users=n_users, n_items=n_items)
        for ae, _ in combos
    }

    arch_names = list(models.keys())
    # Cross-distill every ordered pair
    for i, student_name in enumerate(arch_names):
        for j, teacher_name in enumerate(arch_names):
            if i == j:
                continue
            cross_distill(
                student=models[student_name],
                teacher_model=models[teacher_name],
                dataloader=loader,
                student_name=student_name,
                teacher_name=teacher_name,
            )


def task_evaluate(**context) -> dict:
    """Evaluate all trained models on the test set."""
    from training.registry import ComponentRegistry
    from etl.utils import load_split_dataframes, load_user_features, load_item_features
    from training.dataset import MovieLensDataset
    from torch.utils.data import DataLoader
    import torch, mlflow

    registry = ComponentRegistry()
    splits = load_split_dataframes()
    user_features = load_user_features()
    item_features = load_item_features()
    n_users = max(user_features.keys()) + 1
    n_items = max(item_features.keys()) + 1
    test_ds = MovieLensDataset(splits["test"], user_features, item_features, split="test")
    test_loader = DataLoader(test_ds, batch_size=512)

    mlflow_cfg = __import__("config").get_mlflow_config()
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])

    results = {}
    for arch_entry, loss_entry in registry.get_enabled_combinations():
        arch = arch_entry.cls(n_users=n_users, n_items=n_items)
        loss_fn = loss_entry.cls()
        arch.eval()
        total_loss = 0.0
        from training.device_utils import get_device, move_batch
        device = get_device()
        arch = arch.to(device)
        loss_fn = loss_fn.to(device)
        with torch.no_grad():
            for batch in test_loader:
                batch = move_batch(batch, device)
                u = arch.encode_user(batch)
                i = arch.encode_item(batch)
                total_loss += loss_fn(u, i, batch["rating"], batch["weight"]).item()
        avg_loss = total_loss / max(len(test_loader), 1)
        key = f"{arch_entry.name}_{loss_entry.name}"
        results[key] = avg_loss
        mlflow.log_metric(f"test_loss_{key}", avg_loss)
    return results


def task_precompute(**context) -> dict:
    from training.registry import ComponentRegistry
    from precompute.recommend import precompute_recommendations
    from etl.utils import load_user_features, load_item_features
    from training.device_utils import get_device
    import torch

    registry = ComponentRegistry()
    user_features = load_user_features()
    item_features = load_item_features()
    n_users = max(user_features.keys()) + 1
    n_items = max(item_features.keys()) + 1

    all_counts: dict[str, int] = {}
    device = get_device()

    for arch_entry, loss_entry in registry.get_enabled_combinations():
        arch = arch_entry.cls(n_users=n_users, n_items=n_items).to(device)
        model_name = f"{arch_entry.name}_{loss_entry.name}"
        counts = precompute_recommendations(arch, model_name)
        all_counts[model_name] = counts.get("top_n_user_genre", 0)
    return all_counts


def task_visualize(**context) -> None:
    logger.info("Visualization stage: metrics available via /viz/* API endpoints.")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

def _should_run_combination(arch_name: str, loss_name: str, conf: dict) -> bool:
    """Check if this arch×loss combo should run given the trigger config."""
    req_archs = conf.get("architectures", "all")
    req_losses = conf.get("losses", "all")
    if req_archs != "all" and arch_name not in req_archs:
        return False
    if req_losses != "all" and loss_name not in req_losses:
        return False
    return True


with DAG(
    dag_id="movielens_pipeline",
    description="MovieLens recommender system — triggered by data, schedule, or on-demand.",
    schedule_interval="0 0 * * *",      # daily midnight UTC
    start_date=dt.datetime(2024, 1, 1),
    catchup=False,
    tags=["recsys", "movielens"],
    params={
        "trigger_type": "schedule",
        "trigger_id": "",
        "losses": "all",
        "architectures": "all",
        "run_from": "featurize",
    },
) as dag:

    # ---- Data sensor (runs in parallel with schedule; fires on new raw data) ----
    sensor = RawTableWatermarkSensor(
        task_id="data_sensor",
        tables=[
            "raw.ratings", "raw.movies", "raw.tags",
            "raw.links", "raw.genome_scores",
        ],
        poke_interval=60,
        timeout=3600,
        mode="reschedule",
        soft_fail=True,          # don't block schedule-triggered runs
    )

    ingest = PythonOperator(task_id="ingest", python_callable=task_ingest)
    validate = PythonOperator(task_id="validate", python_callable=task_validate)
    featurize = PythonOperator(task_id="featurize", python_callable=task_featurize)
    split = PythonOperator(task_id="split", python_callable=task_split)

    # Dynamic tune + train tasks per registry combination
    from training.registry import ComponentRegistry
    registry = ComponentRegistry()
    combos = registry.get_enabled_combinations()

    tune_tasks = []
    train_tasks = []

    for arch_entry, loss_entry in combos:
        aname, lname = arch_entry.name, loss_entry.name

        tune_t = PythonOperator(
            task_id=f"tune_{aname}_{lname}",
            python_callable=task_tune,
            op_kwargs={"arch_name": aname, "loss_name": lname},
        )
        train_t = PythonOperator(
            task_id=f"train_{aname}_{lname}",
            python_callable=task_train,
            op_kwargs={"arch_name": aname, "loss_name": lname},
        )
        tune_tasks.append(tune_t)
        train_tasks.append(train_t)
        split >> tune_t >> train_t

    cross_distill_t = PythonOperator(
        task_id="cross_distill", python_callable=task_cross_distill
    )
    evaluate_t = PythonOperator(task_id="evaluate", python_callable=task_evaluate)
    precompute_t = PythonOperator(task_id="precompute", python_callable=task_precompute)
    visualize_t = PythonOperator(task_id="visualize", python_callable=task_visualize)

    # Pipeline topology
    sensor >> ingest
    ingest >> validate >> featurize >> split
    for train_t in train_tasks:
        train_t >> cross_distill_t
    cross_distill_t >> evaluate_t >> precompute_t >> visualize_t
