"""Stage 6 — Train: train every requested arch×loss with SDFT and log to MLflow."""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    # Maps "ArchName_LossName" -> MLflow run_id
    run_ids: dict[str, str] = field(default_factory=dict)


def run(
    best_hparams: dict[str, dict] | None = None,
    losses: list[str] | str = "all",
    architectures: list[str] | str = "all",
    default_hparams: dict | None = None,
    mlflow_tracking_uri: str | None = None,
    save_to_minio: bool = True,
    trigger_type: str = "manual",
) -> TrainResult:
    """Train all requested arch×loss combinations.

    Args:
        best_hparams:         output of stage_tune.run() — maps combo key to hparams.
                              Falls back to *default_hparams* or built-in defaults.
        losses:               filter to specific loss names, or "all".
        architectures:        filter to specific arch names, or "all".
        default_hparams:      hparams used when a combo has no entry in best_hparams.
        mlflow_tracking_uri:  override MLflow URI from config.
        save_to_minio:        push checkpoints to MinIO after each epoch.
        trigger_type:         tag logged to MLflow ("manual", "schedule", "on_demand").

    Returns:
        TrainResult mapping each combination key to its MLflow run_id.
    """
    import mlflow
    from config import get_mlflow_config
    from training.registry import ComponentRegistry
    from etl.utils import load_split_dataframes, load_user_features, load_item_features
    from training.dataset import MovieLensDataset

    uri = mlflow_tracking_uri or get_mlflow_config()["tracking_uri"]
    mlflow.set_tracking_uri(uri)

    _defaults = default_hparams or {
        "lr": 1e-3, "n_epochs": 5, "batch_size": 512,
        "sdft_weight": 0.1, "ema_alpha": 0.02,
        "weight_decay": 0.0, "warmup_mask_dims": 4,
    }
    hparams_map = best_hparams or {}

    registry = ComponentRegistry()
    arch_filter = None if architectures == "all" else (
        architectures if isinstance(architectures, list) else [architectures]
    )
    loss_filter = None if losses == "all" else (
        losses if isinstance(losses, list) else [losses]
    )
    combos = registry.filter_combinations(
        architecture_names=arch_filter,
        loss_names=loss_filter,
    )

    if not combos:
        logger.warning("[train] No enabled combinations match filters.")
        return TrainResult()

    splits_raw = load_split_dataframes()
    uf  = load_user_features()
    itf = load_item_features()
    n_users = max(uf.keys()) + 1
    n_items = max(itf.keys()) + 1
    train_ds = MovieLensDataset(splits_raw["train"], uf, itf, split="train")
    val_ds   = MovieLensDataset(splits_raw["val"],   uf, itf, split="val")

    result = TrainResult()

    for arch_entry, loss_entry in combos:
        key = f"{arch_entry.name}_{loss_entry.name}"
        hparams = hparams_map.get(key, _defaults)
        logger.info("[train] Training %s  hparams=%s", key, hparams)

        arch     = arch_entry.cls(n_users=n_users, n_items=n_items)
        loss_fn  = loss_entry.cls()

        parts = arch_entry.cls.__module__.split(".")
        train_fn = importlib.import_module(".".join(parts[:-1]) + ".trainer").train

        run_id = train_fn(
            arch, loss_fn, train_ds, val_ds,
            hparams=hparams,
            run_name=f"{key}_manual",
            experiment_name=f"train/{key}",
            mlflow_tags={"trigger_type": trigger_type},
            save_to_minio=save_to_minio,
        )
        result.run_ids[key] = run_id
        logger.info("[train] %s → run_id: %s", key, run_id)

    return result
