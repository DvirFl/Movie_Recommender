"""Stage 8 — Evaluate: test-set metrics for all active models, logged to MLflow."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EvaluateResult:
    # Maps "ArchName_LossName" -> test loss value
    test_losses: dict[str, float] = field(default_factory=dict)


def run(
    losses: list[str] | str = "all",
    architectures: list[str] | str = "all",
    batch_size: int = 512,
    mlflow_tracking_uri: str | None = None,
) -> EvaluateResult:
    """Evaluate trained models on the held-out test split.

    Args:
        losses:               filter to specific loss names, or "all".
        architectures:        filter to specific arch names, or "all".
        batch_size:           DataLoader batch size for test inference.
        mlflow_tracking_uri:  override MLflow URI from config.

    Returns:
        EvaluateResult with average test loss per combination.
    """
    import mlflow
    import torch
    from torch.utils.data import DataLoader
    from config import get_mlflow_config
    from training.registry import ComponentRegistry
    from training.device_utils import get_device, move_batch
    from etl.utils import load_split_dataframes, load_user_features, load_item_features
    from training.dataset import MovieLensDataset

    uri = mlflow_tracking_uri or get_mlflow_config()["tracking_uri"]
    mlflow.set_tracking_uri(uri)

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

    splits_raw = load_split_dataframes()
    uf  = load_user_features()
    itf = load_item_features()
    n_users = max(uf.keys()) + 1
    n_items = max(itf.keys()) + 1
    test_ds  = MovieLensDataset(splits_raw["test"], uf, itf, split="test")
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=2)

    device = get_device()
    result = EvaluateResult()

    with mlflow.start_run(run_name="evaluate_manual"):
        mlflow.set_tag("stage", "evaluate")
        mlflow.set_tag("trigger_type", "manual")

        for arch_entry, loss_entry in combos:
            key = f"{arch_entry.name}_{loss_entry.name}"
            logger.info("[evaluate] Testing %s ...", key)

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
            result.test_losses[key] = avg
            mlflow.log_metric(f"test_loss_{key}", avg)
            logger.info("[evaluate] %s → test_loss: %.6f", key, avg)

    return result
