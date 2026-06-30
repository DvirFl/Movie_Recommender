"""Stage 7 — Cross-distill: run cross-architecture SDFT between all trained models."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CrossDistillResult:
    pairs_run: list[tuple[str, str]] = field(default_factory=list)
    skipped: bool = False
    reason: str = ""


def run(
    mlflow_tracking_uri: str | None = None,
    n_epochs: int = 2,
    batch_size: int = 512,
) -> CrossDistillResult:
    """Cross-distil every ordered pair of trained architectures.

    Args:
        mlflow_tracking_uri: override MLflow URI from config.
        n_epochs:            distillation epochs per direction.
        batch_size:          DataLoader batch size.

    Returns:
        CrossDistillResult listing which (teacher, student) pairs were run.
    """
    import mlflow
    import torch
    from torch.utils.data import DataLoader
    from config import get_mlflow_config
    from training.registry import ComponentRegistry
    from training.distillation.cross_distill import cross_distill
    from etl.utils import load_split_dataframes, load_user_features, load_item_features
    from training.dataset import MovieLensDataset
    from training.device_utils import get_device

    uri = mlflow_tracking_uri or get_mlflow_config()["tracking_uri"]
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("distillation/cross")

    registry = ComponentRegistry()
    combos = registry.get_enabled_combinations()

    unique_archs = list({a.name: a for a, _ in combos}.values())
    if len(unique_archs) < 2:
        msg = "Fewer than 2 architectures registered — skipping cross-distillation."
        logger.info("[cross_distill] %s", msg)
        return CrossDistillResult(skipped=True, reason=msg)

    splits_raw = load_split_dataframes()
    uf  = load_user_features()
    itf = load_item_features()
    n_users = max(uf.keys()) + 1
    n_items = max(itf.keys()) + 1
    train_ds = MovieLensDataset(splits_raw["train"], uf, itf, split="train")
    loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    device = get_device()
    arch_models = {
        ae.name: ae.cls(n_users=n_users, n_items=n_items)
        for ae in unique_archs
    }

    result = CrossDistillResult()
    arch_names = list(arch_models.keys())

    for student_name in arch_names:
        for teacher_name in arch_names:
            if student_name == teacher_name:
                continue
            logger.info(
                "[cross_distill] %s → %s (%d epochs)", teacher_name, student_name, n_epochs
            )
            cross_distill(
                student=arch_models[student_name],
                teacher_model=arch_models[teacher_name],
                dataloader=loader,
                n_epochs=n_epochs,
                student_name=student_name,
                teacher_name=teacher_name,
            )
            result.pairs_run.append((teacher_name, student_name))

    logger.info("[cross_distill] Done — pairs: %s", result.pairs_run)
    return result
