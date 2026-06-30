"""Search space composition: shared + architecture-specific + loss-specific."""
from __future__ import annotations

from typing import Any

from training.base.architecture import BaseRecommenderArchitecture
from training.base.loss import BaseRecommenderLoss

# Shared hyperparameters common to all arch×loss combinations
SHARED_SEARCH_SPACE: dict[str, Any] = {
    "lr": ("float", 1e-5, 1e-2, {"log": True}),
    "batch_size": ("categorical", [256, 512, 1024]),
    "n_epochs": ("int", 2, 10),
    "weight_decay": ("float", 0.0, 1e-2),
    "sdft_weight": ("float", 0.0, 0.5),
    "ema_alpha": ("float", 0.005, 0.1, {"log": True}),
    "warmup_mask_dims": ("categorical", [0, 2, 4, 8]),
}


def build_search_space(
    arch: BaseRecommenderArchitecture,
    loss: BaseRecommenderLoss,
) -> dict[str, Any]:
    """Merge shared, arch-specific, and loss-specific search spaces.

    Later entries override earlier ones (arch > shared; loss > arch).
    """
    space = dict(SHARED_SEARCH_SPACE)
    space.update(arch.hparam_search_space())
    space.update(loss.hparam_search_space())
    return space
