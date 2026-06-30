"""Abstract base class for all recommender loss functions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseRecommenderLoss(ABC, nn.Module):
    """Every loss function must subclass this and register in config/registry.yaml.

    Contract:
      - `name`  must be a unique string matching the registry key.
      - `forward` receives pre-computed user/item embeddings + targets + time-decay weights.
      - `hparam_search_space` returns an Optuna-compatible dict of {param: (type, *args)}.
    """

    #: Unique registry key — must match registry.yaml loss name.
    name: str

    @abstractmethod
    def forward(
        self,
        user_emb: torch.Tensor,   # (B, D)
        item_emb: torch.Tensor,   # (B, D)
        targets: torch.Tensor,    # (B,)  ratings or binary labels
        weights: torch.Tensor,    # (B,)  time-decay weights in [0, 1]
    ) -> torch.Tensor:
        """Return scalar loss."""
        ...

    @abstractmethod
    def hparam_search_space(self) -> dict[str, Any]:
        """Return Optuna search space for loss-specific hyperparameters.

        Format:
          {
            "param_name": ("float", low, high, {"log": True}),
            "param_name": ("int", low, high),
            "param_name": ("categorical", [choices]),
          }
        """
        ...
