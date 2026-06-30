"""Abstract base class for all recommender architectures."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseRecommenderArchitecture(ABC, nn.Module):
    """Every architecture must subclass this and register in config/registry.yaml.

    Contract:
      - `name`               unique registry key.
      - `compatible_losses`  list of loss names this arch can be trained with.
      - `encode_user`        returns a (B, D) user embedding tensor.
      - `encode_item`        returns a (B, D) item embedding tensor.
      - `supports_sdft`      whether SDFT self-distillation can be applied.
      - `get_demonstration_context`
                             returns a teacher-conditioned input dict for SDFT;
                             only called when supports_sdft() is True.
      - `hparam_search_space` Optuna search space for arch-specific hparams.
    """

    #: Unique registry key — must match registry.yaml architecture name.
    name: str

    #: Loss names this architecture is compatible with.
    compatible_losses: list[str]

    @abstractmethod
    def encode_user(self, user_features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return (B, D) user embedding."""
        ...

    @abstractmethod
    def encode_item(self, item_features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return (B, D) item embedding."""
        ...

    @abstractmethod
    def hparam_search_space(self) -> dict[str, Any]:
        """Return Optuna search space for architecture-specific hyperparameters.

        Same format as BaseRecommenderLoss.hparam_search_space.
        """
        ...

    @abstractmethod
    def supports_sdft(self) -> bool:
        """Return True if this architecture supports SDFT self-distillation."""
        ...

    @abstractmethod
    def get_demonstration_context(
        self,
        user_features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return teacher-conditioned input for SDFT forward pass.

        Typically prepends a high-rated historical interaction to the user
        feature dict to serve as the expert demonstration context.
        Only called when supports_sdft() is True.
        """
        ...

    def forward(
        self,
        user_features: dict[str, torch.Tensor],
        item_features: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience: returns (user_emb, item_emb)."""
        return self.encode_user(user_features), self.encode_item(item_features)
