"""Time-decayed MSE loss for the Two-Tower model."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.base.loss import BaseRecommenderLoss


class TimedecayMSELoss(BaseRecommenderLoss):
    """Weighted MSE loss where weights are exponential time-decay scores.

    loss = mean(weights * (score(u, i) - rating)^2)

    The score between user and item embeddings is computed as dot product
    (equivalent to cosine after L2-normalisation in training).
    """

    name = "TimedecayMSELoss"

    def __init__(self, rating_scale: float = 5.0) -> None:
        super().__init__()
        self.rating_scale = rating_scale

    def forward(
        self,
        user_emb: torch.Tensor,   # (B, D)
        item_emb: torch.Tensor,   # (B, D)
        targets: torch.Tensor,    # (B,)  raw ratings
        weights: torch.Tensor,    # (B,)  time-decay weights in [0, 1]
    ) -> torch.Tensor:
        # Normalise embeddings for stable dot product
        u_norm = F.normalize(user_emb, p=2, dim=-1)
        i_norm = F.normalize(item_emb, p=2, dim=-1)

        # Predicted score in [-1, 1]; scale to [0, rating_scale]
        raw_score = (u_norm * i_norm).sum(dim=-1)                    # (B,)
        pred = (raw_score + 1.0) / 2.0 * self.rating_scale           # (B,)

        # Normalise targets to same range
        target_norm = targets / self.rating_scale                     # (B,)
        pred_norm = pred / self.rating_scale                          # (B,)

        se = (pred_norm - target_norm) ** 2                           # (B,)
        weighted = weights * se
        return weighted.mean()

    def hparam_search_space(self) -> dict[str, Any]:
        return {
            "decay_lambda": ("float", 0.01, 1.0, {"log": True}),
        }
