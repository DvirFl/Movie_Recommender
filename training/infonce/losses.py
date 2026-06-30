"""Time-decayed InfoNCE (contrastive) loss."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from training.base.loss import BaseRecommenderLoss


class TimedecayInfoNCELoss(BaseRecommenderLoss):
    """Temperature-scaled InfoNCE with time-decay weighting on positive pair strength.

    Positive pairs: (user_emb[i], item_emb[i]) — in-batch.
    Negatives: all other item embeddings in the batch.

    The time-decay weight scales the positive logit:
        logit_pos[i] = weight[i] * sim(u[i], i[i]) / τ

    This encourages the model to push recent positives closer in embedding space.
    """

    name = "TimedecayInfoNCELoss"

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        user_emb: torch.Tensor,   # (B, D)
        item_emb: torch.Tensor,   # (B, D)
        targets: torch.Tensor,    # (B,)  ratings (used for pos/neg threshold)
        weights: torch.Tensor,    # (B,)  time-decay weights
    ) -> torch.Tensor:
        B = user_emb.size(0)

        u_norm = F.normalize(user_emb, p=2, dim=-1)   # (B, D)
        i_norm = F.normalize(item_emb, p=2, dim=-1)   # (B, D)

        # Similarity matrix (B, B)
        sim = torch.matmul(u_norm, i_norm.T) / self.temperature

        # Scale diagonal (positive pairs) by time-decay weight
        diag_scale = weights.detach()                  # (B,)
        sim_weighted = sim.clone()
        sim_weighted[range(B), range(B)] = sim[range(B), range(B)] * diag_scale

        # Standard InfoNCE: cross-entropy with diagonal as target
        labels = torch.arange(B, device=user_emb.device)
        loss_u = F.cross_entropy(sim_weighted, labels)
        loss_i = F.cross_entropy(sim_weighted.T, labels)
        return (loss_u + loss_i) / 2.0

    def hparam_search_space(self) -> dict[str, Any]:
        return {
            "temperature": ("float", 0.01, 1.0, {"log": True}),
            "decay_lambda": ("float", 0.01, 1.0, {"log": True}),
            "rating_threshold": ("float", 3.0, 5.0),
        }
