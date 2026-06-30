"""Two-Tower model: UserTower + ItemTower projecting to shared embedding space."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from training.base.architecture import BaseRecommenderArchitecture


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class UserTower(nn.Module):
    """MLP over [user_id_emb + genre_affinity + rating_count + avg_rating + decayed_history_emb]."""

    def __init__(
        self,
        n_users: int,
        n_genres: int,
        embed_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        history_dim: int = 32,
    ) -> None:
        super().__init__()
        self.user_embed = nn.Embedding(n_users + 1, embed_dim, padding_idx=0)
        self.history_proj = nn.Linear(n_genres, history_dim)
        # input: user_emb + genre_affinity + history_proj + rating_count + avg_rating
        input_dim = embed_dim + n_genres + history_dim + 2
        self.mlp = _build_mlp(input_dim, hidden_dims, output_dim)

    def forward(self, user_features: dict[str, torch.Tensor]) -> torch.Tensor:
        u_emb = self.user_embed(user_features["user_id"])            # (B, E)
        ga = user_features["genre_affinity"]                          # (B, G)
        hist = torch.relu(self.history_proj(ga))                      # (B, H)
        rc = user_features["rating_count"].unsqueeze(1)               # (B, 1)
        ar = user_features["avg_rating"].unsqueeze(1)                 # (B, 1)
        x = torch.cat([u_emb, ga, hist, rc, ar], dim=-1)
        return self.mlp(x)                                            # (B, D)


class ItemTower(nn.Module):
    """MLP over [item_id_emb + genre_multihot + release_year]."""

    def __init__(
        self,
        n_items: int,
        n_genres: int,
        embed_dim: int,
        hidden_dims: list[int],
        output_dim: int,
    ) -> None:
        super().__init__()
        self.item_embed = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        input_dim = embed_dim + n_genres + 1  # +1 for release_year
        self.mlp = _build_mlp(input_dim, hidden_dims, output_dim)

    def forward(self, item_features: dict[str, torch.Tensor]) -> torch.Tensor:
        i_emb = self.item_embed(item_features["movie_id"])            # (B, E)
        gm = item_features["genre_multihot"]                          # (B, G)
        ry = item_features["release_year"].unsqueeze(1)               # (B, 1)
        x = torch.cat([i_emb, gm, ry], dim=-1)
        return self.mlp(x)                                            # (B, D)


class TwoTowerModel(BaseRecommenderArchitecture):
    """Full Two-Tower model implementing BaseRecommenderArchitecture."""

    name = "TwoTower"
    compatible_losses = ["TimedecayMSELoss"]

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_genres: int,
        embed_dim: int = 64,
        hidden_dims: list[int] | None = None,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        self.user_tower = UserTower(n_users, n_genres, embed_dim, hidden_dims, output_dim)
        self.item_tower = ItemTower(n_items, n_genres, embed_dim, hidden_dims, output_dim)
        self.output_dim = output_dim

    def encode_user(self, user_features: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.user_tower(user_features)

    def encode_item(self, item_features: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.item_tower(item_features)

    def supports_sdft(self) -> bool:
        return True

    def get_demonstration_context(
        self, user_features: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return user_features augmented with demonstration movie's genre as expert context."""
        ctx = dict(user_features)
        # Blend the demonstration genre multihot into genre_affinity as teacher signal
        if "demo_genre_multihot" in user_features:
            ctx["genre_affinity"] = (
                user_features["genre_affinity"] * 0.7
                + user_features["demo_genre_multihot"] * 0.3
            )
        return ctx

    def hparam_search_space(self) -> dict[str, Any]:
        return {
            "embed_dim": ("categorical", [32, 64, 128]),
            "hidden_dim_0": ("categorical", [128, 256, 512]),
            "hidden_dim_1": ("categorical", [64, 128, 256]),
            "output_dim": ("categorical", [64, 128, 256]),
        }
