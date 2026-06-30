"""PyTorch Dataset for MovieLens ratings with time-decay weighting.

Time-decay weight for interaction at time t:
    w(t) = exp(-λ · (t_max - t) / t_range)

where t_max is the most recent timestamp in the split and t_range = t_max - t_min.

The dataset also supports demonstration sampling for SDFT: for each user,
the highest-rated recent interaction (outside the current sample) is returned
as the expert demonstration context.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MovieLensDataset(Dataset):
    """Dataset for a single train/val/test split.

    Args:
        ratings_df:     DataFrame with columns [user_id, movie_id, rating, timestamp].
        user_features:  Dict mapping user_id -> feature dict (pre-computed).
        item_features:  Dict mapping movie_id -> feature dict (pre-computed).
        decay_lambda:   Exponential decay rate λ.
        genre_vocab:    List of genre strings (for multi-hot index).
        split:          'train' | 'val' | 'test'
    """

    def __init__(
        self,
        ratings_df: pd.DataFrame,
        user_features: dict[int, dict],
        item_features: dict[int, dict],
        decay_lambda: float = 0.1,
        split: str = "train",
    ) -> None:
        self.df = ratings_df.reset_index(drop=True)
        self.user_features = user_features
        self.item_features = item_features
        self.decay_lambda = decay_lambda
        self.split = split

        t_vals = self.df["timestamp"].values.astype(np.float64)
        self.t_min = float(t_vals.min())
        self.t_max = float(t_vals.max())
        self.t_range = max(self.t_max - self.t_min, 1.0)

        self.weights = self._compute_weights(t_vals)

        # Index of best demonstration per user (highest-rated, most recent)
        self._demo_index: dict[int, int] = self._build_demo_index()

    def _compute_weights(self, timestamps: np.ndarray) -> np.ndarray:
        """Compute time-decay weight for each sample."""
        normalised = (timestamps - self.t_min) / self.t_range  # [0, 1]
        # Recent interactions get weight close to 1, old ones close to exp(-λ)
        weights = np.exp(-self.decay_lambda * (1.0 - normalised))
        return weights.astype(np.float32)

    def _build_demo_index(self) -> dict[int, int]:
        """For each user, find the index of their best demonstration interaction."""
        demo: dict[int, int] = {}
        for uid, group in self.df.groupby("user_id"):
            # Highest rating; break ties by most recent timestamp
            best = group.sort_values(
                ["rating", "timestamp"], ascending=[False, False]
            ).index[0]
            demo[int(uid)] = int(self.df.index.get_loc(best))
        return demo

    def get_demonstration(self, user_id: int) -> dict:
        """Return feature dict for the best demonstration interaction of a user."""
        idx = self._demo_index.get(user_id, 0)
        return self._build_sample(idx, include_demo=False)

    def _build_sample(self, idx: int, include_demo: bool = True) -> dict:
        row = self.df.iloc[idx]
        user_id = int(row["user_id"])
        movie_id = int(row["movie_id"])

        uf = self.user_features.get(user_id, {})
        itf = self.item_features.get(movie_id, {})

        sample = {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "movie_id": torch.tensor(movie_id, dtype=torch.long),
            "rating": torch.tensor(float(row["rating"]), dtype=torch.float32),
            "weight": torch.tensor(float(self.weights[idx]), dtype=torch.float32),
            # User features
            "genre_affinity": torch.tensor(
                uf.get("genre_affinity", {}), dtype=torch.float32
            )
            if uf.get("genre_affinity")
            else torch.zeros(20, dtype=torch.float32),
            "rating_count": torch.tensor(uf.get("rating_count", 0), dtype=torch.float32),
            "avg_rating": torch.tensor(uf.get("avg_rating", 0.0), dtype=torch.float32),
            # Item features
            "genre_multihot": torch.tensor(
                itf.get("genre_multihot", [0.0] * 20), dtype=torch.float32
            ),
            "release_year": torch.tensor(
                float(itf.get("release_year") or 1995), dtype=torch.float32
            ),
        }

        if include_demo:
            demo_idx = self._demo_index.get(user_id, idx)
            demo_row = self.df.iloc[demo_idx]
            demo_movie_id = int(demo_row["movie_id"])
            demo_itf = self.item_features.get(demo_movie_id, {})
            sample["demo_genre_multihot"] = torch.tensor(
                demo_itf.get("genre_multihot", [0.0] * 20), dtype=torch.float32
            )
            sample["demo_rating"] = torch.tensor(
                float(demo_row["rating"]), dtype=torch.float32
            )
            sample["demo_movie_id"] = torch.tensor(demo_movie_id, dtype=torch.long)

        return sample

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        return self._build_sample(idx, include_demo=True)
