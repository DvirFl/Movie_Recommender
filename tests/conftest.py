"""Shared pytest fixtures.

All DB tests use the live test PostgreSQL database (RECSYS_ENV=test).
No DB layer mocking — tests run against the real schema.
"""
from __future__ import annotations

import os
import pytest
import pandas as pd
import torch

os.environ.setdefault("RECSYS_ENV", "test")


# ---------------------------------------------------------------------------
# Tiny synthetic fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_ratings_df() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3],
        "movie_id": [10, 20, 10, 30, 20],
        "rating": [4.0, 3.5, 5.0, 2.0, 4.5],
        "timestamp": [1000, 2000, 1500, 2500, 3000],
    })


@pytest.fixture
def tiny_user_features() -> dict:
    return {
        1: {"decayed_history": {10: 0.8, 20: 0.5}, "genre_affinity": {"Action": 0.6, "Drama": 0.4},
            "rating_count": 2, "avg_rating": 3.75},
        2: {"decayed_history": {10: 0.9, 30: 0.3}, "genre_affinity": {"Comedy": 0.8},
            "rating_count": 2, "avg_rating": 3.5},
        3: {"decayed_history": {20: 1.0}, "genre_affinity": {"Drama": 1.0},
            "rating_count": 1, "avg_rating": 4.5},
    }


@pytest.fixture
def tiny_item_features() -> dict:
    return {
        10: {"genre_multihot": [1.0] + [0.0] * 19, "tag_tfidf": {"action": 0.5}, "release_year": 2010},
        20: {"genre_multihot": [0.0, 1.0] + [0.0] * 18, "tag_tfidf": {"drama": 0.7}, "release_year": 2015},
        30: {"genre_multihot": [0.0, 0.0, 1.0] + [0.0] * 17, "tag_tfidf": {}, "release_year": 2020},
    }


@pytest.fixture
def small_model_kwargs() -> dict:
    return {"n_users": 10, "n_items": 50, "n_genres": 20, "embed_dim": 16,
            "hidden_dims": [32], "output_dim": 32}


@pytest.fixture
def tiny_batch() -> dict[str, torch.Tensor]:
    B = 4
    return {
        "user_id": torch.randint(1, 10, (B,)),
        "movie_id": torch.randint(1, 50, (B,)),
        "rating": torch.rand(B) * 4.5 + 0.5,
        "weight": torch.rand(B),
        "genre_affinity": torch.rand(B, 20),
        "genre_multihot": torch.rand(B, 20),
        "release_year": torch.full((B,), 2010.0),
        "rating_count": torch.rand(B) * 100,
        "avg_rating": torch.rand(B) * 4.5 + 0.5,
        "demo_genre_multihot": torch.rand(B, 20),
        "demo_rating": torch.rand(B) * 5.0,
        "demo_movie_id": torch.randint(1, 50, (B,)),
    }
