"""Tests: MovieLensDataset — time-decay weights, demonstration sampling."""
import pytest
import torch
import numpy as np
from training.dataset import MovieLensDataset


def test_dataset_length(tiny_ratings_df, tiny_user_features, tiny_item_features):
    ds = MovieLensDataset(tiny_ratings_df, tiny_user_features, tiny_item_features)
    assert len(ds) == 5


def test_dataset_item_keys(tiny_ratings_df, tiny_user_features, tiny_item_features):
    ds = MovieLensDataset(tiny_ratings_df, tiny_user_features, tiny_item_features)
    sample = ds[0]
    required = {"user_id", "movie_id", "rating", "weight", "genre_affinity",
                "genre_multihot", "release_year", "rating_count", "avg_rating"}
    assert required.issubset(set(sample.keys()))


def test_weights_between_zero_and_one(tiny_ratings_df, tiny_user_features, tiny_item_features):
    ds = MovieLensDataset(tiny_ratings_df, tiny_user_features, tiny_item_features, decay_lambda=0.1)
    assert np.all(ds.weights >= 0.0)
    assert np.all(ds.weights <= 1.0)


def test_recent_interactions_get_higher_weight(tiny_ratings_df, tiny_user_features, tiny_item_features):
    ds = MovieLensDataset(tiny_ratings_df, tiny_user_features, tiny_item_features, decay_lambda=0.5)
    # Row with highest timestamp should have highest weight
    max_ts_idx = tiny_ratings_df["timestamp"].idxmax()
    max_weight = ds.weights[max_ts_idx]
    assert max_weight == pytest.approx(ds.weights.max(), abs=1e-5)


def test_demonstration_keys_present(tiny_ratings_df, tiny_user_features, tiny_item_features):
    ds = MovieLensDataset(tiny_ratings_df, tiny_user_features, tiny_item_features)
    sample = ds[0]
    assert "demo_genre_multihot" in sample
    assert "demo_rating" in sample
    assert "demo_movie_id" in sample


def test_all_tensors_are_tensors(tiny_ratings_df, tiny_user_features, tiny_item_features):
    ds = MovieLensDataset(tiny_ratings_df, tiny_user_features, tiny_item_features)
    sample = ds[0]
    for k, v in sample.items():
        assert isinstance(v, torch.Tensor), f"Expected tensor for key '{k}', got {type(v)}"


def test_demo_index_built_for_all_users(tiny_ratings_df, tiny_user_features, tiny_item_features):
    ds = MovieLensDataset(tiny_ratings_df, tiny_user_features, tiny_item_features)
    for uid in tiny_ratings_df["user_id"].unique():
        assert uid in ds._demo_index
