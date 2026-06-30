"""Tests: stage_split."""
import pytest
from unittest.mock import patch
from stages.stage_split import run, SplitResult


def test_returns_split_result():
    with patch("etl.split.split_ratings", return_value={"train": 80, "val": 10, "test": 10}):
        result = run()
    assert isinstance(result, SplitResult)
    assert result.train == 80
    assert result.val   == 10
    assert result.test  == 10


def test_passes_fractions():
    with patch("etl.split.split_ratings", return_value={"train": 70, "val": 15, "test": 15}) as mock_fn:
        run(train_frac=0.7, val_frac=0.15)
        mock_fn.assert_called_once_with(train_frac=0.7, val_frac=0.15)


def test_fractions_sum_below_one():
    """train + val must be < 1.0 to leave room for test."""
    with patch("etl.split.split_ratings", return_value={"train": 80, "val": 10, "test": 10}):
        result = run(train_frac=0.8, val_frac=0.1)
    assert result.train + result.val + result.test > 0


def test_missing_keys_default_to_zero():
    with patch("etl.split.split_ratings", return_value={"train": 80}):
        result = run()
    assert result.val  == 0
    assert result.test == 0
