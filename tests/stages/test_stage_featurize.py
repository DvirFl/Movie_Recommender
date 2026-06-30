"""Tests: stage_featurize."""
import pytest
from unittest.mock import patch
from stages.stage_featurize import run, FeaturizeResult


def test_returns_featurize_result():
    with patch("etl.featurize.featurize_all", return_value={"users": 10, "items": 50}):
        result = run()
    assert isinstance(result, FeaturizeResult)
    assert result.n_users == 10
    assert result.n_items == 50


def test_passes_decay_lambda():
    with patch("etl.featurize.featurize_all", return_value={"users": 5, "items": 20}) as mock_fn:
        run(decay_lambda=0.5)
        mock_fn.assert_called_once_with(decay_lambda=0.5)


def test_default_decay_lambda():
    with patch("etl.featurize.featurize_all", return_value={"users": 1, "items": 1}) as mock_fn:
        run()
        _, kwargs = mock_fn.call_args
        assert kwargs.get("decay_lambda", 0.1) == 0.1
