"""Tests: stage_precompute."""
import pytest
from unittest.mock import MagicMock, patch
import torch
from stages.stage_precompute import run, PrecomputeResult


def _make_combo():
    ae = MagicMock(); ae.name = "TwoTower"
    ae.cls.return_value = MagicMock()
    ae.cls.return_value.to.return_value = ae.cls.return_value
    le = MagicMock(); le.name = "TimedecayMSELoss"
    return ae, le


def test_returns_precompute_result():
    combo = _make_combo()
    mock_reg = MagicMock()
    mock_reg.filter_combinations.return_value = [combo]
    with patch("training.registry.ComponentRegistry", return_value=mock_reg), \
         patch("etl.utils.load_user_features", return_value={1: {}}), \
         patch("etl.utils.load_item_features", return_value={10: {}}), \
         patch("training.device_utils.get_device", return_value=torch.device("cpu")), \
         patch("precompute.recommend.precompute_recommendations",
               return_value={"top_n_user_genre": 100, "cold_start_genre": 20}):
        result = run()
    assert isinstance(result, PrecomputeResult)
    assert "TwoTower_TimedecayMSELoss" in result.counts


def test_counts_stored_correctly():
    combo = _make_combo()
    mock_reg = MagicMock()
    mock_reg.filter_combinations.return_value = [combo]
    with patch("training.registry.ComponentRegistry", return_value=mock_reg), \
         patch("etl.utils.load_user_features", return_value={1: {}}), \
         patch("etl.utils.load_item_features", return_value={10: {}}), \
         patch("training.device_utils.get_device", return_value=torch.device("cpu")), \
         patch("precompute.recommend.precompute_recommendations",
               return_value={"top_n_user_genre": 500, "cold_start_genre": 40}):
        result = run()
    assert result.counts["TwoTower_TimedecayMSELoss"]["top_n_user_genre"] == 500
    assert result.counts["TwoTower_TimedecayMSELoss"]["cold_start_genre"] == 40


def test_top_n_passed_through():
    combo = _make_combo()
    mock_reg = MagicMock()
    mock_reg.filter_combinations.return_value = [combo]
    with patch("training.registry.ComponentRegistry", return_value=mock_reg), \
         patch("etl.utils.load_user_features", return_value={1: {}}), \
         patch("etl.utils.load_item_features", return_value={10: {}}), \
         patch("training.device_utils.get_device", return_value=torch.device("cpu")), \
         patch("precompute.recommend.precompute_recommendations",
               return_value={}) as mock_pc:
        run(top_n=50)
    call_kwargs = mock_pc.call_args[1]
    assert call_kwargs["top_n"] == 50


def test_empty_when_no_combos():
    mock_reg = MagicMock()
    mock_reg.filter_combinations.return_value = []
    with patch("training.registry.ComponentRegistry", return_value=mock_reg), \
         patch("etl.utils.load_user_features", return_value={1: {}}), \
         patch("etl.utils.load_item_features", return_value={10: {}}), \
         patch("training.device_utils.get_device", return_value=torch.device("cpu")):
        result = run(losses=["NonExistent"])
    assert result.counts == {}
