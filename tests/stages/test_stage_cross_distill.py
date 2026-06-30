"""Tests: stage_cross_distill."""
import pytest
from unittest.mock import MagicMock, patch
from stages.stage_cross_distill import run, CrossDistillResult


def _make_combos(n=2):
    combos = []
    names = ["TwoTower", "InfoNCEEncoder"][:n]
    for name in names:
        ae = MagicMock(); ae.name = name; ae.cls.return_value = MagicMock()
        le = MagicMock(); le.name = f"{name}Loss"
        combos.append((ae, le))
    return combos


def test_skips_when_fewer_than_two_archs():
    with patch("training.registry.ComponentRegistry") as mock_reg_cls, \
         patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_experiment"):
        reg = MagicMock()
        reg.get_enabled_combinations.return_value = _make_combos(n=1)
        mock_reg_cls.return_value = reg
        result = run()
    assert result.skipped is True
    assert result.pairs_run == []


def test_runs_cross_pairs_for_two_archs():
    combos = _make_combos(n=2)
    with patch("training.registry.ComponentRegistry") as mock_reg_cls, \
         patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_experiment"), \
         patch("etl.utils.load_split_dataframes",
               return_value={"train": MagicMock()}), \
         patch("etl.utils.load_user_features", return_value={1: {}}), \
         patch("etl.utils.load_item_features", return_value={10: {}}), \
         patch("training.dataset.MovieLensDataset"), \
         patch("torch.utils.data.DataLoader"), \
         patch("training.distillation.cross_distill.cross_distill") as mock_cd, \
         patch("training.device_utils.get_device", return_value=__import__("torch").device("cpu")):
        reg = MagicMock()
        reg.get_enabled_combinations.return_value = combos
        mock_reg_cls.return_value = reg
        result = run()
    assert result.skipped is False
    # 2 archs → 2 ordered pairs (A→B and B→A)
    assert len(result.pairs_run) == 2
    assert mock_cd.call_count == 2


def test_result_contains_correct_pair_names():
    combos = _make_combos(n=2)
    with patch("training.registry.ComponentRegistry") as mock_reg_cls, \
         patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_experiment"), \
         patch("etl.utils.load_split_dataframes",
               return_value={"train": MagicMock()}), \
         patch("etl.utils.load_user_features", return_value={1: {}}), \
         patch("etl.utils.load_item_features", return_value={10: {}}), \
         patch("training.dataset.MovieLensDataset"), \
         patch("torch.utils.data.DataLoader"), \
         patch("training.distillation.cross_distill.cross_distill"), \
         patch("training.device_utils.get_device", return_value=__import__("torch").device("cpu")):
        reg = MagicMock()
        reg.get_enabled_combinations.return_value = combos
        mock_reg_cls.return_value = reg
        result = run()
    pair_set = set(result.pairs_run)
    assert ("TwoTower", "InfoNCEEncoder") in pair_set
    assert ("InfoNCEEncoder", "TwoTower") in pair_set
