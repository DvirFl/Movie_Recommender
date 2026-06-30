"""Tests: stage_tune."""
import pytest
from unittest.mock import MagicMock, patch
from stages.stage_tune import run, TuneResult


def _mock_registry(combos):
    reg = MagicMock()
    reg.filter_combinations.return_value = combos
    return reg


def _make_combo(arch_name="TwoTower", loss_name="TimedecayMSELoss"):
    arch = MagicMock()
    arch.name = arch_name
    arch.cls.__module__ = "training.two_tower.towers"
    arch.cls.return_value = MagicMock()

    loss = MagicMock()
    loss.name = loss_name
    loss.cls.return_value = MagicMock()
    return arch, loss


def test_returns_tune_result():
    combo = _make_combo()
    with patch("training.registry.ComponentRegistry", return_value=_mock_registry([combo])), \
         patch("etl.utils.load_split_dataframes", return_value={"train": MagicMock(), "val": MagicMock()}), \
         patch("etl.utils.load_user_features", return_value={1: {}, 2: {}}), \
         patch("etl.utils.load_item_features", return_value={10: {}, 20: {}}), \
         patch("training.dataset.MovieLensDataset"), \
         patch("training.hparam.tuner.run_sweep", return_value={"lr": 1e-3}) as mock_sweep, \
         patch("mlflow.set_tracking_uri"):
        result = run()
    assert isinstance(result, TuneResult)
    assert "TwoTower_TimedecayMSELoss" in result.best_hparams


def test_returns_empty_result_when_no_combos():
    with patch("training.registry.ComponentRegistry") as mock_reg_cls, \
         patch("mlflow.set_tracking_uri"):
        mock_reg = MagicMock()
        mock_reg.filter_combinations.return_value = []
        mock_reg_cls.return_value = mock_reg
        result = run(losses=["NonExistent"])
    assert result.best_hparams == {}


def test_filter_by_loss_name():
    combo = _make_combo()
    with patch("training.registry.ComponentRegistry") as mock_reg_cls, \
         patch("etl.utils.load_split_dataframes", return_value={"train": MagicMock(), "val": MagicMock()}), \
         patch("etl.utils.load_user_features", return_value={1: {}}), \
         patch("etl.utils.load_item_features", return_value={10: {}}), \
         patch("training.dataset.MovieLensDataset"), \
         patch("training.hparam.tuner.run_sweep", return_value={"lr": 1e-3}), \
         patch("mlflow.set_tracking_uri"):
        mock_reg = MagicMock()
        mock_reg.filter_combinations.return_value = [combo]
        mock_reg_cls.return_value = mock_reg
        run(losses=["TimedecayMSELoss"])
        mock_reg.filter_combinations.assert_called_once_with(
            architecture_names=None,
            loss_names=["TimedecayMSELoss"],
        )


def test_best_hparams_stored_per_key():
    combo = _make_combo()
    with patch("training.registry.ComponentRegistry", return_value=_mock_registry([combo])), \
         patch("etl.utils.load_split_dataframes", return_value={"train": MagicMock(), "val": MagicMock()}), \
         patch("etl.utils.load_user_features", return_value={1: {}}), \
         patch("etl.utils.load_item_features", return_value={10: {}}), \
         patch("training.dataset.MovieLensDataset"), \
         patch("training.hparam.tuner.run_sweep", return_value={"lr": 5e-4, "batch_size": 256}), \
         patch("mlflow.set_tracking_uri"):
        result = run()
    assert result.best_hparams["TwoTower_TimedecayMSELoss"]["lr"] == 5e-4
