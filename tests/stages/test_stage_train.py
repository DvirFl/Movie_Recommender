"""Tests: stage_train."""
import pytest
from unittest.mock import MagicMock, patch
from stages.stage_train import run, TrainResult


def _make_combo(arch_name="TwoTower", loss_name="TimedecayMSELoss"):
    arch = MagicMock()
    arch.name = arch_name
    arch.cls.__module__ = "training.two_tower.towers"
    arch.cls.return_value = MagicMock()
    loss = MagicMock()
    loss.name = loss_name
    loss.cls.return_value = MagicMock()
    return arch, loss


def _patch_train(combos, run_id="run-abc"):
    mock_reg = MagicMock()
    mock_reg.filter_combinations.return_value = combos
    mock_trainer = MagicMock(return_value=run_id)
    return mock_reg, mock_trainer


def test_returns_train_result():
    combo = _make_combo()
    mock_reg, mock_trainer = _patch_train([combo])
    with patch("training.registry.ComponentRegistry", return_value=mock_reg), \
         patch("etl.utils.load_split_dataframes", return_value={"train": MagicMock(), "val": MagicMock()}), \
         patch("etl.utils.load_user_features", return_value={1: {}}), \
         patch("etl.utils.load_item_features", return_value={10: {}}), \
         patch("training.dataset.MovieLensDataset"), \
         patch("importlib.import_module") as mock_import, \
         patch("mlflow.set_tracking_uri"):
        mock_import.return_value.train = mock_trainer
        result = run()
    assert isinstance(result, TrainResult)
    assert "TwoTower_TimedecayMSELoss" in result.run_ids


def test_uses_best_hparams_from_tune():
    combo = _make_combo()
    mock_reg, mock_trainer = _patch_train([combo])
    best = {"TwoTower_TimedecayMSELoss": {"lr": 5e-4, "n_epochs": 3}}
    with patch("training.registry.ComponentRegistry", return_value=mock_reg), \
         patch("etl.utils.load_split_dataframes", return_value={"train": MagicMock(), "val": MagicMock()}), \
         patch("etl.utils.load_user_features", return_value={1: {}}), \
         patch("etl.utils.load_item_features", return_value={10: {}}), \
         patch("training.dataset.MovieLensDataset"), \
         patch("importlib.import_module") as mock_import, \
         patch("mlflow.set_tracking_uri"):
        mock_import.return_value.train = mock_trainer
        run(best_hparams=best)
    call_kwargs = mock_trainer.call_args[1]
    assert call_kwargs["hparams"]["lr"] == 5e-4


def test_uses_default_hparams_when_tune_missing():
    combo = _make_combo()
    mock_reg, mock_trainer = _patch_train([combo])
    with patch("training.registry.ComponentRegistry", return_value=mock_reg), \
         patch("etl.utils.load_split_dataframes", return_value={"train": MagicMock(), "val": MagicMock()}), \
         patch("etl.utils.load_user_features", return_value={1: {}}), \
         patch("etl.utils.load_item_features", return_value={10: {}}), \
         patch("training.dataset.MovieLensDataset"), \
         patch("importlib.import_module") as mock_import, \
         patch("mlflow.set_tracking_uri"):
        mock_import.return_value.train = mock_trainer
        run(best_hparams=None)
    call_kwargs = mock_trainer.call_args[1]
    assert "lr" in call_kwargs["hparams"]


def test_returns_empty_when_no_combos():
    mock_reg = MagicMock()
    mock_reg.filter_combinations.return_value = []
    with patch("training.registry.ComponentRegistry", return_value=mock_reg), \
         patch("mlflow.set_tracking_uri"):
        result = run(losses=["NonExistent"])
    assert result.run_ids == {}


def test_no_minio_flag_passed():
    combo = _make_combo()
    mock_reg, mock_trainer = _patch_train([combo])
    with patch("training.registry.ComponentRegistry", return_value=mock_reg), \
         patch("etl.utils.load_split_dataframes", return_value={"train": MagicMock(), "val": MagicMock()}), \
         patch("etl.utils.load_user_features", return_value={1: {}}), \
         patch("etl.utils.load_item_features", return_value={10: {}}), \
         patch("training.dataset.MovieLensDataset"), \
         patch("importlib.import_module") as mock_import, \
         patch("mlflow.set_tracking_uri"):
        mock_import.return_value.train = mock_trainer
        run(save_to_minio=False)
    call_kwargs = mock_trainer.call_args[1]
    assert call_kwargs["save_to_minio"] is False
