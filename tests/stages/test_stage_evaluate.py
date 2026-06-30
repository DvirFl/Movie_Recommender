"""Tests: stage_evaluate."""
import pytest
from unittest.mock import MagicMock, patch
import torch
from stages.stage_evaluate import run, EvaluateResult


def _make_combo(arch_name="TwoTower", loss_name="TimedecayMSELoss"):
    arch = MagicMock(); arch.name = arch_name
    arch.cls.return_value = MagicMock(
        encode_user=MagicMock(return_value=torch.randn(4, 32)),
        encode_item=MagicMock(return_value=torch.randn(4, 32)),
    )
    arch.cls.return_value.to.return_value = arch.cls.return_value
    loss = MagicMock(); loss.name = loss_name
    loss.cls.return_value = MagicMock(
        return_value=torch.tensor(0.5),
        to=MagicMock(return_value=MagicMock(return_value=torch.tensor(0.5))),
    )
    return arch, loss


def test_returns_evaluate_result():
    combo = _make_combo()
    mock_reg = MagicMock()
    mock_reg.filter_combinations.return_value = [combo]
    with patch("training.registry.ComponentRegistry", return_value=mock_reg), \
         patch("etl.utils.load_split_dataframes",
               return_value={"test": MagicMock()}), \
         patch("etl.utils.load_user_features", return_value={1: {}}), \
         patch("etl.utils.load_item_features", return_value={10: {}}), \
         patch("training.dataset.MovieLensDataset"), \
         patch("torch.utils.data.DataLoader", return_value=[]), \
         patch("training.device_utils.get_device", return_value=torch.device("cpu")), \
         patch("mlflow.set_tracking_uri"), \
         patch("mlflow.start_run"), \
         patch("mlflow.set_tag"), \
         patch("mlflow.log_metric"):
        result = run()
    assert isinstance(result, EvaluateResult)


def test_test_loss_stored_per_key():
    combo = _make_combo()
    mock_reg = MagicMock()
    mock_reg.filter_combinations.return_value = [combo]

    mock_loss_instance = MagicMock()
    mock_loss_instance.return_value = torch.tensor(0.42)
    combo[1].cls.return_value = mock_loss_instance
    mock_loss_instance.to.return_value = mock_loss_instance

    mock_arch_instance = MagicMock()
    mock_arch_instance.encode_user.return_value = torch.randn(4, 32)
    mock_arch_instance.encode_item.return_value = torch.randn(4, 32)
    mock_arch_instance.to.return_value = mock_arch_instance
    combo[0].cls.return_value = mock_arch_instance

    fake_batch = {
        "rating": torch.rand(4),
        "weight": torch.rand(4),
        "user_id": torch.randint(0, 5, (4,)),
        "movie_id": torch.randint(0, 10, (4,)),
    }

    with patch("training.registry.ComponentRegistry", return_value=mock_reg), \
         patch("etl.utils.load_split_dataframes",
               return_value={"test": MagicMock()}), \
         patch("etl.utils.load_user_features", return_value={1: {}}), \
         patch("etl.utils.load_item_features", return_value={10: {}}), \
         patch("training.dataset.MovieLensDataset"), \
         patch("torch.utils.data.DataLoader", return_value=[fake_batch]), \
         patch("training.device_utils.get_device", return_value=torch.device("cpu")), \
         patch("training.device_utils.move_batch", side_effect=lambda b, d: b), \
         patch("mlflow.set_tracking_uri"), \
         patch("mlflow.start_run", return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))), \
         patch("mlflow.set_tag"), \
         patch("mlflow.log_metric"):
        result = run()
    assert "TwoTower_TimedecayMSELoss" in result.test_losses


def test_empty_when_no_combos():
    mock_reg = MagicMock()
    mock_reg.filter_combinations.return_value = []
    with patch("training.registry.ComponentRegistry", return_value=mock_reg), \
         patch("etl.utils.load_split_dataframes",
               return_value={"test": MagicMock()}), \
         patch("etl.utils.load_user_features", return_value={1: {}}), \
         patch("etl.utils.load_item_features", return_value={10: {}}), \
         patch("training.dataset.MovieLensDataset"), \
         patch("torch.utils.data.DataLoader", return_value=[]), \
         patch("training.device_utils.get_device", return_value=torch.device("cpu")), \
         patch("mlflow.set_tracking_uri"), \
         patch("mlflow.start_run", return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))), \
         patch("mlflow.set_tag"), \
         patch("mlflow.log_metric"):
        result = run(losses=["NonExistent"])
    assert result.test_losses == {}
