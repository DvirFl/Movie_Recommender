"""Tests: device resolution with mocked availability."""
import pytest
from unittest.mock import patch, MagicMock
import torch


def test_get_device_returns_cpu_when_nothing_available():
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=False), \
         patch.dict("sys.modules", {"torch_xla": None, "torch_xla.core": None,
                                    "torch_xla.core.xla_model": None}):
        from training.device_utils import get_device
        device = get_device()
        assert device.type == "cpu"


def test_get_device_returns_cuda_when_available():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.backends.mps.is_available", return_value=False), \
         patch.dict("sys.modules", {"torch_xla": None, "torch_xla.core": None,
                                    "torch_xla.core.xla_model": None}):
        from training.device_utils import get_device
        device = get_device()
        assert device.type == "cuda"


def test_get_device_prefers_cuda_over_mps():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.backends.mps.is_available", return_value=True), \
         patch.dict("sys.modules", {"torch_xla": None, "torch_xla.core": None,
                                    "torch_xla.core.xla_model": None}):
        from training.device_utils import get_device
        device = get_device()
        assert device.type == "cuda"


def test_get_device_returns_mps_when_no_cuda():
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=True), \
         patch.dict("sys.modules", {"torch_xla": None, "torch_xla.core": None,
                                    "torch_xla.core.xla_model": None}):
        from training.device_utils import get_device
        device = get_device()
        assert device.type == "mps"


def test_device_type_string():
    from training.device_utils import device_type
    assert device_type(torch.device("cuda")) == "cuda"
    assert device_type(torch.device("cpu")) == "cpu"
    assert device_type(torch.device("mps")) == "mps"


def test_move_batch_moves_tensors():
    from training.device_utils import move_batch
    batch = {"a": torch.tensor([1.0]), "b": "not_a_tensor"}
    result = move_batch(batch, torch.device("cpu"))
    assert isinstance(result["a"], torch.Tensor)
    assert result["b"] == "not_a_tensor"


def test_autocast_context_cpu_no_error():
    from training.device_utils import autocast_context
    with autocast_context(torch.device("cpu")):
        x = torch.randn(4, 8)
        y = x @ x.T
    assert y.shape == (4, 4)


def test_get_dataloader_config_returns_dict():
    from training.device_utils import get_dataloader_config
    cfg = get_dataloader_config(torch.device("cpu"))
    assert "batch_size" in cfg
    assert isinstance(cfg["batch_size"], int)
