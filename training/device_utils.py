"""Device resolution utilities.

Priority order: TPU (torch_xla) > CUDA GPU > Apple MPS > CPU.
All trainers call get_device() at startup and log the result to MLflow.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import torch
import torch.nn as nn

from config import get_device_defaults


def get_device() -> torch.device:
    """Resolve the best available compute device."""
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import]
        return xm.xla_device()
    except ImportError:
        pass

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def device_type(device: torch.device) -> str:
    """Return a canonical string for config lookup: 'cuda', 'mps', 'xla', 'cpu'."""
    name = device.type
    if name == "cuda":
        return "cuda"
    if name == "mps":
        return "mps"
    # torch_xla devices report as 'xla'
    if name == "xla":
        return "xla"
    return "cpu"


def get_dataloader_config(device: torch.device) -> dict:
    """Return DataLoader kwargs appropriate for the resolved device."""
    dtype = device_type(device)
    return get_device_defaults(dtype)


@contextmanager
def autocast_context(device: torch.device) -> Generator[None, None, None]:
    """Enable mixed precision for CUDA and MPS; no-op for TPU/CPU."""
    dtype = device_type(device)
    if dtype in ("cuda", "mps"):
        with torch.autocast(device_type=dtype, dtype=torch.float16):
            yield
    else:
        yield


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move all tensors in a feature dict to the target device."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def log_device_to_mlflow(device: torch.device) -> None:
    """Tag the active MLflow run with the resolved device."""
    try:
        import mlflow
        mlflow.set_tag("device", str(device))
        mlflow.set_tag("device_type", device_type(device))
        if device_type(device) == "cuda":
            mlflow.set_tag("cuda_device_name", torch.cuda.get_device_name(device))
    except Exception:
        pass  # MLflow not active — silent
