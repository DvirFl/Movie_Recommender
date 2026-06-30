"""Configuration loader — reads config/registry.yaml once and caches."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).parent / "registry.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    path = Path(os.environ.get("RECSYS_CONFIG", str(_CONFIG_PATH)))
    with path.open() as fh:
        return yaml.safe_load(fh)


def get_minio_config() -> dict[str, Any]:
    return load_config()["minio"]


def get_mlflow_config() -> dict[str, Any]:
    return load_config()["mlflow"]


def get_optuna_config() -> dict[str, Any]:
    return load_config()["optuna"]


def get_device_defaults(device_type: str) -> dict[str, Any]:
    defaults = load_config().get("device_defaults", {})
    return defaults.get(device_type, defaults["cpu"])
