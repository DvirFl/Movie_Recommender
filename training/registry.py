"""Component registry — discovers architectures and losses from config/registry.yaml.

Usage:
    registry = ComponentRegistry()
    combos = registry.get_enabled_combinations()
    # -> [(TwoTowerModel_instance, TimedecayMSELoss_instance), ...]

Adding a new arch or loss:
    1. Implement BaseRecommenderArchitecture or BaseRecommenderLoss.
    2. Add entry to config/registry.yaml with enabled: true.
    3. Done — no changes here needed.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Type

from config import load_config
from training.base.architecture import BaseRecommenderArchitecture
from training.base.loss import BaseRecommenderLoss


@dataclass
class ArchEntry:
    name: str
    cls: Type[BaseRecommenderArchitecture]
    compatible_losses: list[str]
    enabled: bool


@dataclass
class LossEntry:
    name: str
    cls: Type[BaseRecommenderLoss]
    enabled: bool


class ComponentRegistry:
    """Loads all registered architectures and losses from the YAML config."""

    def __init__(self) -> None:
        cfg = load_config()
        self._losses: dict[str, LossEntry] = {}
        self._architectures: dict[str, ArchEntry] = {}

        for entry in cfg.get("losses", []):
            cls = self._import(entry["module"], entry["class"])
            self._losses[entry["name"]] = LossEntry(
                name=entry["name"],
                cls=cls,
                enabled=entry.get("enabled", True),
            )

        for entry in cfg.get("architectures", []):
            cls = self._import(entry["module"], entry["class"])
            self._architectures[entry["name"]] = ArchEntry(
                name=entry["name"],
                cls=cls,
                compatible_losses=entry.get("compatible_losses", []),
                enabled=entry.get("enabled", True),
            )

    @staticmethod
    def _import(module_path: str, class_name: str) -> type:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def get_enabled_losses(self) -> dict[str, LossEntry]:
        return {k: v for k, v in self._losses.items() if v.enabled}

    def get_enabled_architectures(self) -> dict[str, ArchEntry]:
        return {k: v for k, v in self._architectures.items() if v.enabled}

    def get_enabled_combinations(self) -> list[tuple[ArchEntry, LossEntry]]:
        """Return all enabled (arch, loss) pairs where loss is compatible."""
        combos = []
        for arch in self.get_enabled_architectures().values():
            for loss_name in arch.compatible_losses:
                loss = self._losses.get(loss_name)
                if loss and loss.enabled:
                    combos.append((arch, loss))
        return combos

    def filter_combinations(
        self,
        architecture_names: list[str] | None = None,
        loss_names: list[str] | None = None,
    ) -> list[tuple[ArchEntry, LossEntry]]:
        """Return combinations filtered by requested arch/loss names.

        None means 'all enabled'.
        """
        combos = self.get_enabled_combinations()
        if architecture_names:
            combos = [(a, l) for a, l in combos if a.name in architecture_names]
        if loss_names:
            combos = [(a, l) for a, l in combos if l.name in loss_names]
        return combos

    def get_arch_entry(self, name: str) -> ArchEntry:
        entry = self._architectures.get(name)
        if not entry:
            raise KeyError(f"Architecture '{name}' not found in registry.")
        return entry

    def get_loss_entry(self, name: str) -> LossEntry:
        entry = self._losses.get(name)
        if not entry:
            raise KeyError(f"Loss '{name}' not found in registry.")
        return entry
