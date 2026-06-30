"""Tests: ComponentRegistry — loading, filtering, extensibility."""
import pytest
import os
import tempfile
import textwrap

from training.registry import ComponentRegistry


def test_registry_loads_enabled_combinations():
    registry = ComponentRegistry()
    combos = registry.get_enabled_combinations()
    assert len(combos) >= 1
    for arch, loss in combos:
        assert loss.name in arch.compatible_losses


def test_registry_get_arch_entry():
    registry = ComponentRegistry()
    entry = registry.get_arch_entry("TwoTower")
    assert entry.name == "TwoTower"
    assert entry.enabled is True


def test_registry_get_loss_entry():
    registry = ComponentRegistry()
    entry = registry.get_loss_entry("TimedecayMSELoss")
    assert entry.name == "TimedecayMSELoss"


def test_registry_missing_arch_raises():
    registry = ComponentRegistry()
    with pytest.raises(KeyError):
        registry.get_arch_entry("NonExistentArch")


def test_registry_filter_by_arch():
    registry = ComponentRegistry()
    combos = registry.filter_combinations(architecture_names=["TwoTower"])
    assert all(a.name == "TwoTower" for a, _ in combos)


def test_registry_filter_by_loss():
    registry = ComponentRegistry()
    combos = registry.filter_combinations(loss_names=["TimedecayMSELoss"])
    assert all(l.name == "TimedecayMSELoss" for _, l in combos)


def test_disabled_entry_excluded():
    """An entry with enabled: false must not appear in enabled combinations."""
    yaml_content = textwrap.dedent("""
        losses:
          - name: TimedecayMSELoss
            module: training.two_tower.losses
            class: TimedecayMSELoss
            enabled: false
          - name: TimedecayInfoNCELoss
            module: training.infonce.losses
            class: TimedecayInfoNCELoss
            enabled: true
        architectures:
          - name: InfoNCEEncoder
            module: training.infonce.encoders
            class: InfoNCEModel
            compatible_losses: [TimedecayInfoNCELoss]
            enabled: true
        device_defaults:
          cpu:
            batch_size: 256
            num_workers: 0
            pin_memory: false
        minio:
          endpoint: "localhost:9000"
          access_key: "minioadmin"
          secret_key: "minioadmin"
          secure: false
          buckets:
            checkpoints: "model-checkpoints"
            faiss: "faiss-indices"
            teachers: "teacher-snapshots"
            cross_distill: "cross-distill"
        mlflow:
          tracking_uri: "sqlite:///test_mlflow.db"
          artifact_root: "/tmp/mlflow-artifacts"
        optuna:
          n_trials: 2
          sampler: "TPE"
          pruner: "Median"
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = f.name

    try:
        os.environ["RECSYS_CONFIG"] = tmp_path
        # Clear lru_cache so new config is loaded
        from config import load_config
        load_config.cache_clear()
        registry = ComponentRegistry()
        combos = registry.get_enabled_combinations()
        loss_names = [l.name for _, l in combos]
        assert "TimedecayMSELoss" not in loss_names
        assert "TimedecayInfoNCELoss" in loss_names
    finally:
        os.unsetenv("RECSYS_CONFIG")
        del os.environ["RECSYS_CONFIG"]
        load_config.cache_clear()
        os.unlink(tmp_path)


def test_new_arch_in_yaml_appears_in_combinations():
    """Adding an entry to YAML with enabled: true makes it appear in combinations."""
    yaml_content = textwrap.dedent("""
        losses:
          - name: TimedecayMSELoss
            module: training.two_tower.losses
            class: TimedecayMSELoss
            enabled: true
        architectures:
          - name: TwoTower
            module: training.two_tower.towers
            class: TwoTowerModel
            compatible_losses: [TimedecayMSELoss]
            enabled: true
        device_defaults:
          cpu:
            batch_size: 256
            num_workers: 0
            pin_memory: false
        minio:
          endpoint: "localhost:9000"
          access_key: "minioadmin"
          secret_key: "minioadmin"
          secure: false
          buckets:
            checkpoints: "model-checkpoints"
            faiss: "faiss-indices"
            teachers: "teacher-snapshots"
            cross_distill: "cross-distill"
        mlflow:
          tracking_uri: "sqlite:///test_mlflow.db"
          artifact_root: "/tmp/mlflow-artifacts"
        optuna:
          n_trials: 2
          sampler: "TPE"
          pruner: "Median"
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = f.name

    try:
        os.environ["RECSYS_CONFIG"] = tmp_path
        from config import load_config
        load_config.cache_clear()
        registry = ComponentRegistry()
        arch_names = [a.name for a, _ in registry.get_enabled_combinations()]
        assert "TwoTower" in arch_names
    finally:
        del os.environ["RECSYS_CONFIG"]
        load_config.cache_clear()
        os.unlink(tmp_path)
