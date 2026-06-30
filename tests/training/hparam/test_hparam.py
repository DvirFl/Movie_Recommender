"""Tests: Hyperparameter tuning — Optuna study runs ≥2 trials, best config retrievable."""
import pytest
from unittest.mock import patch, MagicMock

from training.hparam.search_spaces import build_search_space, SHARED_SEARCH_SPACE
from training.two_tower.towers import TwoTowerModel
from training.two_tower.losses import TimedecayMSELoss
from training.infonce.encoders import InfoNCEModel
from training.infonce.losses import TimedecayInfoNCELoss


def test_shared_search_space_has_required_keys():
    required = {"lr", "batch_size", "n_epochs", "sdft_weight", "ema_alpha"}
    assert required.issubset(set(SHARED_SEARCH_SPACE.keys()))


def test_build_search_space_merges_all():
    arch = TwoTowerModel(n_users=5, n_items=10, output_dim=16, hidden_dims=[16])
    loss = TimedecayMSELoss()
    space = build_search_space(arch, loss)
    # Must contain shared + arch + loss keys
    assert "lr" in space                    # shared
    assert "embed_dim" in space             # arch-specific
    assert "decay_lambda" in space          # loss-specific


def test_build_search_space_infonce():
    arch = InfoNCEModel(n_users=5, n_items=10, output_dim=16, hidden_dims=[16], n_layers=1)
    loss = TimedecayInfoNCELoss()
    space = build_search_space(arch, loss)
    assert "n_heads" in space
    assert "temperature" in space


def test_optuna_study_runs_two_trials():
    """Optuna study with n_trials=2 must complete without error."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    call_count = {"n": 0}

    def objective(trial):
        call_count["n"] += 1
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        return lr  # dummy metric

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2)

    assert call_count["n"] == 2
    assert study.best_params is not None
    assert "lr" in study.best_params


def test_suggest_params_all_types():
    import optuna
    from training.hparam.tuner import _suggest_params

    space = {
        "lr": ("float", 1e-5, 1e-2, {"log": True}),
        "n_layers": ("int", 1, 4),
        "activation": ("categorical", ["relu", "gelu"]),
    }

    def objective(trial):
        params = _suggest_params(trial, space)
        assert "lr" in params
        assert "n_layers" in params
        assert "activation" in params
        assert params["activation"] in ["relu", "gelu"]
        return params["lr"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2)
