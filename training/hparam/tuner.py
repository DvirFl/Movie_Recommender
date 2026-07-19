"""Hyperparameter tuning via Optuna with RDB storage, logging only best result to MLflow."""
from __future__ import annotations

import os
from typing import Any, Callable

import mlflow
import optuna

from config import get_optuna_config
from training.base.architecture import BaseRecommenderArchitecture
from training.base.loss import BaseRecommenderLoss
from training.hparam.search_spaces import build_search_space

# Suppress per-trial Optuna logs — only warnings and above
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_params(
    trial: optuna.Trial,
    search_space: dict[str, Any],
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = spec[0]
        if kind == "float":
            low, high = spec[1], spec[2]
            kwargs = spec[3] if len(spec) > 3 else {}
            params[name] = trial.suggest_float(name, low, high, **kwargs)
        elif kind == "int":
            low, high = spec[1], spec[2]
            params[name] = trial.suggest_int(name, low, high)
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec[1])
        else:
            raise ValueError(f"Unknown hparam type: {kind}")
    return params


def run_sweep(
    arch_cls: type,
    arch_kwargs: dict[str, Any],
    loss_cls: type,
    objective_fn: Callable[[dict[str, Any]], list[float]],
    experiment_name: str,
    study_name: str | None = None,
) -> dict[str, Any]:
    """Run an Optuna hyperparameter sweep for an arch×loss combination.

    Uses RDB storage (OPTUNA_STORAGE_URL env var) for persistence and resumability.
    Only logs the best result to MLflow — no per-trial MLflow calls.

    Args:
        arch_cls:        architecture class.
        arch_kwargs:     fixed constructor kwargs (n_users, n_items, etc.)
        loss_cls:        loss class.
        objective_fn:    callable(hparams_dict) -> list[float] of per-epoch val losses.
        experiment_name: MLflow experiment to log best result under.
        study_name:      Optuna study name (defaults to experiment_name).

    Returns:
        best_params dict.
    """
    cfg = get_optuna_config()

    arch_instance = arch_cls(**arch_kwargs)
    loss_instance = loss_cls()
    search_space = build_search_space(arch_instance, loss_instance)

    sampler_cls = getattr(optuna.samplers, f"{cfg['sampler']}Sampler")
    pruner_cls  = getattr(optuna.pruners,  f"{cfg['pruner']}Pruner")

    # Use RDB storage if configured — enables persistence and resumability
    storage_url = os.environ.get("OPTUNA_STORAGE_URL", None)

    study = optuna.create_study(
        study_name=study_name or experiment_name,
        direction="minimize",
        storage=storage_url,
        load_if_exists=True,
        sampler=sampler_cls(),
        pruner=pruner_cls(),
    )

    def objective(trial: optuna.Trial) -> float:
        hparams = _suggest_params(trial, search_space)
        epoch_losses = objective_fn(hparams)
        for step, loss in enumerate(epoch_losses):
            trial.report(loss, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return epoch_losses[-1] if epoch_losses else 9999.0

    study.optimize(
        objective,
        n_trials=cfg["n_trials"],
    )

    best = study.best_params

    # Log only the best result to MLflow — no per-trial HTTP calls
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"best_{study_name or experiment_name}"):
        mlflow.set_tag("sweep_result", "best")
        mlflow.set_tag("architecture", arch_cls.__name__)
        mlflow.set_tag("loss", loss_cls.__name__)
        mlflow.set_tag("n_trials", str(len(study.trials)))
        mlflow.log_params(best)
        mlflow.log_metric("best_val_metric", study.best_value)

    return best