"""Hyperparameter tuning via Optuna, with each trial logged as a nested MLflow run."""
from __future__ import annotations

from typing import Any, Callable

import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback

from config import get_optuna_config
from training.base.architecture import BaseRecommenderArchitecture
from training.base.loss import BaseRecommenderLoss
from training.hparam.search_spaces import build_search_space


def _suggest_params(
    trial: optuna.Trial,
    search_space: dict[str, Any],
) -> dict[str, Any]:
    """Suggest values for all parameters in the search space."""
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
    objective_fn: Callable[[dict[str, Any]], float],
    experiment_name: str,
    study_name: str | None = None,
) -> dict[str, Any]:
    """Run an Optuna hyperparameter sweep for an arch×loss combination.

    Args:
        arch_cls:        architecture class (implements BaseRecommenderArchitecture).
        arch_kwargs:     fixed constructor kwargs (n_users, n_items, etc.)
        loss_cls:        loss class (implements BaseRecommenderLoss).
        objective_fn:    callable(hparams_dict) -> val_metric (lower is better).
        experiment_name: MLflow experiment to log trials under.
        study_name:      Optuna study name (defaults to experiment_name).

    Returns:
        best_params dict.
    """
    cfg = get_optuna_config()

    # Build combined search space: shared + arch-specific + loss-specific
    # if "arch_kwargs" in locals() or "arch_kwargs" in globals():
    #         if "n_genres" in arch_kwargs:
    #             # If it's hardcoded to 20, override it to 18 for MovieLens 1M
    #             arch_kwargs["n_genres"] = 18
    arch_instance = arch_cls(**arch_kwargs)
    loss_instance = loss_cls()
    search_space = build_search_space(arch_instance, loss_instance)

    mlflow.set_experiment(experiment_name)
    mlflow_cb = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="val_metric",
        create_experiment=False,
        mlflow_kwargs={"nested": True},
    )

    sampler_cls = getattr(optuna.samplers, f"{cfg['sampler']}Sampler")
    pruner_cls = getattr(optuna.pruners, f"{cfg['pruner']}Pruner")

    study = optuna.create_study(
        study_name=study_name or experiment_name,
        direction="minimize",
        sampler=sampler_cls(),
        pruner=pruner_cls(),
    )

    with mlflow.start_run(run_name=f"sweep_{study_name or experiment_name}"):
        mlflow.set_tag("sweep", "true")
        mlflow.set_tag("architecture", arch_cls.__name__ if hasattr(arch_cls, "__name__") else str(arch_cls))
        mlflow.set_tag("loss", loss_cls.__name__ if hasattr(loss_cls, "__name__") else str(loss_cls))

        def objective(trial: optuna.Trial) -> float:
            hparams = _suggest_params(trial, search_space)
            return objective_fn(hparams)

        study.optimize(
            objective,
            n_trials=cfg["n_trials"],
            # callbacks=[mlflow_cb],
        )

    best = study.best_params
    # Log best config to MLflow
    with mlflow.start_run(run_name=f"best_config_{study_name or experiment_name}", nested=False):
        mlflow.set_tag("sweep_result", "best")
        mlflow.log_params(best)
        mlflow.log_metric("best_val_metric", study.best_value)

    return best
