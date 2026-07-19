"""Stage 5 — Tune: Optuna hyperparameter sweep for every requested arch×loss combination."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TuneResult:
    # Maps "ArchName_LossName" -> best hparams dict
    best_hparams: dict[str, dict] = field(default_factory=dict)


def run(
    losses: list[str] | str = "all",
    architectures: list[str] | str = "all",
    n_trials: int | None = None,
    mlflow_tracking_uri: str | None = None,
) -> TuneResult:
    """Run Optuna hyperparameter sweeps for the requested arch×loss combinations.

    Args:
        losses:               list of loss names, or "all" for every enabled loss.
        architectures:        list of arch names, or "all" for every enabled architecture.
        n_trials:             override the number of Optuna trials from registry.yaml.
        mlflow_tracking_uri:  override MLflow tracking URI from config.

    Returns:
        TuneResult mapping each combination key to its best hparams dict.
    """
    import mlflow
    from config import get_mlflow_config
    from training.registry import ComponentRegistry
    from training.hparam.tuner import run_sweep
    from etl.utils import load_split_dataframes, load_user_features, load_item_features
    from training.dataset import MovieLensDataset
    import importlib

    uri = mlflow_tracking_uri or get_mlflow_config()["tracking_uri"]
    mlflow.set_tracking_uri(uri)

    registry = ComponentRegistry()
    arch_filter = None if architectures == "all" else (
        architectures if isinstance(architectures, list) else [architectures]
    )
    loss_filter = None if losses == "all" else (
        losses if isinstance(losses, list) else [losses]
    )
    combos = registry.filter_combinations(
        architecture_names=arch_filter,
        loss_names=loss_filter,
    )

    if not combos:
        logger.warning("[tune] No enabled combinations match the requested filters.")
        return TuneResult()

    # Load data once for all combos
    splits_raw = load_split_dataframes()
    uf = load_user_features()
    itf = load_item_features()
    n_users = max(uf.keys()) + 1
    n_items = max(itf.keys()) + 1

    sample_item = next(iter(itf.values()))
    n_genres = len(sample_item['genre_multihot'])
    # logger.info("[tune] n_genres=%d from item_id=%s", n_genres, next(iter(itf)))

    train_ds = MovieLensDataset(splits_raw["train"], uf, itf, split="train")
    val_ds   = MovieLensDataset(splits_raw["val"],   uf, itf, split="val")

    result = TuneResult()

    for arch_entry, loss_entry in combos:
        key = f"{arch_entry.name}_{loss_entry.name}"
        logger.info("[tune] Sweeping %s ...", key)

        # Resolve trainer for this architecture
        parts = arch_entry.cls.__module__.split(".")
        trainer_mod = importlib.import_module(".".join(parts[:-1]) + ".trainer")
        train_fn = trainer_mod.train

        # def objective(hparams: dict) -> float:
        #     arch = arch_entry.cls(n_users=n_users, n_items=n_items, n_genres=n_genres)
        #     loss = loss_entry.cls()
        #     sweep_hparams = {**hparams,
        #                      "n_epochs": max(1, hparams.get("n_epochs", 2) // 4)}
        #     run_id = train_fn(
        #         arch, loss, train_ds, val_ds,
        #         hparams=sweep_hparams,
        #         experiment_name=f"hparam/{key}",
        #         save_to_minio=False,
        #         mlflow_tags={"sweep_trial": "true"},
        #     )
        #     client = mlflow.tracking.MlflowClient()
        #     run = client.get_run(run_id)
        #     return float(run.data.metrics.get("best_val_loss", 9999.0))

        def objective(hparams: dict) -> list[float]:
            arch = arch_entry.cls(n_users=n_users, n_items=n_items, n_genres=n_genres)
            loss = loss_entry.cls()
            sweep_hparams = {
                **hparams,
                "n_epochs": max(1, hparams.get("n_epochs", 2) // 4),
                "sdft_weight": 0.0,   # skip SDFT during sweep
                "batch_size": 512,    # cap batch size during sweep
            }
            run_id = train_fn(
                arch, loss, train_ds, val_ds,
                hparams=sweep_hparams,
                experiment_name=f"hparam/{key}",
                save_to_minio=False,
                mlflow_tags={"sweep_trial": "true"},
            )
            client = mlflow.tracking.MlflowClient()
            history = client.get_metric_history(run_id, "val_loss")
            return [m.value for m in sorted(history, key=lambda m: m.step)] or [9999.0]
        
        # Override n_trials if supplied
        if n_trials is not None:
            import optuna
            from config import get_optuna_config
            cfg = get_optuna_config()
            sampler_cls = getattr(optuna.samplers, f"{cfg['sampler']}Sampler")
            pruner_cls  = getattr(optuna.pruners,  f"{cfg['pruner']}Pruner")
            study = optuna.create_study(
                study_name=key, direction="minimize",
                sampler=sampler_cls(), pruner=pruner_cls(),
            )
            study.optimize(objective, n_trials=n_trials)
            best = study.best_params
        else:
            if mlflow.active_run():
                logger.warning(
                    "[tune] Found lingering active MLflow run (%s). Ending it now for a clean state.",
                    mlflow.active_run().info.run_id
                )
                mlflow.end_run()

            best = run_sweep(
                arch_cls=arch_entry.cls,
                arch_kwargs={"n_users": n_users, "n_items": n_items, "n_genres": n_genres},
                loss_cls=loss_entry.cls,
                objective_fn=objective,
                experiment_name=f"hparam/{key}",
                study_name=key,
            )

        result.best_hparams[key] = best
        logger.info("[tune] Best hparams for %s: %s", key, best)

    return result
