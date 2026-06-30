"""InfoNCE trainer: training loop + within-architecture SDFT + MLflow logging.

Identical contract to two_tower/trainer.py but operates on the InfoNCE architecture.
Kept as a separate module so each architecture's training loop can diverge
(e.g. multi-negative mining, sequence padding) without coupling.
"""
from __future__ import annotations

import time
from typing import Any

import mlflow
import torch
from torch.utils.data import DataLoader

from training.base.architecture import BaseRecommenderArchitecture
from training.base.loss import BaseRecommenderLoss
from training.dataset import MovieLensDataset
from training.device_utils import (
    autocast_context, get_dataloader_config, get_device,
    log_device_to_mlflow, move_batch,
)
from training.distillation.sdft import EMATeacher, sdft_loss
from precompute.minio_io import MinIOClient


def train(
    model: BaseRecommenderArchitecture,
    loss_fn: BaseRecommenderLoss,
    train_dataset: MovieLensDataset,
    val_dataset: MovieLensDataset,
    hparams: dict[str, Any],
    run_name: str | None = None,
    experiment_name: str | None = None,
    mlflow_tags: dict[str, str] | None = None,
    save_to_minio: bool = True,
) -> str:
    """Train InfoNCE model and log to MLflow. Returns run_id."""
    device = get_device()
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    dl_cfg = get_dataloader_config(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams.get("batch_size", dl_cfg["batch_size"]),
        shuffle=True,
        num_workers=dl_cfg.get("num_workers", 2),
        pin_memory=dl_cfg.get("pin_memory", False),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hparams.get("batch_size", dl_cfg["batch_size"]),
        shuffle=False,
        num_workers=dl_cfg.get("num_workers", 2),
        pin_memory=dl_cfg.get("pin_memory", False),
    )

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=hparams.get("lr", 1e-3),
        weight_decay=hparams.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hparams.get("n_epochs", 5)
    )

    use_sdft = model.supports_sdft() and hparams.get("sdft_weight", 0.1) > 0
    ema_teacher: EMATeacher | None = (
        EMATeacher(model, alpha=hparams.get("ema_alpha", 0.02)) if use_sdft else None
    )
    n_epochs = hparams.get("n_epochs", 5)

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name or f"{model.name}_{loss_fn.name}") as run:
        log_device_to_mlflow(device)
        mlflow.log_params(hparams)
        mlflow.set_tag("architecture", model.name)
        mlflow.set_tag("loss", loss_fn.name)
        mlflow.set_tag("distillation_type", "within" if use_sdft else "none")
        for k, v in (mlflow_tags or {}).items():
            mlflow.set_tag(k, v)

        best_val_loss = float("inf")
        sdft_weight = hparams.get("sdft_weight", 0.1)

        for epoch in range(n_epochs):
            t0 = time.time()
            model.train()
            loss_fn.train()
            train_loss, train_kl = 0.0, 0.0

            for batch in train_loader:
                batch = move_batch(batch, device)
                optimizer.zero_grad()

                with autocast_context(device):
                    user_emb = model.encode_user(batch)
                    item_emb = model.encode_item(batch)
                    main_loss = loss_fn(
                        user_emb, item_emb, batch["rating"], batch["weight"]
                    )
                    kl = torch.tensor(0.0, device=device)
                    if use_sdft and ema_teacher is not None:
                        mask = hparams.get("warmup_mask_dims", 4) if epoch == 0 else 0
                        kl = sdft_loss(
                            student=model,
                            teacher=ema_teacher.get_teacher(),
                            batch=batch,
                            arch_encode_user_fn=model.encode_user,
                            arch_encode_item_fn=model.encode_item,
                            get_demonstration_context_fn=model.get_demonstration_context,
                            warmup_mask_dims=mask,
                        )
                    total = main_loss + sdft_weight * kl

                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if ema_teacher:
                    ema_teacher.update(model)
                train_loss += main_loss.item()
                train_kl += kl.item()

            scheduler.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = move_batch(batch, device)
                    val_loss += loss_fn(
                        model.encode_user(batch),
                        model.encode_item(batch),
                        batch["rating"],
                        batch["weight"],
                    ).item()

            n_tr = max(len(train_loader), 1)
            n_v = max(len(val_loader), 1)
            with torch.no_grad():
                sb = move_batch(next(iter(val_loader)), device)
                u_norm = model.encode_user(sb).norm(dim=-1).mean().item()
                i_norm = model.encode_item(sb).norm(dim=-1).mean().item()

            mlflow.log_metrics(
                {
                    "train_loss": train_loss / n_tr,
                    "val_loss": val_loss / n_v,
                    "kl_divergence": train_kl / n_tr,
                    "user_emb_norm": u_norm,
                    "item_emb_norm": i_norm,
                    "epoch_time_s": time.time() - t0,
                },
                step=epoch,
            )

            if val_loss / n_v < best_val_loss:
                best_val_loss = val_loss / n_v
                ckpt = f"/tmp/{run.info.run_id}_best.pt"
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "teacher_state": ema_teacher.state_dict() if ema_teacher else None,
                        "epoch": epoch,
                        "val_loss": best_val_loss,
                        "hparams": hparams,
                    },
                    ckpt,
                )
                mlflow.log_artifact(ckpt, artifact_path="checkpoints")
                if save_to_minio:
                    MinIOClient().upload_checkpoint(ckpt, run.info.run_id, epoch)

        mlflow.log_metric("best_val_loss", best_val_loss)
        return run.info.run_id
