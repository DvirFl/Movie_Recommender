"""Two-Tower trainer: training loop + within-architecture SDFT + MLflow logging."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, default_collate
from torch.nn.utils.rnn import pad_sequence

from training.base.architecture import BaseRecommenderArchitecture
from training.base.loss import BaseRecommenderLoss
from training.device_utils import (
    autocast_context, get_dataloader_config, get_device,
    log_device_to_mlflow, move_batch,
)
from training.distillation.sdft import EMATeacher, sdft_loss
from training.dataset import MovieLensDataset
from precompute.minio_io import MinIOClient


def collate_variable_sequences(batch):
    """
    Custom collate function that automatically pads variable-length sequences 
    (like user history timelines) to the maximum length within the current batch,
    safely ignoring 0-d scalar tensors.
    """
    if not batch:
        return {}
        
    first_elem = batch[0]
    
    # Scenario A: Dataset returns dictionaries
    if isinstance(first_elem, dict):
        collated_batch = {}
        for key in first_elem.keys():
            val = first_elem[key]
            
            # FIX: Check if it's a list or a tensor with at least 1 dimension (iterable)
            is_sequence = isinstance(val, list) or (isinstance(val, torch.Tensor) and val.ndim > 0)
            
            if is_sequence and not isinstance(val, str):
                lengths = [len(sample[key]) for sample in batch]
                if len(set(lengths)) > 1:  # Mismatched sequence lengths found
                    tensors = [
                        torch.tensor(sample[key]) if not isinstance(sample[key], torch.Tensor) else sample[key]
                        for sample in batch
                    ]
                    # Dynamic padding to [batch_size, max_seq_len_in_batch]
                    collated_batch[key] = pad_sequence(tensors, batch_first=True, padding_value=0)
                    collated_batch[f"{key}_mask"] = (collated_batch[key] != 0).float()
                    continue
            
            # Standard collation fallback for uniform keys (scalars, IDs, fixed multi-hots)
            collated_batch[key] = default_collate([sample[key] for sample in batch])
        return collated_batch

    # Scenario B: Dataset returns tuples/lists
    elif isinstance(first_elem, (tuple, list)):
        collated_items = []
        for i in range(len(first_elem)):
            val = first_elem[i]
            
            # FIX: Check if it's a list or a tensor with at least 1 dimension (iterable)
            is_sequence = isinstance(val, list) or (isinstance(val, torch.Tensor) and val.ndim > 0)
            
            if is_sequence and not isinstance(val, str):
                lengths = [len(sample[i]) for sample in batch]
                if len(set(lengths)) > 1:
                    tensors = [
                        torch.tensor(sample[i]) if not isinstance(sample[i], torch.Tensor) else sample[i]
                        for sample in batch
                    ]
                    collated_items.append(pad_sequence(tensors, batch_first=True, padding_value=0))
                    continue
            collated_items.append(default_collate([sample[i] for sample in batch]))
        return tuple(collated_items)

    return default_collate(batch)

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
    """Train a model with optional SDFT and log everything to MLflow.

    Args:
        model:            architecture instance (implements BaseRecommenderArchitecture).
        loss_fn:          loss instance (implements BaseRecommenderLoss).
        train_dataset:    training split MovieLensDataset.
        val_dataset:      validation split MovieLensDataset.
        hparams:          hyperparameters dict (lr, n_epochs, ema_alpha, sdft_weight, ...).
        run_name:         MLflow run name.
        experiment_name:  MLflow experiment name.
        mlflow_tags:      extra tags for the MLflow run.
        save_to_minio:    whether to push checkpoints to MinIO.

    Returns:
        MLflow run_id of the completed training run.
    """
    device = get_device()
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    dl_cfg = get_dataloader_config(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams.get("batch_size", dl_cfg["batch_size"]),
        shuffle=True,
        num_workers=dl_cfg.get("num_workers", 2),
        collate_fn=collate_variable_sequences,
        pin_memory=dl_cfg.get("pin_memory", False),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hparams.get("batch_size", dl_cfg["batch_size"]),
        shuffle=False,
        num_workers=dl_cfg.get("num_workers", 2),
        collate_fn=collate_variable_sequences,
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
    ema_teacher: EMATeacher | None = None
    if use_sdft:
        ema_teacher = EMATeacher(model, alpha=hparams.get("ema_alpha", 0.02))

    n_epochs = hparams.get("n_epochs", 5)

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    is_nested = mlflow.active_run() is not None

    with mlflow.start_run(run_name=run_name or f"{model.name}_{loss_fn.name}", nested=is_nested) as run:
        log_device_to_mlflow(device)
        mlflow.log_params(hparams)
        mlflow.set_tag("architecture", model.name)
        mlflow.set_tag("loss", loss_fn.name)
        mlflow.set_tag("distillation_type", "within" if use_sdft else "none")
        for k, v in (mlflow_tags or {}).items():
            mlflow.set_tag(k, v)

        best_val_loss = float("inf")
        sdft_weight = hparams.get("sdft_weight", 0.1)
        warmup_mask_dims = hparams.get("warmup_mask_dims", 4)

        for epoch in range(n_epochs):
            epoch_start = time.time()
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
                        mask = warmup_mask_dims if epoch == 0 else 0
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

                if ema_teacher is not None:
                    ema_teacher.update(model)

                train_loss += main_loss.item()
                train_kl += kl.item()

            scheduler.step()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = move_batch(batch, device)
                    user_emb = model.encode_user(batch)
                    item_emb = model.encode_item(batch)
                    val_loss += loss_fn(
                        user_emb, item_emb, batch["rating"], batch["weight"]
                    ).item()

            n_train = max(len(train_loader), 1)
            n_val = max(len(val_loader), 1)
            avg_train = train_loss / n_train
            avg_kl = train_kl / n_train
            avg_val = val_loss / n_val
            epoch_time = time.time() - epoch_start

            # Embedding norm diagnostics
            with torch.no_grad():
                sample_batch = move_batch(next(iter(val_loader)), device)
                u_emb = model.encode_user(sample_batch)
                i_emb = model.encode_item(sample_batch)
                u_norm = u_emb.norm(dim=-1).mean().item()
                i_norm = i_emb.norm(dim=-1).mean().item()

            mlflow.log_metrics(
                {
                    "train_loss": avg_train,
                    "val_loss": avg_val,
                    "kl_divergence": avg_kl,
                    "user_emb_norm": u_norm,
                    "item_emb_norm": i_norm,
                    "epoch_time_s": epoch_time,
                },
                step=epoch,
            )

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                ckpt_path = f"/tmp/{run.info.run_id}_best.pt"
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "teacher_state": ema_teacher.state_dict() if ema_teacher else None,
                        "epoch": epoch,
                        "val_loss": avg_val,
                        "hparams": hparams,
                    },
                    ckpt_path,
                )
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

                if save_to_minio:
                    minio = MinIOClient()
                    minio.upload_checkpoint(
                        local_path=ckpt_path,
                        run_id=run.info.run_id,
                        epoch=epoch,
                    )

        mlflow.log_metric("best_val_loss", best_val_loss)
        return run.info.run_id
