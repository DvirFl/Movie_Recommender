"""Cross-architecture SDFT distillation.

After within-architecture training is complete, each trained model acts as
teacher for the other:
  - TwoTower (EMA teacher) → InfoNCE student
  - InfoNCE (EMA teacher) → TwoTower student

Both directions run independently and are logged as separate MLflow runs
under the experiment 'distillation/cross'.
"""
from __future__ import annotations

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.distillation.sdft import analytic_kl_loss, EMATeacher
from training.device_utils import get_device, move_batch, autocast_context, log_device_to_mlflow


def cross_distill(
    student: nn.Module,
    teacher_model: nn.Module,
    dataloader: DataLoader,
    n_epochs: int = 2,
    lr: float = 1e-4,
    ema_alpha: float = 0.02,
    student_name: str = "student",
    teacher_name: str = "teacher",
    warmup_mask_dims: int = 4,
) -> nn.Module:
    """Distil teacher embeddings into student via cross-architecture reverse KL.

    Args:
        student:          student architecture being updated.
        teacher_model:    frozen teacher from the other architecture.
        dataloader:       training data loader.
        n_epochs:         number of distillation epochs.
        lr:               learning rate for student.
        ema_alpha:        EMA rate for student's own EMA teacher (stability).
        student_name:     name tag for MLflow.
        teacher_name:     name tag for MLflow.
        warmup_mask_dims: mask first N embedding dims in epoch 0.

    Returns:
        Updated student model (on CPU).
    """
    device = get_device()
    student = student.to(device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)

    # Student also gets its own EMA teacher for stability
    student_ema = EMATeacher(student, alpha=ema_alpha)

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)

    with mlflow.start_run(run_name=f"cross_distill_{teacher_name}_to_{student_name}", nested=True):
        mlflow.set_tag("distillation_type", "cross")
        mlflow.set_tag("student", student_name)
        mlflow.set_tag("teacher", teacher_name)
        log_device_to_mlflow(device)
        mlflow.log_params({"n_epochs": n_epochs, "lr": lr, "ema_alpha": ema_alpha})

        for epoch in range(n_epochs):
            mask_dims = warmup_mask_dims if epoch == 0 else 0
            student.train()
            epoch_loss = 0.0

            for step, batch in enumerate(dataloader):
                batch = move_batch(batch, device)
                optimizer.zero_grad()

                with autocast_context(device):
                    # Student forward (no demo context)
                    s_user = student.encode_user(batch)
                    s_item = student.encode_item(batch)
                    s_emb = torch.cat([s_user, s_item], dim=-1)

                    # Teacher forward (with demo context from the other architecture)
                    demo_ctx = teacher_model.get_demonstration_context(batch) \
                        if hasattr(teacher_model, "get_demonstration_context") else batch
                    with torch.no_grad():
                        t_user = teacher_model.encode_user(demo_ctx)
                        t_item = teacher_model.encode_item(batch)
                    t_emb = torch.cat([t_user, t_item], dim=-1)

                    # Project teacher emb to student's dim if needed
                    if s_emb.shape[-1] != t_emb.shape[-1]:
                        min_dim = min(s_emb.shape[-1], t_emb.shape[-1])
                        s_emb = s_emb[..., :min_dim]
                        t_emb = t_emb[..., :min_dim]

                    loss = analytic_kl_loss(s_emb, t_emb, warmup_mask_dims=mask_dims)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                student_ema.update(student)
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(dataloader), 1)
            mlflow.log_metric("cross_kl_loss", avg_loss, step=epoch)

    return student.cpu()
