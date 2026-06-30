"""Self-Distillation Fine-Tuning (SDFT) utilities.

Implements the method from Shenfeld et al. (2026) "Self-Distillation Enables Continual Learning",
adapted to the embedding regression / contrastive setting.

Key components:
  - EMATeacher: maintains an EMA copy of student parameters.
  - analytic_kl_loss: reverse KL between student and teacher embedding distributions,
    using the analytic per-dimension estimator (lower variance than sample-based).
  - SDFTMixin: mixin for trainer classes to attach SDFT in one call.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMATeacher:
    """Maintains an exponential moving average (EMA) copy of a student model.

    Teacher update rule (per step):
        ϕ ← α·θ + (1−α)·ϕ

    where θ are the current student parameters and ϕ are the teacher parameters.

    Using EMA rather than the frozen base or live student provides stable
    training — validated in ablation A.3 of the SDFT paper.
    """

    def __init__(self, student: nn.Module, alpha: float = 0.02) -> None:
        """
        Args:
            student: the student model whose weights are tracked.
            alpha:   EMA update rate (fraction of student weight incorporated per step).
                     Typical range: 0.01–0.05.
        """
        self.alpha = alpha
        self.teacher: nn.Module = copy.deepcopy(student)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        """Update teacher parameters: ϕ ← α·θ + (1−α)·ϕ."""
        for t_param, s_param in zip(
            self.teacher.parameters(), student.parameters()
        ):
            t_param.data.copy_(
                self.alpha * s_param.data + (1.0 - self.alpha) * t_param.data
            )

    def get_teacher(self) -> nn.Module:
        return self.teacher

    def state_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "teacher_state": self.teacher.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.alpha = state["alpha"]
        self.teacher.load_state_dict(state["teacher_state"])


def analytic_kl_loss(
    student_emb: torch.Tensor,   # (B, D)  student embedding output
    teacher_emb: torch.Tensor,   # (B, D)  teacher embedding output (detached)
    warmup_mask_dims: int = 0,   # mask loss over first N dimensions during warmup
) -> torch.Tensor:
    """Reverse KL between student and teacher embedding distributions.

    Treats each embedding dimension as an independent Gaussian with unit variance,
    so the per-dimension KL reduces to squared difference (analytic form).

    This is the analytic per-dimension estimator — lower variance than
    Monte-Carlo sampling, as chosen in the SDFT paper (Appendix A.1).

    KL(student || teacher) ≈ 0.5 * ||student_emb - teacher_emb||^2 (per-dim mean)
    """
    teacher_emb = teacher_emb.detach()   # teacher gradient is not propagated

    diff = student_emb - teacher_emb     # (B, D)

    if warmup_mask_dims > 0:
        # Mask first N dimensions during warmup to suppress teacher artifacts
        diff[:, :warmup_mask_dims] = 0.0

    return 0.5 * (diff ** 2).mean()


def sdft_loss(
    student: nn.Module,
    teacher: nn.Module,
    batch: dict[str, torch.Tensor],
    arch_encode_user_fn,
    arch_encode_item_fn,
    get_demonstration_context_fn,
    warmup_mask_dims: int = 0,
) -> torch.Tensor:
    """Compute the SDFT distillation loss for a single batch.

    Args:
        student:                     student model (being trained).
        teacher:                     EMA teacher model (frozen).
        batch:                       feature batch from DataLoader.
        arch_encode_user_fn:         bound method student.encode_user.
        arch_encode_item_fn:         bound method student.encode_item.
        get_demonstration_context_fn: bound method arch.get_demonstration_context.
        warmup_mask_dims:            mask first N dims during warmup.

    Returns:
        scalar SDFT KL loss.
    """
    # Student forward pass (no demonstration)
    student_user_emb = arch_encode_user_fn(batch)           # (B, D)
    student_item_emb = arch_encode_item_fn(batch)           # (B, D)
    student_emb = torch.cat([student_user_emb, student_item_emb], dim=-1)  # (B, 2D)

    # Teacher forward pass (with demonstration context)
    demo_ctx = get_demonstration_context_fn(batch)
    with torch.no_grad():
        teacher_user_emb = teacher.encode_user(demo_ctx)    # (B, D)
        teacher_item_emb = teacher.encode_item(batch)       # (B, D)
    teacher_emb = torch.cat([teacher_user_emb, teacher_item_emb], dim=-1)  # (B, 2D)

    return analytic_kl_loss(student_emb, teacher_emb, warmup_mask_dims)
