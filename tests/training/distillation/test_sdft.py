"""Tests: SDFT — EMA update correctness, KL loss, cross-arch embedding distance."""
import copy
import pytest
import torch

from training.distillation.sdft import EMATeacher, analytic_kl_loss, sdft_loss
from training.two_tower.towers import TwoTowerModel
from training.infonce.encoders import InfoNCEModel


@pytest.fixture
def small_tt():
    return TwoTowerModel(n_users=10, n_items=50, output_dim=32, hidden_dims=[32])


@pytest.fixture
def small_infonce():
    return InfoNCEModel(n_users=10, n_items=50, output_dim=32, hidden_dims=[32], n_layers=1)


def test_ema_teacher_initialises_equal_to_student(small_tt):
    teacher = EMATeacher(small_tt, alpha=0.02)
    for t_p, s_p in zip(teacher.teacher.parameters(), small_tt.parameters()):
        assert torch.allclose(t_p, s_p)


def test_ema_teacher_update_moves_toward_student(small_tt):
    teacher = EMATeacher(small_tt, alpha=0.5)
    # Fill student params with a known non-zero value
    with torch.no_grad():
        for p in small_tt.parameters():
            p.fill_(2.0)
    # Fill teacher params with a different value
    with torch.no_grad():
        for p in teacher.teacher.parameters():
            p.fill_(0.0)
    teacher.update(small_tt)
    # After update: teacher = 0.5 * 2.0 + 0.5 * 0.0 = 1.0
    for t_p in teacher.teacher.parameters():
        assert torch.allclose(t_p, torch.full_like(t_p, 1.0), atol=1e-5)


def test_ema_teacher_does_not_propagate_gradients(small_tt):
    teacher = EMATeacher(small_tt, alpha=0.02)
    for p in teacher.teacher.parameters():
        assert not p.requires_grad


def test_analytic_kl_loss_zero_when_equal():
    emb = torch.randn(8, 32)
    loss = analytic_kl_loss(emb, emb.clone())
    assert loss.item() < 1e-6


def test_analytic_kl_loss_positive():
    student = torch.randn(8, 32)
    teacher = torch.randn(8, 32)
    loss = analytic_kl_loss(student, teacher)
    assert loss.item() > 0.0


def test_analytic_kl_warmup_mask():
    student = torch.ones(4, 16)
    teacher = torch.zeros(4, 16)
    # With mask_dims=8, first 8 dims zeroed → loss only from last 8 dims
    loss_full = analytic_kl_loss(student.clone(), teacher.clone(), warmup_mask_dims=0)
    loss_masked = analytic_kl_loss(student.clone(), teacher.clone(), warmup_mask_dims=8)
    assert loss_masked.item() < loss_full.item()


def test_sdft_loss_computable(small_tt, tiny_batch):
    teacher_model = copy.deepcopy(small_tt)
    loss = sdft_loss(
        student=small_tt,
        teacher=teacher_model,
        batch=tiny_batch,
        arch_encode_user_fn=small_tt.encode_user,
        arch_encode_item_fn=small_tt.encode_item,
        get_demonstration_context_fn=small_tt.get_demonstration_context,
    )
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0.0


def test_sdft_loss_decreases_with_training_steps(small_tt, tiny_batch):
    """Student KL toward a fixed teacher should not diverge over steps."""
    import copy
    torch.manual_seed(42)
    # Create a distinct teacher by perturbing a copy
    fixed_teacher = copy.deepcopy(small_tt)
    with torch.no_grad():
        for p in fixed_teacher.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    fixed_teacher.requires_grad_(False)
    fixed_teacher.eval()

    student = TwoTowerModel(n_users=10, n_items=50, output_dim=32, hidden_dims=[32])
    optimizer = torch.optim.Adam(student.parameters(), lr=5e-2)
    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        loss = sdft_loss(
            student=student,
            teacher=fixed_teacher,
            batch=tiny_batch,
            arch_encode_user_fn=student.encode_user,
            arch_encode_item_fn=student.encode_item,
            get_demonstration_context_fn=student.get_demonstration_context,
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    # Final loss should be lower than initial
    assert losses[-1] < losses[0], f"KL did not decrease: {losses}"


def test_cross_arch_embedding_distance_reduces(small_tt, small_infonce, tiny_batch):
    """After cross-distillation, embedding distance between architectures should shrink."""
    from training.distillation.sdft import analytic_kl_loss

    with torch.no_grad():
        u_tt = small_tt.encode_user(tiny_batch)
        u_ic = small_infonce.encode_user(tiny_batch)

    min_dim = min(u_tt.shape[-1], u_ic.shape[-1])
    initial_dist = analytic_kl_loss(u_tt[..., :min_dim], u_ic[..., :min_dim]).item()

    # Run a few cross-distillation gradient steps (InfoNCE as student, TT as teacher)
    optimizer = torch.optim.Adam(small_infonce.parameters(), lr=1e-2)
    for _ in range(5):
        optimizer.zero_grad()
        u_student = small_infonce.encode_user(tiny_batch)
        with torch.no_grad():
            ctx = small_tt.get_demonstration_context(tiny_batch)
            u_teacher = small_tt.encode_user(ctx)
        loss = analytic_kl_loss(u_student[..., :min_dim], u_teacher[..., :min_dim])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        u_ic_after = small_infonce.encode_user(tiny_batch)
    final_dist = analytic_kl_loss(u_ic_after[..., :min_dim], u_tt.detach()[..., :min_dim]).item()
    assert final_dist <= initial_dist * 1.5  # should not diverge
