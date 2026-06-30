"""Tests: InfoNCEModel and TimedecayInfoNCELoss."""
import pytest
import torch

from training.infonce.encoders import InfoNCEModel, UserEncoder, ItemEncoder
from training.infonce.losses import TimedecayInfoNCELoss


@pytest.fixture
def model():
    return InfoNCEModel(n_users=10, n_items=50, output_dim=32, hidden_dims=[32], n_layers=1)


def test_user_encoder_shape(tiny_batch):
    enc = UserEncoder(n_users=10, n_items=50, n_genres=20, embed_dim=16,
                      n_heads=2, n_layers=1, output_dim=32)
    out = enc(tiny_batch)
    assert out.shape == (4, 32)


def test_item_encoder_shape(tiny_batch):
    enc = ItemEncoder(n_items=50, n_genres=20, embed_dim=16, hidden_dims=[32], output_dim=32)
    out = enc(tiny_batch)
    assert out.shape == (4, 32)


def test_infonce_model_forward(model, tiny_batch):
    u, i = model(tiny_batch, tiny_batch)
    assert u.shape == (4, 32)
    assert i.shape == (4, 32)


def test_infonce_no_nan(model, tiny_batch):
    u = model.encode_user(tiny_batch)
    i = model.encode_item(tiny_batch)
    assert not torch.isnan(u).any()
    assert not torch.isnan(i).any()


def test_infonce_loss_positive(tiny_batch):
    loss_fn = TimedecayInfoNCELoss(temperature=0.1)
    u = torch.randn(4, 32)
    i = torch.randn(4, 32)
    loss = loss_fn(u, i, tiny_batch["rating"], tiny_batch["weight"])
    assert loss.item() > 0.0


def test_infonce_loss_backward(model, tiny_batch):
    loss_fn = TimedecayInfoNCELoss()
    u = model.encode_user(tiny_batch)
    i = model.encode_item(tiny_batch)
    loss = loss_fn(u, i, tiny_batch["rating"], tiny_batch["weight"])
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any()


def test_infonce_compatible_losses():
    assert "TimedecayInfoNCELoss" in InfoNCEModel.compatible_losses


def test_infonce_supports_sdft(model):
    assert model.supports_sdft() is True


def test_infonce_demo_context_keys(model, tiny_batch):
    ctx = model.get_demonstration_context(tiny_batch)
    assert "user_id" in ctx
    assert "genre_affinity" in ctx
