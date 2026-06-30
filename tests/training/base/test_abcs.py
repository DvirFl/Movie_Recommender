"""Tests: ABC contract compliance for TwoTowerModel and InfoNCEModel."""
import pytest
import torch

from training.base.architecture import BaseRecommenderArchitecture
from training.base.loss import BaseRecommenderLoss
from training.two_tower.towers import TwoTowerModel
from training.two_tower.losses import TimedecayMSELoss
from training.infonce.encoders import InfoNCEModel
from training.infonce.losses import TimedecayInfoNCELoss


@pytest.mark.parametrize("arch_cls,kwargs", [
    (TwoTowerModel, {"n_users": 10, "n_items": 50, "output_dim": 32, "hidden_dims": [32]}),
    (InfoNCEModel, {"n_users": 10, "n_items": 50, "output_dim": 32, "hidden_dims": [32], "n_layers": 1}),
])
def test_architecture_is_subclass(arch_cls, kwargs):
    arch = arch_cls(**kwargs)
    assert isinstance(arch, BaseRecommenderArchitecture)


@pytest.mark.parametrize("arch_cls,kwargs", [
    (TwoTowerModel, {"n_users": 10, "n_items": 50, "output_dim": 32, "hidden_dims": [32]}),
    (InfoNCEModel, {"n_users": 10, "n_items": 50, "output_dim": 32, "hidden_dims": [32], "n_layers": 1}),
])
def test_encode_user_shape(arch_cls, kwargs, tiny_batch):
    arch = arch_cls(**kwargs)
    out = arch.encode_user(tiny_batch)
    assert out.shape == (4, kwargs["output_dim"])
    assert out.dtype == torch.float32


@pytest.mark.parametrize("arch_cls,kwargs", [
    (TwoTowerModel, {"n_users": 10, "n_items": 50, "output_dim": 32, "hidden_dims": [32]}),
    (InfoNCEModel, {"n_users": 10, "n_items": 50, "output_dim": 32, "hidden_dims": [32], "n_layers": 1}),
])
def test_encode_item_shape(arch_cls, kwargs, tiny_batch):
    arch = arch_cls(**kwargs)
    out = arch.encode_item(tiny_batch)
    assert out.shape == (4, kwargs["output_dim"])


@pytest.mark.parametrize("arch_cls,kwargs", [
    (TwoTowerModel, {"n_users": 10, "n_items": 50, "output_dim": 32, "hidden_dims": [32]}),
    (InfoNCEModel, {"n_users": 10, "n_items": 50, "output_dim": 32, "hidden_dims": [32], "n_layers": 1}),
])
def test_supports_sdft_returns_bool(arch_cls, kwargs):
    arch = arch_cls(**kwargs)
    assert isinstance(arch.supports_sdft(), bool)


@pytest.mark.parametrize("arch_cls,kwargs", [
    (TwoTowerModel, {"n_users": 10, "n_items": 50, "output_dim": 32, "hidden_dims": [32]}),
    (InfoNCEModel, {"n_users": 10, "n_items": 50, "output_dim": 32, "hidden_dims": [32], "n_layers": 1}),
])
def test_get_demonstration_context_returns_dict(arch_cls, kwargs, tiny_batch):
    arch = arch_cls(**kwargs)
    ctx = arch.get_demonstration_context(tiny_batch)
    assert isinstance(ctx, dict)
    assert "user_id" in ctx


@pytest.mark.parametrize("arch_cls,kwargs", [
    (TwoTowerModel, {"n_users": 10, "n_items": 50, "output_dim": 32, "hidden_dims": [32]}),
    (InfoNCEModel, {"n_users": 10, "n_items": 50, "output_dim": 32, "hidden_dims": [32], "n_layers": 1}),
])
def test_hparam_search_space_not_empty(arch_cls, kwargs):
    arch = arch_cls(**kwargs)
    space = arch.hparam_search_space()
    assert isinstance(space, dict)
    assert len(space) > 0


@pytest.mark.parametrize("loss_cls", [TimedecayMSELoss, TimedecayInfoNCELoss])
def test_loss_is_subclass(loss_cls):
    loss = loss_cls()
    assert isinstance(loss, BaseRecommenderLoss)


@pytest.mark.parametrize("loss_cls", [TimedecayMSELoss, TimedecayInfoNCELoss])
def test_loss_forward_scalar(loss_cls, tiny_batch):
    loss = loss_cls()
    B, D = 4, 32
    u = torch.randn(B, D)
    i = torch.randn(B, D)
    out = loss(u, i, tiny_batch["rating"], tiny_batch["weight"])
    assert out.shape == ()
    assert out.item() >= 0.0


@pytest.mark.parametrize("loss_cls", [TimedecayMSELoss, TimedecayInfoNCELoss])
def test_loss_hparam_space_not_empty(loss_cls):
    space = loss_cls().hparam_search_space()
    assert isinstance(space, dict)
    assert len(space) > 0


def test_name_attribute_set():
    assert TwoTowerModel.name == "TwoTower"
    assert InfoNCEModel.name == "InfoNCEEncoder"
    assert TimedecayMSELoss.name == "TimedecayMSELoss"
    assert TimedecayInfoNCELoss.name == "TimedecayInfoNCELoss"
