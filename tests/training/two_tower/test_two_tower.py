"""Tests: TwoTowerModel forward pass shapes, loss, FAISS index."""
import pytest
import torch
import numpy as np

from training.two_tower.towers import TwoTowerModel, UserTower, ItemTower
from training.two_tower.losses import TimedecayMSELoss
from training.scoring import FAISSIndex, ScoringMethod, LearnedScoringHead, score_with_learned_head


@pytest.fixture
def model():
    return TwoTowerModel(n_users=10, n_items=50, output_dim=32, hidden_dims=[32])


def test_user_tower_output_shape(tiny_batch):
    tower = UserTower(n_users=10, n_genres=20, embed_dim=16, hidden_dims=[32], output_dim=32)
    out = tower(tiny_batch)
    assert out.shape == (4, 32)


def test_item_tower_output_shape(tiny_batch):
    tower = ItemTower(n_items=50, n_genres=20, embed_dim=16, hidden_dims=[32], output_dim=32)
    out = tower(tiny_batch)
    assert out.shape == (4, 32)


def test_two_tower_forward(model, tiny_batch):
    u, i = model(tiny_batch, tiny_batch)
    assert u.shape == (4, 32)
    assert i.shape == (4, 32)


def test_two_tower_encode_user_no_nan(model, tiny_batch):
    out = model.encode_user(tiny_batch)
    assert not torch.isnan(out).any()


def test_two_tower_encode_item_no_nan(model, tiny_batch):
    out = model.encode_item(tiny_batch)
    assert not torch.isnan(out).any()


def test_mse_loss_range(tiny_batch):
    loss_fn = TimedecayMSELoss()
    u = torch.randn(4, 32)
    i = torch.randn(4, 32)
    loss = loss_fn(u, i, tiny_batch["rating"], tiny_batch["weight"])
    assert 0.0 <= loss.item() <= 10.0


def test_mse_loss_backward(tiny_batch):
    model = TwoTowerModel(n_users=10, n_items=50, output_dim=32, hidden_dims=[32])
    loss_fn = TimedecayMSELoss()
    u = model.encode_user(tiny_batch)
    i = model.encode_item(tiny_batch)
    loss = loss_fn(u, i, tiny_batch["rating"], tiny_batch["weight"])
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any()


def test_faiss_index_build_and_search():
    dim = 32
    n_items = 100
    item_embs = np.random.randn(n_items, dim).astype(np.float32)
    item_ids = np.arange(n_items)
    user_emb = np.random.randn(dim).astype(np.float32)

    for method in (ScoringMethod.COSINE, ScoringMethod.DOT, ScoringMethod.L2):
        idx = FAISSIndex(method, dim)
        idx.build(item_embs, item_ids)
        result = idx.search(user_emb, top_k=10)
        assert len(result.movie_ids) == 10
        assert len(result.scores) == 10
        assert all(0 <= mid < n_items for mid in result.movie_ids)


def test_learned_scoring_head_shape():
    head = LearnedScoringHead(emb_dim=32)
    u = torch.randn(8, 32)
    i = torch.randn(8, 32)
    scores = head(u, i)
    assert scores.shape == (8,)


def test_score_with_learned_head_top_k():
    head = LearnedScoringHead(emb_dim=32)
    user_emb = torch.randn(32)
    all_items = torch.randn(50, 32)
    result = score_with_learned_head(head, user_emb, all_items, top_k=5)
    assert len(result.movie_ids) == 5
    assert len(result.scores) == 5


def test_compatible_losses_attribute():
    assert "TimedecayMSELoss" in TwoTowerModel.compatible_losses
