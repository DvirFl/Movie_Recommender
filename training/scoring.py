"""Scoring layer: 4 retrieval methods over trained embeddings.

Methods:
  1. cosine  — FAISS IndexFlatIP on L2-normalised embeddings
  2. dot     — FAISS IndexFlatIP on raw embeddings
  3. l2      — FAISS IndexFlatL2
  4. learned — concat(user_emb, item_emb) → MLP → scalar, then exact sort
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class ScoringMethod(str, Enum):
    COSINE = "cosine"
    DOT = "dot"
    L2 = "l2"
    LEARNED = "learned"


class RetrievalResult(NamedTuple):
    movie_ids: np.ndarray   # (n_results,) indices into item_ids array
    scores: np.ndarray      # (n_results,)


class LearnedScoringHead(nn.Module):
    """Small MLP: concat(user_emb, item_emb) → scalar score."""

    def __init__(self, emb_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([user_emb, item_emb], dim=-1)).squeeze(-1)


class FAISSIndex:
    """Wraps a FAISS index for a given scoring method."""

    def __init__(self, method: ScoringMethod, dim: int) -> None:
        if not _HAS_FAISS:
            raise ImportError("faiss is not installed.")
        self.method = method
        self.dim = dim
        self._item_ids: np.ndarray | None = None

        if method == ScoringMethod.L2:
            self._index = faiss.IndexFlatL2(dim)
        else:
            # Both cosine (normalised) and dot use IndexFlatIP
            self._index = faiss.IndexFlatIP(dim)

        # Use GPU index if CUDA is available
        if torch.cuda.is_available():
            try:
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
            except Exception:
                pass  # fall back to CPU

    def build(self, item_embs: np.ndarray, item_ids: np.ndarray) -> None:
        """Add item embeddings to the index.

        Args:
            item_embs: (N, D) float32 array of item embeddings.
            item_ids:  (N,) array of movie_id values.
        """
        embs = item_embs.astype(np.float32)
        if self.method == ScoringMethod.COSINE:
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs = embs / np.clip(norms, 1e-8, None)
        self._index.add(embs)
        self._item_ids = item_ids.copy()

    def search(self, user_emb: np.ndarray, top_k: int = 10) -> RetrievalResult:
        """Retrieve top-k items for a user embedding.

        Args:
            user_emb: (D,) or (1, D) float32 user embedding.
            top_k:    number of results to return.

        Returns:
            RetrievalResult with movie_ids and scores.
        """
        q = user_emb.astype(np.float32).reshape(1, -1)
        if self.method == ScoringMethod.COSINE:
            q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-8, None)
        scores, indices = self._index.search(q, top_k)
        return RetrievalResult(
            movie_ids=self._item_ids[indices[0]],
            scores=scores[0],
        )

    def save(self, path: str | Path) -> None:
        cpu_index = faiss.index_gpu_to_cpu(self._index) if torch.cuda.is_available() else self._index
        faiss.write_index(cpu_index, str(path))
        np.save(str(path) + ".ids.npy", self._item_ids)

    @classmethod
    def load(cls, path: str | Path, method: ScoringMethod, dim: int) -> "FAISSIndex":
        obj = cls.__new__(cls)
        obj.method = method
        obj.dim = dim
        obj._index = faiss.read_index(str(path))
        obj._item_ids = np.load(str(path) + ".ids.npy")
        return obj


def score_with_learned_head(
    head: LearnedScoringHead,
    user_emb: torch.Tensor,        # (1, D) or (D,)
    all_item_embs: torch.Tensor,   # (N, D)
    top_k: int = 10,
) -> RetrievalResult:
    """Score all items with the learned head and return top-k."""
    head.eval()
    with torch.no_grad():
        u = user_emb.unsqueeze(0) if user_emb.dim() == 1 else user_emb  # (1, D)
        u_expanded = u.expand(all_item_embs.size(0), -1)                 # (N, D)
        scores = head(u_expanded, all_item_embs)                          # (N,)
        top_scores, top_idx = scores.topk(min(top_k, len(scores)))
    return RetrievalResult(
        movie_ids=top_idx.cpu().numpy(),
        scores=top_scores.cpu().numpy(),
    )
