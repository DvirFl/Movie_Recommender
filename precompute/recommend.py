"""ETL Stage 8: Pre-compute recommendations for all users × genres and cold-start.

Writes results to serving.top_n_user_genre and serving.cold_start_genre.
Builds and stores FAISS indices in MinIO for all 4 scoring methods.

FIX: the personalized per-user×genre path used to run one global top-k FAISS
search and then filter that fixed set down per genre — which silently
returned fewer than top_n results whenever a genre had few matches among the
global top-k. This version builds genre-restricted FAISS sub-indices up
front and searches directly within each genre's candidate pool, so "top_n"
always means top_n within that genre (capped only by how many movies that
genre actually has).
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sqlalchemy import delete

from db.connection import get_session
from db.models import ColdStartGenre, TopNUserGenre
from etl.featurize import GENRE_VOCAB
from etl.utils import load_item_features, load_user_features
from precompute.minio_io import MinIOClient
from training.scoring import (
    FAISSIndex, LearnedScoringHead, RetrievalResult,
    ScoringMethod, score_with_learned_head,
)

logger = logging.getLogger(__name__)

TOP_N = 20


def _build_item_tensors(
    item_features: dict[int, dict],
) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (movie_ids, genre_multihot, release_year_tensor, movie_id_tensor)."""
    movie_ids = sorted(item_features.keys())
    gm = torch.tensor(
        [item_features[mid]["genre_multihot"] for mid in movie_ids], dtype=torch.float32
    )
    ry = torch.tensor(
        [float(item_features[mid].get("release_year") or 1995) for mid in movie_ids],
        dtype=torch.float32,
    )
    mid_t = torch.tensor(movie_ids, dtype=torch.long)
    return movie_ids, gm, ry, mid_t


def _genre_item_positions(item_features: dict[int, dict], movie_ids: list[int]) -> dict[str, np.ndarray]:
    """Precompute once: which positions in movie_ids/all_item_embs belong to each genre."""
    positions: dict[str, list[int]] = {g: [] for g in GENRE_VOCAB}
    for i, mid in enumerate(movie_ids):
        gm = item_features.get(mid, {}).get("genre_multihot", [])
        for g, v in zip(GENRE_VOCAB, gm):
            if v > 0:
                positions[g].append(i)
    return {g: np.array(idx, dtype=np.int64) for g, idx in positions.items()}


def precompute_recommendations(
    model: Any,                          # BaseRecommenderArchitecture
    model_name: str,
    learned_head: LearnedScoringHead | None = None,
    top_n: int = TOP_N,
    batch_size: int = 512,
) -> dict[str, int]:
    """Compute and store all recommendations for one model.

    For each scoring method, per genre:
      1. Restrict the candidate pool to items tagged with that genre.
      2. Retrieve the top_n nearest neighbors *within that pool*.
      3. Aggregate cold-start per genre.

    Returns counts of rows written per table.
    """
    model.eval()
    device = next(model.parameters()).device

    user_features = load_user_features()
    item_features = load_item_features()
    movie_ids, gm, ry, mid_t = _build_item_tensors(item_features)

    item_batch = {
        "movie_id": mid_t.to(device),
        "genre_multihot": gm.to(device),
        "release_year": ry.to(device),
    }
    with torch.no_grad():
        all_item_embs = model.encode_item(item_batch).cpu().numpy()   # (N, D)

    movie_ids_np = np.array(movie_ids)
    dim = all_item_embs.shape[1]

    # Precompute once which item positions belong to each genre — shared across all users.
    genre_positions = _genre_item_positions(item_features, movie_ids)

    minio = MinIOClient()

    # Global indices — used for the FAISS index upload (realtime/other consumers).
    faiss_indices: dict[ScoringMethod, FAISSIndex] = {}
    for method in (ScoringMethod.COSINE, ScoringMethod.DOT, ScoringMethod.L2):
        idx = FAISSIndex(method, dim)
        idx.build(all_item_embs, movie_ids_np)
        with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as f:
            tmp_path = f.name
        idx.save(tmp_path)
        minio.upload_faiss_index(tmp_path, model_name, method.value)
        faiss_indices[method] = idx

    # Genre-restricted sub-indices — this is what fixes the "fewer than top_n" bug.
    genre_faiss_indices: dict[tuple[ScoringMethod, str], FAISSIndex] = {}
    for method in (ScoringMethod.COSINE, ScoringMethod.DOT, ScoringMethod.L2):
        for genre, positions in genre_positions.items():
            if len(positions) == 0:
                continue
            idx = FAISSIndex(method, dim)
            idx.build(all_item_embs[positions], movie_ids_np[positions])
            genre_faiss_indices[(method, genre)] = idx

    # Per-user retrieval
    user_genre_rows: list[dict] = []
    all_methods = list(ScoringMethod)

    for user_id, uf in user_features.items():
        import torch as _t
        ga = list(uf["genre_affinity"])
        ga_padded = (ga + [0.0] * 20)[:20]
        user_batch = {
            "user_id": _t.tensor([user_id], dtype=_t.long).to(device),
            "genre_affinity": _t.tensor([ga_padded], dtype=_t.float32).to(device),
            "rating_count": _t.tensor([float(uf["rating_count"])], dtype=_t.float32).to(device),
            "avg_rating": _t.tensor([float(uf["avg_rating"])], dtype=_t.float32).to(device),
        }
        with torch.no_grad():
            user_emb = model.encode_user(user_batch).cpu().numpy()[0]   # (D,)

        for method in all_methods:
            for genre, positions in genre_positions.items():
                if len(positions) == 0:
                    continue

                if method == ScoringMethod.LEARNED:
                    if learned_head is None:
                        continue
                    sub_embs = all_item_embs[positions]
                    sub_ids = movie_ids_np[positions]
                    result = score_with_learned_head(
                        learned_head,
                        torch.tensor(user_emb),
                        torch.tensor(sub_embs),
                        top_k=min(top_n, len(positions)),
                    )
                    rec_movie_ids = sub_ids[result.movie_ids].tolist()
                    rec_scores = result.scores.tolist()
                else:
                    idx = genre_faiss_indices.get((method, genre))
                    if idx is None:
                        continue
                    result = idx.search(user_emb, top_k=min(top_n, len(positions)))
                    rec_movie_ids = result.movie_ids.tolist()
                    rec_scores = result.scores.tolist()

                user_genre_rows.append({
                    "user_id": user_id,
                    "genre": genre,
                    "model_name": model_name,
                    "scoring_method": method.value,
                    "movie_ids": rec_movie_ids[:top_n],
                    "scores": rec_scores[:top_n],
                })

    # Cold-start: aggregate top items per genre across all users
    cold_start_rows: list[dict] = []
    for method in all_methods:
        if method == ScoringMethod.LEARNED and learned_head is None:
            continue
        for genre, positions in genre_positions.items():
            if len(positions) == 0:
                continue
            genre_embs = all_item_embs[positions]
            scores = np.linalg.norm(genre_embs, axis=1)
            top_k_idx = np.argsort(-scores)[:top_n]
            cold_start_rows.append({
                "genre": genre,
                "model_name": model_name,
                "scoring_method": method.value,
                "movie_ids": movie_ids_np[positions][top_k_idx].tolist(),
                "scores": scores[top_k_idx].tolist(),
            })

    # Write to DB
    with get_session() as session:
        session.execute(
            delete(TopNUserGenre).where(TopNUserGenre.model_name == model_name)
        )
        session.execute(
            delete(ColdStartGenre).where(ColdStartGenre.model_name == model_name)
        )
        session.bulk_insert_mappings(TopNUserGenre, user_genre_rows)      # type: ignore
        session.bulk_insert_mappings(ColdStartGenre, cold_start_rows)     # type: ignore

    logger.info(
        "Precomputed %d user×genre rows and %d cold-start rows for model '%s'.",
        len(user_genre_rows), len(cold_start_rows), model_name,
    )
    return {
        "top_n_user_genre": len(user_genre_rows),
        "cold_start_genre": len(cold_start_rows),
    }