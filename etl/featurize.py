"""ETL Stage 3: Feature engineering.

Computes:
  - User features: time-decayed interaction history, genre affinity, rating stats
  - Item features: genre multi-hot encoding, tag TF-IDF, release year
Writes results to features.user_features and features.item_features.
"""
from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from typing import Any

import pandas as pd
from sqlalchemy import select, text

from db.connection import get_session
from db.models import ItemFeature, RawMovie, RawRating, RawTag, UserFeature

logger = logging.getLogger(__name__)

GENRE_VOCAB = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    "(no genres listed)",
]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRE_VOCAB)}


def _extract_year(title: str) -> int | None:
    match = re.search(r"\((\d{4})\)", title)
    return int(match.group(1)) if match else None


def _time_decay_weight(timestamp: int, t_max: int, t_range: int, lam: float = 0.1) -> float:
    if t_range == 0:
        return 1.0
    norm = (timestamp - (t_max - t_range)) / t_range
    return math.exp(-lam * (1.0 - norm))


def compute_user_features(decay_lambda: float = 0.1) -> int:
    """Compute and upsert user features. Returns number of users processed."""
    with get_session() as session:
        ratings_df = pd.read_sql(
            select(RawRating.user_id, RawRating.movie_id, RawRating.rating, RawRating.timestamp),
            session.bind,
        )
        movies_df = pd.read_sql(
            select(RawMovie.movie_id, RawMovie.genres),
            session.bind,
        )

    t_max = int(ratings_df["timestamp"].max())
    t_min = int(ratings_df["timestamp"].min())
    t_range = max(t_max - t_min, 1)

    movie_genres: dict[int, list[str]] = {}
    for _, row in movies_df.iterrows():
        movie_genres[int(row["movie_id"])] = row["genres"].split("|")

    user_features = []
    for user_id, group in ratings_df.groupby("user_id"):
        decayed_history: dict[int, float] = {}
        genre_affinity: dict[str, float] = defaultdict(float)
        total_weight = 0.0

        for _, row in group.iterrows():
            w = _time_decay_weight(int(row["timestamp"]), t_max, t_range, decay_lambda)
            mid = int(row["movie_id"])
            decayed_history[mid] = float(w * row["rating"])
            for g in movie_genres.get(mid, []):
                genre_affinity[g] += w
            total_weight += w

        # Normalise genre affinity
        if total_weight > 0:
            genre_affinity = {k: v / total_weight for k, v in genre_affinity.items()}

        genre_affinity_vec = [genre_affinity.get(g, 0.0) for g in GENRE_VOCAB]

        user_features.append({
            "user_id": int(user_id),
            "decayed_history": decayed_history,
            "genre_affinity": genre_affinity_vec,
            "rating_count": int(len(group)),
            "avg_rating": float(group["rating"].mean()),
        })

    with get_session() as session:
        for uf in user_features:
            existing = session.get(UserFeature, uf["user_id"])
            if existing:
                for k, v in uf.items():
                    setattr(existing, k, v)
            else:
                session.add(UserFeature(**uf))

    logger.info("Computed features for %d users.", len(user_features))
    return len(user_features)


def compute_item_features() -> int:
    """Compute and upsert item features. Returns number of items processed."""
    with get_session() as session:
        movies_df = pd.read_sql(
            select(RawMovie.movie_id, RawMovie.title, RawMovie.genres),
            session.bind,
        )
        tags_df = pd.read_sql(
            select(RawTag.movie_id, RawTag.tag),
            session.bind,
        )

    # TF-IDF for tags (simple implementation)
    from sklearn.feature_extraction.text import TfidfVectorizer
    movie_tag_docs: dict[int, str] = defaultdict(str)
    for _, row in tags_df.iterrows():
        movie_tag_docs[int(row["movie_id"])] += " " + str(row["tag"])

    all_movie_ids = [int(r["movie_id"]) for _, r in movies_df.iterrows()]
    docs = [movie_tag_docs.get(mid, "") for mid in all_movie_ids]

    vectorizer = TfidfVectorizer(max_features=100, min_df=1)
    if any(d.strip() for d in docs):
        tfidf_matrix = vectorizer.fit_transform(docs)
        vocab = vectorizer.get_feature_names_out().tolist()
    else:
        tfidf_matrix = None
        vocab = []

    item_features = []
    for i, (_, row) in enumerate(movies_df.iterrows()):
        mid = int(row["movie_id"])
        genres = row["genres"].split("|")
        multihot = [0.0] * len(GENRE_VOCAB)
        for g in genres:
            if g in GENRE_TO_IDX:
                multihot[GENRE_TO_IDX[g]] = 1.0

        tfidf_dict: dict[str, float] = {}
        if tfidf_matrix is not None:
            row_vec = tfidf_matrix[i].toarray()[0]
            tfidf_dict = {vocab[j]: float(row_vec[j]) for j in range(len(vocab)) if row_vec[j] > 0}

        item_features.append({
            "movie_id": mid,
            "genre_multihot": multihot,
            "tag_tfidf": tfidf_dict,
            "release_year": _extract_year(str(row["title"])),
        })

    with get_session() as session:
        for itf in item_features:
            existing = session.get(ItemFeature, itf["movie_id"])
            if existing:
                for k, v in itf.items():
                    setattr(existing, k, v)
            else:
                session.add(ItemFeature(**itf))

    logger.info("Computed features for %d items.", len(item_features))
    return len(item_features)


def featurize_all(decay_lambda: float = 0.1) -> dict[str, int]:
    return {
        "users": compute_user_features(decay_lambda),
        "items": compute_item_features(),
    }
