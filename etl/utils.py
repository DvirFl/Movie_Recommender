"""Shared ETL utilities."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sqlalchemy import select, text

from db.connection import get_session
from db.models import RawRating, SplitIndex, UserFeature, ItemFeature

logger = logging.getLogger(__name__)


def load_split_dataframes() -> dict[str, pd.DataFrame]:
    """Load train/val/test rating DataFrames joined with split indices."""
    with get_session() as session:
        df = pd.read_sql(
            text("""
                SELECT r.id, r.user_id, r.movie_id, r.rating, r.timestamp, s.split
                FROM raw.ratings r
                JOIN features.split_indices s ON r.id = s.rating_id
            """),
            session.bind,
        )
    splits = {}
    for split_name in ("train", "val", "test"):
        splits[split_name] = df[df["split"] == split_name].reset_index(drop=True)
    return splits


def load_user_features() -> dict[int, dict]:
    """Load all user features from features.user_features as a dict keyed by user_id."""
    with get_session() as session:
        rows = session.execute(select(UserFeature)).scalars().all()
        return {
            row.user_id: {
                "decayed_history": row.decayed_history,
                "genre_affinity": row.genre_affinity,
                "rating_count": row.rating_count,
                "avg_rating": row.avg_rating,
            }
            for row in rows
        }


def load_item_features() -> dict[int, dict]:
    """Load all item features from features.item_features as a dict keyed by movie_id."""
    with get_session() as session:
        rows = session.execute(select(ItemFeature)).scalars().all()
        return {
            row.movie_id: {
                "genre_multihot": row.genre_multihot,
                "tag_tfidf": row.tag_tfidf,
                "release_year": row.release_year,
            }
            for row in rows
        }


def count_raw_rows() -> dict[str, int]:
    """Return row counts for each raw table (used by /viz pipeline stage sizes)."""
    with get_session() as session:
        return {
            "raw.ratings": session.scalar(text("SELECT COUNT(*) FROM raw.ratings")),
            "raw.movies": session.scalar(text("SELECT COUNT(*) FROM raw.movies")),
            "raw.tags": session.scalar(text("SELECT COUNT(*) FROM raw.tags")),
            "raw.links": session.scalar(text("SELECT COUNT(*) FROM raw.links")),
            "raw.genome_scores": session.scalar(text("SELECT COUNT(*) FROM raw.genome_scores")),
            "features.user_features": session.scalar(text("SELECT COUNT(*) FROM features.user_features")),
            "features.item_features": session.scalar(text("SELECT COUNT(*) FROM features.item_features")),
            "features.split_indices": session.scalar(text("SELECT COUNT(*) FROM features.split_indices")),
        }
