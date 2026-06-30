"""ETL Stage 4: Temporal train/val/test split with no leakage.

Strategy: sort all ratings by timestamp globally, then assign:
  - train: first 80%
  - val:   next 10%
  - test:  last 10%

This ensures no future interactions leak into training.
Writes split indices to features.split_indices.
"""
from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import select, delete

from db.connection import get_session
from db.models import RawRating, SplitIndex

logger = logging.getLogger(__name__)


def split_ratings(
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> dict[str, int]:
    """Compute temporal split and write to features.split_indices.

    Args:
        train_frac: fraction of data for training.
        val_frac:   fraction for validation; remainder goes to test.

    Returns:
        dict with counts per split.
    """
    assert train_frac + val_frac < 1.0, "train + val fractions must be < 1.0"

    with get_session() as session:
        df = pd.read_sql(
            select(RawRating.id, RawRating.timestamp).order_by(RawRating.timestamp),
            session.bind,
        )

    n = len(df)
    if n == 0:
        raise ValueError("raw.ratings is empty — run ingest first.")

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    splits = (
        ["train"] * n_train
        + ["val"] * n_val
        + ["test"] * (n - n_train - n_val)
    )
    df["split"] = splits

    records = [
        {"rating_id": int(row["id"]), "split": row["split"]}
        for _, row in df.iterrows()
    ]

    with get_session() as session:
        # Clear existing split indices
        session.execute(delete(SplitIndex))
        session.bulk_insert_mappings(SplitIndex, records)  # type: ignore[arg-type]

    counts = df["split"].value_counts().to_dict()
    logger.info("Split complete: %s", counts)
    return counts
