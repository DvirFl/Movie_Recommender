"""Stage 4 — Split: temporal train/val/test split with no leakage."""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    train: int = 0
    val: int = 0
    test: int = 0


def run(train_frac: float = 0.8, val_frac: float = 0.1) -> SplitResult:
    """Temporal split: sort by timestamp, assign first 80 % train, next 10 % val, rest test.

    Args:
        train_frac: fraction of interactions for training.
        val_frac:   fraction for validation; remainder goes to test.

    Returns:
        SplitResult with row counts per split.
    """
    from etl.split import split_ratings

    logger.info(
        "[split] Splitting — train: %.0f%%  val: %.0f%%  test: %.0f%%",
        train_frac * 100, val_frac * 100, (1 - train_frac - val_frac) * 100,
    )
    counts = split_ratings(train_frac=train_frac, val_frac=val_frac)
    result = SplitResult(
        train=counts.get("train", 0),
        val=counts.get("val", 0),
        test=counts.get("test", 0),
    )
    logger.info("[split] Done — %s", counts)
    return result
