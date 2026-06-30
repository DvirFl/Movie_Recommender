"""Stage 3 — Featurize: compute user + item features and write to features.* schema."""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FeaturizeResult:
    n_users: int = 0
    n_items: int = 0


def run(decay_lambda: float = 0.1) -> FeaturizeResult:
    """Compute time-decayed user history, genre multi-hot, tag TF-IDF.

    Args:
        decay_lambda: exponential decay rate λ for time-decay weighting.
                      Higher → more weight on recent interactions.

    Returns:
        FeaturizeResult with counts of processed users and items.
    """
    from etl.featurize import featurize_all

    logger.info("[featurize] Computing features (decay_lambda=%.3f)...", decay_lambda)
    counts = featurize_all(decay_lambda=decay_lambda)
    result = FeaturizeResult(n_users=counts["users"], n_items=counts["items"])
    logger.info("[featurize] Done — users: %d  items: %d", result.n_users, result.n_items)
    return result
