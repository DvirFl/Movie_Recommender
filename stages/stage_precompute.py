"""Stage 9 — Precompute: ANN retrieval for all models × scoring methods → serving.* tables."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PrecomputeResult:
    # Maps "ArchName_LossName" -> {"top_n_user_genre": N, "cold_start_genre": N}
    counts: dict[str, dict[str, int]] = field(default_factory=dict)


def run(
    losses: list[str] | str = "all",
    architectures: list[str] | str = "all",
    top_n: int = 20,
    upload_faiss: bool = True,
) -> PrecomputeResult:
    """Pre-compute Top-N per user×genre and cold-start recommendations.

    For each model, builds FAISS indices for cosine / dot / L2 scoring and
    runs the learned-head scoring, then writes results to serving.* tables.

    Args:
        losses:        filter to specific loss names, or "all".
        architectures: filter to specific arch names, or "all".
        top_n:         number of recommendations to store per user×genre.
        upload_faiss:  whether to upload FAISS index files to MinIO.

    Returns:
        PrecomputeResult with row counts written per model.
    """
    from training.registry import ComponentRegistry
    from training.device_utils import get_device
    from precompute.recommend import precompute_recommendations
    from etl.utils import load_user_features, load_item_features

    registry = ComponentRegistry()
    arch_filter = None if architectures == "all" else (
        architectures if isinstance(architectures, list) else [architectures]
    )
    loss_filter = None if losses == "all" else (
        losses if isinstance(losses, list) else [losses]
    )
    combos = registry.filter_combinations(
        architecture_names=arch_filter,
        loss_names=loss_filter,
    )

    uf  = load_user_features()
    itf = load_item_features()
    n_users = max(uf.keys()) + 1
    n_items = max(itf.keys()) + 1
    device  = get_device()

    result = PrecomputeResult()

    for arch_entry, loss_entry in combos:
        key = f"{arch_entry.name}_{loss_entry.name}"
        logger.info("[precompute] Running %s ...", key)

        arch = arch_entry.cls(n_users=n_users, n_items=n_items).to(device)
        counts = precompute_recommendations(
            model=arch,
            model_name=key,
            top_n=top_n,
        )
        result.counts[key] = counts
        logger.info("[precompute] %s → %s", key, counts)

    return result
