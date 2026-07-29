"""POST /recommend — fan out across every registered architecture, with per-movie explanations."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from db.connection import get_session
from db.models import ColdStartGenre, ItemFeature, TopNUserGenre, UserFeature
from etl.featurize import GENRE_VOCAB
from serving.schemas import (
    ArchitectureResult, MultiArchRecommendRequest, MultiArchRecommendResponse,
    RecommendedMovie, RecommendRequest, RecommendResponse,
)
from training.registry import ComponentRegistry

router = APIRouter(prefix="/recommend", tags=["recommend"])


def _fetch_row(session, user_id: int | None, genre: str | None, model_name: str, scoring_method: str):
    """Look up the precomputed row for one model. Returns (row, source) or (None, None)."""
    if user_id is None:
        stmt = select(ColdStartGenre).where(
            ColdStartGenre.model_name == model_name,
            ColdStartGenre.scoring_method == scoring_method,
        )
        if genre:
            stmt = stmt.where(ColdStartGenre.genre == genre)
        return session.scalars(stmt).first(), "precomputed"

    stmt = select(TopNUserGenre).where(
        TopNUserGenre.user_id == user_id,
        TopNUserGenre.model_name == model_name,
        TopNUserGenre.scoring_method == scoring_method,
    )
    if genre:
        stmt = stmt.where(TopNUserGenre.genre == genre)
    return session.scalars(stmt).first(), "precomputed"


def _top_genres(vec: list[float], k: int = 2) -> set[str]:
    """Highest-affinity genres from a user's genre_affinity vector (ordered per GENRE_VOCAB)."""
    if not vec:
        return set()
    pairs = sorted(zip(GENRE_VOCAB, vec), key=lambda p: p[1], reverse=True)
    return {g for g, v in pairs[:k] if v > 0}


def _explain_one(
    architecture: str,
    scoring_method: str,
    score: float,
    top_user_genres: set[str],
    movie_genre_multihot: list[float] | None,
    cold_start_genre: str | None,
) -> str:
    movie_genres = (
        {g for g, v in zip(GENRE_VOCAB, movie_genre_multihot) if v > 0}
        if movie_genre_multihot else set()
    )
    overlap = sorted(top_user_genres & movie_genres)

    if overlap:
        return (
            f"{architecture} ranked this {score:.3f} ({scoring_method}) — "
            f"matches your affinity for {', '.join(overlap)}."
        )
    if cold_start_genre:
        return (
            f"{architecture} popularity pick for {cold_start_genre} among new users "
            f"({score:.3f} {scoring_method})."
        )
    return (
        f"{architecture} placed this close to your interaction history in embedding "
        f"space ({score:.3f} {scoring_method})."
    )


@router.post("", response_model=MultiArchRecommendResponse)
def recommend_all_architectures(req: MultiArchRecommendRequest) -> MultiArchRecommendResponse:
    """Return recommendations from every enabled architecture×loss combination."""
    registry = ComponentRegistry()
    combos = registry.get_enabled_combinations()
    if req.architectures != "all":
        combos = [(a, l) for a, l in combos if a.name in req.architectures]
    if not combos:
        raise HTTPException(status_code=404, detail="No enabled architectures match the request.")

    results: list[ArchitectureResult] = []

    with get_session() as session:
        top_user_genres: set[str] = set()
        if req.user_id is not None:
            uf = session.get(UserFeature, req.user_id)
            top_user_genres = _top_genres(uf.genre_affinity if uf else [])

        for arch_entry, loss_entry in combos:
            model_name = f"{arch_entry.name}_{loss_entry.name}"
            row, source = _fetch_row(session, req.user_id, req.genre, model_name, req.scoring_method)
            if row is None:
                continue  # this arch has no precomputed rows for the request — skip, don't fail the batch

            movie_ids = row.movie_ids[: req.top_n]
            scores = row.scores[: req.top_n]

            item_rows = session.execute(
                select(ItemFeature).where(ItemFeature.movie_id.in_(movie_ids))
            ).scalars().all()
            item_gm = {r.movie_id: r.genre_multihot for r in item_rows}

            recs = [
                RecommendedMovie(
                    movie_id=mid,
                    score=score,
                    explanation=_explain_one(
                        architecture=arch_entry.name,
                        scoring_method=req.scoring_method,
                        score=score,
                        top_user_genres=top_user_genres,
                        movie_genre_multihot=item_gm.get(mid),
                        cold_start_genre=req.genre if req.user_id is None else None,
                    ),
                )
                for mid, score in zip(movie_ids, scores)
            ]

            results.append(ArchitectureResult(
                model_name=model_name,
                architecture=arch_entry.name,
                loss=loss_entry.name,
                scoring_method=req.scoring_method,
                source=source,
                recommendations=recs,
            ))

    if not results:
        raise HTTPException(status_code=404, detail="No precomputed recommendations found for this request.")

    return MultiArchRecommendResponse(
        user_id=req.user_id, genre=req.genre, scoring_method=req.scoring_method, results=results,
    )


@router.post("/single", response_model=RecommendResponse)
def recommend_single(req: RecommendRequest) -> RecommendResponse:
    """Original single-model contract — kept for internal reuse (batch, ab_test) and direct callers."""
    with get_session() as session:
        row, source = _fetch_row(session, req.user_id, req.genre, req.model_name, req.scoring_method)
        if row is None:
            raise HTTPException(status_code=404, detail=f"No recommendations found for model '{req.model_name}'.")
        return RecommendResponse(
            user_id=req.user_id,
            genre=req.genre or getattr(row, "genre", None),
            model_name=req.model_name,
            scoring_method=req.scoring_method,
            movie_ids=row.movie_ids[: req.top_n],
            scores=row.scores[: req.top_n],
            source=source,
        )


def _recommend_single(req: RecommendRequest) -> RecommendResponse:
    """Internal helper used by batch.py / ab_test.py — same as the /recommend/single route."""
    return recommend_single(req)
