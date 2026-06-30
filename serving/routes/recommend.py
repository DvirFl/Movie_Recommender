"""GET/POST /recommend — serve pre-computed recommendations."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from db.connection import get_session
from db.models import ColdStartGenre, TopNUserGenre
from serving.schemas import RecommendRequest, RecommendResponse

router = APIRouter(prefix="/recommend", tags=["recommend"])


@router.post("", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    with get_session() as session:
        if req.user_id is None:
            # Cold-start path
            stmt = (
                select(ColdStartGenre)
                .where(
                    ColdStartGenre.model_name == req.model_name,
                    ColdStartGenre.scoring_method == req.scoring_method,
                )
            )
            if req.genre:
                stmt = stmt.where(ColdStartGenre.genre == req.genre)
            row = session.scalars(stmt).first()
            if row is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No cold-start recommendations found for model '{req.model_name}' "
                           f"scoring '{req.scoring_method}' genre '{req.genre}'.",
                )
            return RecommendResponse(
                user_id=None,
                genre=req.genre,
                model_name=req.model_name,
                scoring_method=req.scoring_method,
                movie_ids=row.movie_ids[: req.top_n],
                scores=row.scores[: req.top_n],
                source="precomputed",
            )

        # Personalized path
        stmt = (
            select(TopNUserGenre)
            .where(
                TopNUserGenre.user_id == req.user_id,
                TopNUserGenre.model_name == req.model_name,
                TopNUserGenre.scoring_method == req.scoring_method,
            )
        )
        if req.genre:
            stmt = stmt.where(TopNUserGenre.genre == req.genre)

        row = session.scalars(stmt).first()
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendations found for user_id={req.user_id}.",
            )
        return RecommendResponse(
            user_id=req.user_id,
            genre=req.genre or row.genre,
            model_name=req.model_name,
            scoring_method=req.scoring_method,
            movie_ids=row.movie_ids[: req.top_n],
            scores=row.scores[: req.top_n],
            source="precomputed",
        )
