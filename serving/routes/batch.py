"""POST /batch — batch recommendation scoring for a list of user_ids."""
from __future__ import annotations

from fastapi import APIRouter

from serving.routes.recommend import recommend
from serving.schemas import BatchRequest, BatchResponse, RecommendRequest, RecommendResponse

router = APIRouter(prefix="/batch", tags=["batch"])


@router.post("", response_model=BatchResponse)
def batch_recommend(req: BatchRequest) -> BatchResponse:
    results: list[RecommendResponse] = []
    for uid in req.user_ids:
        try:
            result = recommend(
                RecommendRequest(
                    user_id=uid,
                    genre=req.genre,
                    model_name=req.model_name,
                    scoring_method=req.scoring_method,
                    top_n=req.top_n,
                )
            )
        except Exception:
            # Fall back to cold-start for users with no precomputed recs
            result = recommend(
                RecommendRequest(
                    user_id=None,
                    genre=req.genre,
                    model_name=req.model_name,
                    scoring_method=req.scoring_method,
                    top_n=req.top_n,
                )
            )
        results.append(result)
    return BatchResponse(results=results)
