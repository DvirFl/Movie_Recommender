"""POST /ab_test — deterministic A/B routing between two models."""
from __future__ import annotations

from fastapi import APIRouter

from serving.routes.recommend import recommend
from serving.schemas import ABTestRequest, ABTestResponse, RecommendRequest

router = APIRouter(prefix="/ab_test", tags=["ab_test"])

# Default model pair for A/B testing
MODEL_A = "TwoTower_TimedecayMSELoss"
MODEL_B = "InfoNCEEncoder_TimedecayInfoNCELoss"


def _assign_model(user_id: int) -> str:
    """Deterministically assign a user to model A or B based on user_id parity."""
    return MODEL_A if user_id % 2 == 0 else MODEL_B


@router.post("", response_model=ABTestResponse)
def ab_test(req: ABTestRequest) -> ABTestResponse:
    assigned = _assign_model(req.user_id)

    def _get_recs(model_name: str) -> "RecommendResponse":
        return recommend(
            RecommendRequest(
                user_id=req.user_id,
                genre=req.genre,
                model_name=model_name,
                scoring_method="cosine",
                top_n=req.top_n,
            )
        )

    return ABTestResponse(
        user_id=req.user_id,
        model_a=_get_recs(MODEL_A),
        model_b=_get_recs(MODEL_B),
        assigned_model=assigned,
    )
