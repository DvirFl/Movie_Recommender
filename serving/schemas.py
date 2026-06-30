"""Pydantic schemas for all API endpoints."""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /recommend
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    user_id: Optional[int] = None           # None → cold-start
    genre: Optional[str] = None
    model_name: str = "TwoTower_TimedecayMSELoss"
    scoring_method: Literal["cosine", "dot", "l2", "learned"] = "cosine"
    top_n: int = Field(default=10, ge=1, le=100)


class RecommendResponse(BaseModel):
    user_id: Optional[int]
    genre: Optional[str]
    model_name: str
    scoring_method: str
    movie_ids: list[int]
    scores: list[float]
    source: Literal["precomputed", "realtime"]


# ---------------------------------------------------------------------------
# /ab_test
# ---------------------------------------------------------------------------

class ABTestRequest(BaseModel):
    user_id: int
    genre: Optional[str] = None
    top_n: int = Field(default=10, ge=1, le=100)


class ABTestResponse(BaseModel):
    user_id: int
    model_a: RecommendResponse
    model_b: RecommendResponse
    assigned_model: str                     # which model this user is assigned to


# ---------------------------------------------------------------------------
# /batch
# ---------------------------------------------------------------------------

class BatchRequest(BaseModel):
    user_ids: list[int]
    genre: Optional[str] = None
    model_name: str = "TwoTower_TimedecayMSELoss"
    scoring_method: Literal["cosine", "dot", "l2", "learned"] = "cosine"
    top_n: int = Field(default=10, ge=1, le=100)


class BatchResponse(BaseModel):
    results: list[RecommendResponse]


# ---------------------------------------------------------------------------
# /trigger
# ---------------------------------------------------------------------------

class TriggerRequest(BaseModel):
    losses: list[str] | Literal["all"] = "all"
    architectures: list[str] | Literal["all"] = "all"
    run_from: Literal["featurize"] = "featurize"
    requester: Optional[str] = None


class TriggerResponse(BaseModel):
    trigger_id: str
    airflow_run_id: Optional[str]
    status: str
    requested_losses: list[str] | str
    requested_architectures: list[str] | str


# ---------------------------------------------------------------------------
# /viz
# ---------------------------------------------------------------------------

class MetricPoint(BaseModel):
    step: int
    value: float


class RunMetrics(BaseModel):
    run_id: str
    run_name: str
    architecture: str
    loss: str
    distillation_type: str
    metrics: dict[str, list[MetricPoint]]
    params: dict[str, str]
    tags: dict[str, str]


class PipelineStageSize(BaseModel):
    stage: str
    row_count: int


class VizSummaryResponse(BaseModel):
    runs: list[RunMetrics]
    pipeline_sizes: list[PipelineStageSize]
