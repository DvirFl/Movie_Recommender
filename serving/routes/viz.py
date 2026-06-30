"""GET /viz/* — training visualization data sourced from MLflow tracking API."""
from __future__ import annotations

from fastapi import APIRouter, Query
import mlflow
from mlflow.tracking import MlflowClient

from etl.utils import count_raw_rows
from serving.schemas import MetricPoint, PipelineStageSize, RunMetrics, VizSummaryResponse

router = APIRouter(prefix="/viz", tags=["viz"])


def _get_client() -> MlflowClient:
    return MlflowClient()


@router.get("/runs", response_model=list[RunMetrics])
def get_runs(
    experiment_name: str | None = Query(default=None),
    architecture: str | None = Query(default=None),
    distillation_type: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> list[RunMetrics]:
    """Return training run metrics from MLflow."""
    client = _get_client()

    # Resolve experiment IDs
    if experiment_name:
        exp = client.get_experiment_by_name(experiment_name)
        experiment_ids = [exp.experiment_id] if exp else []
    else:
        experiment_ids = [e.experiment_id for e in client.search_experiments()]

    filter_parts = []
    if architecture:
        filter_parts.append(f"tags.architecture = '{architecture}'")
    if distillation_type:
        filter_parts.append(f"tags.distillation_type = '{distillation_type}'")
    filter_str = " AND ".join(filter_parts)

    runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string=filter_str,
        max_results=limit,
        order_by=["start_time DESC"],
    )

    results = []
    for run in runs:
        metrics_dict: dict[str, list[MetricPoint]] = {}
        for metric_key in run.data.metrics:
            history = client.get_metric_history(run.info.run_id, metric_key)
            metrics_dict[metric_key] = [
                MetricPoint(step=h.step, value=h.value) for h in history
            ]

        results.append(
            RunMetrics(
                run_id=run.info.run_id,
                run_name=run.info.run_name or "",
                architecture=run.data.tags.get("architecture", ""),
                loss=run.data.tags.get("loss", ""),
                distillation_type=run.data.tags.get("distillation_type", "none"),
                metrics=metrics_dict,
                params={k: str(v) for k, v in run.data.params.items()},
                tags=run.data.tags,
            )
        )
    return results


@router.get("/pipeline_sizes", response_model=list[PipelineStageSize])
def get_pipeline_sizes() -> list[PipelineStageSize]:
    """Return row counts at each pipeline stage."""
    counts = count_raw_rows()
    return [PipelineStageSize(stage=k, row_count=v or 0) for k, v in counts.items()]


@router.get("/summary", response_model=VizSummaryResponse)
def get_summary() -> VizSummaryResponse:
    runs = get_runs(limit=20)
    sizes = get_pipeline_sizes()
    return VizSummaryResponse(runs=runs, pipeline_sizes=sizes)


@router.get("/hparam_sweep")
def get_hparam_sweep(experiment_name: str) -> list[dict]:
    """Return all sweep trial results for a given experiment."""
    client = _get_client()
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        return []

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.sweep = 'true'",
        max_results=500,
    )
    return [
        {
            "run_id": r.info.run_id,
            "params": r.data.params,
            "metrics": r.data.metrics,
            "status": r.info.status,
        }
        for r in runs
    ]
