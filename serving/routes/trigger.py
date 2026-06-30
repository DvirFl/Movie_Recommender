"""POST /trigger — on-demand DAG trigger endpoint."""
from __future__ import annotations

import uuid
import datetime as dt
import logging

import httpx
from fastapi import APIRouter, HTTPException

from db.connection import get_session
from db.models import TriggerLog
from serving.schemas import TriggerRequest, TriggerResponse

router = APIRouter(prefix="/trigger", tags=["trigger"])
logger = logging.getLogger(__name__)

AIRFLOW_BASE_URL = "http://localhost:8080/api/v1"
AIRFLOW_DAG_ID = "movielens_pipeline"
AIRFLOW_AUTH = ("airflow", "airflow")  # override via env in production


@router.post("", response_model=TriggerResponse)
def trigger_pipeline(req: TriggerRequest) -> TriggerResponse:
    trigger_id = str(uuid.uuid4())

    losses = req.losses if isinstance(req.losses, list) else ["all"]
    architectures = req.architectures if isinstance(req.architectures, list) else ["all"]

    # Persist trigger log entry
    log_entry = TriggerLog(
        trigger_id=trigger_id,
        trigger_type="on_demand",
        requested_losses=losses if req.losses != "all" else None,
        requested_architectures=architectures if req.architectures != "all" else None,
        status="pending",
        requester=req.requester,
        created_at=dt.datetime.now(tz=dt.timezone.utc),
    )
    with get_session() as session:
        session.add(log_entry)

    # Trigger Airflow DAG via REST API
    airflow_run_id: str | None = None
    try:
        payload = {
            "conf": {
                "trigger_id": trigger_id,
                "trigger_type": "on_demand",
                "losses": req.losses,
                "architectures": req.architectures,
                "run_from": req.run_from,
            }
        }
        response = httpx.post(
            f"{AIRFLOW_BASE_URL}/dags/{AIRFLOW_DAG_ID}/dagRuns",
            json=payload,
            auth=AIRFLOW_AUTH,
            timeout=10.0,
        )
        response.raise_for_status()
        airflow_run_id = response.json().get("dag_run_id")
    except Exception as exc:
        logger.error("Failed to trigger Airflow DAG: %s", exc)
        with get_session() as session:
            entry = session.get(TriggerLog, trigger_id)
            if entry:
                entry.status = "failed"
        raise HTTPException(status_code=502, detail=f"Airflow trigger failed: {exc}")

    # Update log with airflow_run_id
    with get_session() as session:
        entry = session.get(TriggerLog, trigger_id)
        if entry:
            entry.airflow_run_id = airflow_run_id
            entry.status = "running"

    return TriggerResponse(
        trigger_id=trigger_id,
        airflow_run_id=airflow_run_id,
        status="running",
        requested_losses=req.losses,
        requested_architectures=req.architectures,
    )
