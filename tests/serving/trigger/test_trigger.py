"""Tests: /trigger endpoint — watermark sensor, log entry, DAG activation."""
import pytest
from unittest.mock import patch, MagicMock


def test_trigger_log_entry_created():
    """TriggerLog entry is persisted when trigger is called."""
    from fastapi.testclient import TestClient
    from serving.api import app
    client = TestClient(app)

    with patch("serving.routes.trigger.get_session") as mock_ctx, \
         patch("serving.routes.trigger.httpx.post") as mock_post:
        session = MagicMock()
        mock_ctx.return_value.__enter__.return_value = session
        mock_ctx.return_value.__exit__.return_value = False
        session.get.return_value = MagicMock()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"dag_run_id": "dag_run_xyz"}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        resp = client.post("/trigger", json={
            "losses": ["TimedecayInfoNCELoss"],
            "architectures": "all",
            "run_from": "featurize",
            "requester": "test_user",
        })
        assert resp.status_code == 200
        # TriggerLog was added to session
        session.add.assert_called_once()
        added_obj = session.add.call_args[0][0]
        from db.models import TriggerLog
        assert isinstance(added_obj, TriggerLog)
        assert added_obj.trigger_type == "on_demand"
        assert added_obj.requester == "test_user"


def test_trigger_fails_gracefully_when_airflow_down():
    from fastapi.testclient import TestClient
    from serving.api import app
    client = TestClient(app)

    with patch("serving.routes.trigger.get_session") as mock_ctx, \
         patch("serving.routes.trigger.httpx.post", side_effect=Exception("connection refused")):
        session = MagicMock()
        mock_ctx.return_value.__enter__.return_value = session
        mock_ctx.return_value.__exit__.return_value = False
        session.get.return_value = MagicMock()

        resp = client.post("/trigger", json={
            "losses": "all",
            "architectures": "all",
            "run_from": "featurize",
        })
        assert resp.status_code == 502


def test_watermark_sensor_poke_no_new_data():
    """Sensor returns False when no new data — tests check_watermark directly."""
    from airflow.dags.pipeline_logic import check_watermark

    with patch("db.connection.get_session") as mock_ctx:
        session = MagicMock()
        mock_ctx.return_value.__enter__.return_value = session
        mock_ctx.return_value.__exit__.return_value = False
        session.execute.return_value.scalar.return_value = 0

        has_new, _ = check_watermark(["raw.ratings"])
        assert has_new is False


def test_watermark_sensor_poke_fires_on_new_data():
    """Sensor returns True when new data is detected — tests check_watermark directly."""
    from airflow.dags.pipeline_logic import check_watermark

    with patch("db.connection.get_session") as mock_ctx:
        session = MagicMock()
        mock_ctx.return_value.__enter__.return_value = session
        mock_ctx.return_value.__exit__.return_value = False
        session.execute.return_value.scalar.return_value = 42

        has_new, table = check_watermark(["raw.ratings"])
        assert has_new is True
        assert table == "raw.ratings"
