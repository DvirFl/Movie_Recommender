"""Tests: FastAPI endpoints — recommend, ab_test, batch, trigger, viz, health."""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from serving.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /recommend
# ---------------------------------------------------------------------------

def _mock_cold_start_row():
    row = MagicMock()
    row.movie_ids = [10, 20, 30]
    row.scores = [0.9, 0.8, 0.7]
    row.genre = "Action"
    return row


def _mock_user_row():
    row = MagicMock()
    row.movie_ids = [10, 20, 30]
    row.scores = [0.9, 0.8, 0.7]
    row.genre = "Action"
    return row


def test_recommend_cold_start():
    with patch("serving.routes.recommend.get_session") as mock_ctx:
        mock_session = MagicMock()
        mock_ctx.return_value.__enter__.return_value = mock_session
        mock_ctx.return_value.__exit__.return_value = False
        mock_session.scalars.return_value.first.return_value = _mock_cold_start_row()

        response = client.post("/recommend", json={
            "model_name": "TwoTower_TimedecayMSELoss",
            "scoring_method": "cosine",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "precomputed"
        assert len(data["movie_ids"]) > 0


def test_recommend_personalized():
    with patch("serving.routes.recommend.get_session") as mock_ctx:
        mock_session = MagicMock()
        mock_ctx.return_value.__enter__.return_value = mock_session
        mock_ctx.return_value.__exit__.return_value = False
        mock_session.scalars.return_value.first.return_value = _mock_user_row()

        response = client.post("/recommend", json={
            "user_id": 1,
            "model_name": "TwoTower_TimedecayMSELoss",
            "scoring_method": "cosine",
        })
        assert response.status_code == 200
        assert response.json()["user_id"] == 1


def test_recommend_not_found():
    with patch("serving.routes.recommend.get_session") as mock_ctx:
        mock_session = MagicMock()
        mock_ctx.return_value.__enter__.return_value = mock_session
        mock_ctx.return_value.__exit__.return_value = False
        mock_session.scalars.return_value.first.return_value = None

        response = client.post("/recommend", json={
            "user_id": 99999,
            "model_name": "TwoTower_TimedecayMSELoss",
            "scoring_method": "cosine",
        })
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# /batch
# ---------------------------------------------------------------------------

def test_batch_recommend():
    with patch("serving.routes.recommend.get_session") as mock_ctx:
        mock_session = MagicMock()
        mock_ctx.return_value.__enter__.return_value = mock_session
        mock_ctx.return_value.__exit__.return_value = False
        mock_session.scalars.return_value.first.return_value = _mock_user_row()

        response = client.post("/batch", json={
            "user_ids": [1, 2, 3],
            "model_name": "TwoTower_TimedecayMSELoss",
            "scoring_method": "cosine",
        })
        assert response.status_code == 200
        assert len(response.json()["results"]) == 3


# ---------------------------------------------------------------------------
# /trigger
# ---------------------------------------------------------------------------

def test_trigger_on_demand():
    with patch("serving.routes.trigger.get_session") as mock_ctx, \
         patch("serving.routes.trigger.httpx.post") as mock_post:

        mock_session = MagicMock()
        mock_ctx.return_value.__enter__.return_value = mock_session
        mock_ctx.return_value.__exit__.return_value = False
        mock_session.get.return_value = MagicMock()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"dag_run_id": "test_run_123"}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        response = client.post("/trigger", json={
            "losses": ["TimedecayMSELoss"],
            "architectures": ["TwoTower"],
            "run_from": "featurize",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["trigger_id"] is not None


def test_trigger_creates_log_entry():
    with patch("serving.routes.trigger.get_session") as mock_ctx, \
         patch("serving.routes.trigger.httpx.post") as mock_post:
        mock_session = MagicMock()
        mock_ctx.return_value.__enter__.return_value = mock_session
        mock_ctx.return_value.__exit__.return_value = False
        mock_session.get.return_value = MagicMock()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"dag_run_id": "run_abc"}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        client.post("/trigger", json={"losses": "all", "architectures": "all", "run_from": "featurize"})
        mock_session.add.assert_called_once()


# ---------------------------------------------------------------------------
# /viz
# ---------------------------------------------------------------------------

def test_viz_pipeline_sizes():
    with patch("serving.routes.viz.count_raw_rows") as mock_counts:
        mock_counts.return_value = {"raw.ratings": 1000, "raw.movies": 50}
        response = client.get("/viz/pipeline_sizes")
        assert response.status_code == 200
        data = response.json()
        assert any(d["stage"] == "raw.ratings" for d in data)


def test_viz_runs_empty():
    with patch("serving.routes.viz._get_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        mock_client.search_experiments.return_value = []
        mock_client.search_runs.return_value = []

        response = client.get("/viz/runs")
        assert response.status_code == 200
        assert response.json() == []
