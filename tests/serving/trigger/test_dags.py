"""Tests: DAG pipeline logic — all tests run against pipeline_logic.py only.

No Airflow runtime is required.  The DAG files themselves (dag_*.py, common.py)
are NOT imported here because they require a live Airflow installation.
pipeline_logic.py has zero Airflow imports and is the sole unit under test.
"""
from __future__ import annotations

import datetime as dt
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dag_run(conf: dict | None = None, run_id: str = "test_run"):
    dr = MagicMock()
    dr.conf = conf or {}
    dr.run_id = run_id
    return dr


def _ctx(conf: dict | None = None) -> dict:
    ti = MagicMock()
    ti.xcom_pull.return_value = None
    return {"dag_run": _dag_run(conf), "ti": ti}


# ---------------------------------------------------------------------------
# get_active_combinations
# ---------------------------------------------------------------------------

class TestGetActiveCombinations:

    def test_all_returns_all_enabled(self):
        from airflow.dags.pipeline_logic import get_active_combinations
        combos = get_active_combinations({"losses": "all", "architectures": "all"})
        assert len(combos) >= 1

    def test_filter_by_loss_list(self):
        from airflow.dags.pipeline_logic import get_active_combinations
        combos = get_active_combinations({"losses": ["TimedecayMSELoss"], "architectures": "all"})
        assert combos and all(l.name == "TimedecayMSELoss" for _, l in combos)

    def test_filter_by_arch_list(self):
        from airflow.dags.pipeline_logic import get_active_combinations
        combos = get_active_combinations({"losses": "all", "architectures": ["TwoTower"]})
        assert combos and all(a.name == "TwoTower" for a, _ in combos)

    def test_filter_by_both(self):
        from airflow.dags.pipeline_logic import get_active_combinations
        combos = get_active_combinations({
            "losses": ["TimedecayMSELoss"],
            "architectures": ["TwoTower"],
        })
        assert all(a.name == "TwoTower" and l.name == "TimedecayMSELoss" for a, l in combos)

    def test_unknown_loss_returns_empty(self):
        from airflow.dags.pipeline_logic import get_active_combinations
        combos = get_active_combinations({"losses": ["DoesNotExist"], "architectures": "all"})
        assert combos == []

    def test_unknown_arch_returns_empty(self):
        from airflow.dags.pipeline_logic import get_active_combinations
        combos = get_active_combinations({"losses": "all", "architectures": ["DoesNotExist"]})
        assert combos == []

    def test_loss_as_string_normalised_to_list(self):
        from airflow.dags.pipeline_logic import get_active_combinations
        combos = get_active_combinations({"losses": "TimedecayMSELoss", "architectures": "all"})
        assert combos and all(l.name == "TimedecayMSELoss" for _, l in combos)

    def test_arch_as_string_normalised_to_list(self):
        from airflow.dags.pipeline_logic import get_active_combinations
        combos = get_active_combinations({"losses": "all", "architectures": "TwoTower"})
        assert combos and all(a.name == "TwoTower" for a, _ in combos)

    def test_empty_conf_returns_all(self):
        from airflow.dags.pipeline_logic import get_active_combinations
        combos_empty = get_active_combinations({})
        combos_all   = get_active_combinations({"losses": "all", "architectures": "all"})
        assert len(combos_empty) == len(combos_all)


# ---------------------------------------------------------------------------
# should_run_combination
# ---------------------------------------------------------------------------

class TestShouldRunCombination:

    def test_returns_true_when_all(self):
        from airflow.dags.pipeline_logic import should_run_combination
        assert should_run_combination("TwoTower", "TimedecayMSELoss",
                                      {"losses": "all", "architectures": "all"}) is True

    def test_returns_true_when_explicitly_listed(self):
        from airflow.dags.pipeline_logic import should_run_combination
        assert should_run_combination(
            "TwoTower", "TimedecayMSELoss",
            {"losses": ["TimedecayMSELoss"], "architectures": ["TwoTower"]},
        ) is True

    def test_returns_false_when_arch_not_listed(self):
        from airflow.dags.pipeline_logic import should_run_combination
        assert should_run_combination(
            "InfoNCEEncoder", "TimedecayInfoNCELoss",
            {"losses": "all", "architectures": ["TwoTower"]},
        ) is False

    def test_returns_false_when_loss_not_listed(self):
        from airflow.dags.pipeline_logic import should_run_combination
        assert should_run_combination(
            "TwoTower", "TimedecayMSELoss",
            {"losses": ["TimedecayInfoNCELoss"], "architectures": "all"},
        ) is False

    def test_empty_conf_allows_all(self):
        from airflow.dags.pipeline_logic import should_run_combination
        assert should_run_combination("TwoTower", "TimedecayMSELoss", {}) is True

    def test_nonexistent_combo_returns_false(self):
        from airflow.dags.pipeline_logic import should_run_combination
        assert should_run_combination("Ghost", "FakeLoss",
                                      {"losses": "all", "architectures": "all"}) is False


# ---------------------------------------------------------------------------
# check_watermark
# ---------------------------------------------------------------------------

class TestCheckWatermark:
    """get_session is imported inside check_watermark — patch at db.connection."""

    def _make_session_ctx(self, row_counts: list[int]):
        scalars = iter(row_counts)
        session = MagicMock()
        session.execute.return_value.scalar.side_effect = lambda: next(scalars, 0)
        ctx = MagicMock()
        ctx.__enter__.return_value = session
        ctx.__exit__.return_value = False
        return ctx

    def test_returns_false_when_no_new_rows(self):
        from airflow.dags.pipeline_logic import check_watermark
        with patch("db.connection.get_session", return_value=self._make_session_ctx([0, 0])):
            has_new, table = check_watermark(["raw.ratings", "raw.movies"])
        assert has_new is False
        assert table is None

    def test_returns_true_on_first_table_with_new_rows(self):
        from airflow.dags.pipeline_logic import check_watermark
        with patch("db.connection.get_session", return_value=self._make_session_ctx([3])):
            has_new, table = check_watermark(["raw.ratings"])
        assert has_new is True
        assert table == "raw.ratings"

    def test_returns_true_when_second_table_has_new_rows(self):
        from airflow.dags.pipeline_logic import check_watermark
        with patch("db.connection.get_session", return_value=self._make_session_ctx([0, 7])):
            has_new, table = check_watermark(["raw.ratings", "raw.movies"])
        assert has_new is True
        assert table == "raw.movies"

    def test_returns_false_all_zero(self):
        from airflow.dags.pipeline_logic import check_watermark, RAW_TABLES
        zeros = [0] * len(RAW_TABLES)
        with patch("db.connection.get_session", return_value=self._make_session_ctx(zeros)):
            has_new, _ = check_watermark(RAW_TABLES)
        assert has_new is False


# ---------------------------------------------------------------------------
# check_new_data_for_daily  (delegates to check_watermark)
# ---------------------------------------------------------------------------

class TestCheckNewDataForDaily:

    def test_returns_true_when_new_data(self):
        from airflow.dags.pipeline_logic import check_new_data_for_daily
        with patch("airflow.dags.pipeline_logic.check_watermark", return_value=(True, "raw.ratings")):
            assert check_new_data_for_daily() is True

    def test_returns_false_when_no_new_data(self):
        from airflow.dags.pipeline_logic import check_new_data_for_daily
        with patch("airflow.dags.pipeline_logic.check_watermark", return_value=(False, None)):
            assert check_new_data_for_daily() is False


# ---------------------------------------------------------------------------
# update_trigger_log
# ---------------------------------------------------------------------------

class TestUpdateTriggerLog:

    def test_noop_when_trigger_id_is_none(self):
        from airflow.dags.pipeline_logic import update_trigger_log
        update_trigger_log(None, "success")  # must not raise

    def test_sets_status_success(self):
        from airflow.dags.pipeline_logic import update_trigger_log
        from db.models import TriggerLog
        mock_entry = MagicMock(spec=TriggerLog)
        with patch("db.connection.get_session") as mock_ctx:
            session = MagicMock()
            mock_ctx.return_value.__enter__.return_value = session
            mock_ctx.return_value.__exit__.return_value = False
            session.get.return_value = mock_entry
            update_trigger_log("tid-123", "success")
        assert mock_entry.status == "success"
        assert mock_entry.completed_at is not None

    def test_sets_status_failed(self):
        from airflow.dags.pipeline_logic import update_trigger_log
        from db.models import TriggerLog
        mock_entry = MagicMock(spec=TriggerLog)
        with patch("db.connection.get_session") as mock_ctx:
            session = MagicMock()
            mock_ctx.return_value.__enter__.return_value = session
            mock_ctx.return_value.__exit__.return_value = False
            session.get.return_value = mock_entry
            update_trigger_log("tid-456", "failed")
        assert mock_entry.status == "failed"
        assert mock_entry.completed_at is not None

    def test_noop_when_entry_not_found(self):
        from airflow.dags.pipeline_logic import update_trigger_log
        with patch("db.connection.get_session") as mock_ctx:
            session = MagicMock()
            mock_ctx.return_value.__enter__.return_value = session
            mock_ctx.return_value.__exit__.return_value = False
            session.get.return_value = None
            update_trigger_log("unknown-id", "success")  # must not raise

    def test_silently_handles_db_error(self):
        from airflow.dags.pipeline_logic import update_trigger_log
        with patch("db.connection.get_session", side_effect=Exception("DB down")):
            update_trigger_log("tid-789", "success")  # must not propagate


# ---------------------------------------------------------------------------
# run_validate
# ---------------------------------------------------------------------------

class TestRunValidate:

    def test_raises_on_failed_report(self):
        from airflow.dags.pipeline_logic import run_validate
        report = MagicMock(passed=False, issues=["null user_id"], stats={})
        with patch("etl.validate.validate_all", return_value=report):
            with pytest.raises(ValueError, match="Validation failed"):
                run_validate(**_ctx())

    def test_returns_stats_on_pass(self):
        from airflow.dags.pipeline_logic import run_validate
        report = MagicMock(passed=True, stats={"ratings_count": 500})
        with patch("etl.validate.validate_all", return_value=report):
            result = run_validate(**_ctx())
        assert result["ratings_count"] == 500


# ---------------------------------------------------------------------------
# run_featurize / run_split
# ---------------------------------------------------------------------------

class TestRunFeaturize:
    def test_returns_counts(self):
        from airflow.dags.pipeline_logic import run_featurize
        with patch("etl.featurize.featurize_all", return_value={"users": 10, "items": 50}):
            assert run_featurize(**_ctx()) == {"users": 10, "items": 50}


class TestRunSplit:
    def test_returns_counts(self):
        from airflow.dags.pipeline_logic import run_split
        with patch("etl.split.split_ratings",
                   return_value={"train": 80, "val": 10, "test": 10}):
            assert run_split(**_ctx()) == {"train": 80, "val": 10, "test": 10}


# ---------------------------------------------------------------------------
# run_finalize
# ---------------------------------------------------------------------------

class TestRunFinalize:
    def test_updates_trigger_log(self):
        from airflow.dags.pipeline_logic import run_finalize
        with patch("airflow.dags.pipeline_logic.update_trigger_log") as mock_log:
            run_finalize(**_ctx(conf={"trigger_id": "tid-999"}))
            mock_log.assert_called_once_with("tid-999", "success")

    def test_noop_without_trigger_id(self):
        from airflow.dags.pipeline_logic import run_finalize
        with patch("airflow.dags.pipeline_logic.update_trigger_log") as mock_log:
            run_finalize(**_ctx(conf={}))
            mock_log.assert_called_once_with(None, "success")


# ---------------------------------------------------------------------------
# resolve_trainer
# ---------------------------------------------------------------------------

class TestResolveTrainer:
    def test_resolves_two_tower_trainer(self):
        from airflow.dags.pipeline_logic import resolve_trainer
        train_fn = resolve_trainer("TwoTower")
        assert callable(train_fn)

    def test_resolves_infonce_trainer(self):
        from airflow.dags.pipeline_logic import resolve_trainer
        train_fn = resolve_trainer("InfoNCEEncoder")
        assert callable(train_fn)

    def test_raises_for_unknown_arch(self):
        from airflow.dags.pipeline_logic import resolve_trainer
        with pytest.raises(KeyError):
            resolve_trainer("GhostArch")


# ---------------------------------------------------------------------------
# RAW_TABLES constant
# ---------------------------------------------------------------------------

class TestRawTables:
    def test_all_expected_tables_present(self):
        from airflow.dags.pipeline_logic import RAW_TABLES
        expected = {"raw.ratings", "raw.movies", "raw.tags", "raw.links", "raw.genome_scores"}
        assert expected == set(RAW_TABLES)

    def test_table_names_have_schema_prefix(self):
        from airflow.dags.pipeline_logic import RAW_TABLES
        for t in RAW_TABLES:
            assert "." in t, f"Table '{t}' missing schema prefix"
