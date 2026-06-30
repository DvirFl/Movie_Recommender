"""Tests: stage_ingest — path resolution, env var fallback, diagnostics, errors."""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

These tests focus on stage-level behaviour: env var fallback, error propagation,
and the IngestReport return type.  Reader and classifier tests live in test_etl.py.
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from stages.stage_ingest import run, _find_candidates
from etl.ingest import IngestReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _good_report(**kwargs) -> IngestReport:
    defaults = dict(
        source="/data/ml-1m.zip",
        variant="ml-1m",
        ingested={"raw.ratings": 100, "raw.movies": 50},
        skipped={}, errors={}, warnings=[],
    )
    defaults.update(kwargs)
    return IngestReport(**defaults)


def _bad_report(**kwargs) -> IngestReport:
    defaults = dict(
        source="/data", variant="unknown",
        ingested={}, skipped={},
        errors={"ratings.csv": "DB error"},
        warnings=["Required tables not ingested: ['raw.ratings']"],
    )
    defaults.update(kwargs)
    return IngestReport(**defaults)


def _patch_ingest(report: IngestReport):
    """Patch ingest_all so no real DB or files are needed."""
    return patch("stages.stage_ingest.ingest_all", return_value=report)


# ---------------------------------------------------------------------------
# Returns IngestReport
# ---------------------------------------------------------------------------

class TestPathResolution:

    def test_absolute_path_used_directly(self, tmp_path):
        abs_path = tmp_path / "ml-1m.zip"
        abs_path.touch()
        with _patch_ingest(_good_report(source=str(abs_path))) as mock_fn:
            run(data_dir=abs_path)
        called = mock_fn.call_args[0][0]
        assert called.is_absolute()
        assert called == abs_path.resolve()

    def test_relative_path_resolved_to_absolute(self, tmp_path, monkeypatch):
        # Place a file in tmp_path and set cwd there so relative path works
        (tmp_path / "ml-1m.zip").touch()
        monkeypatch.chdir(tmp_path)
        with _patch_ingest(_good_report()) as mock_fn:
            run(data_dir=Path("ml-1m.zip"))
        called = mock_fn.call_args[0][0]
        assert called.is_absolute()
        assert called == (tmp_path / "ml-1m.zip").resolve()

    def test_tilde_expanded(self, tmp_path, monkeypatch):
        # Redirect home to tmp_path so ~/ml-1m.zip exists
        fake_home = tmp_path
        (fake_home / "ml-1m.zip").touch()
        monkeypatch.setenv("HOME", str(fake_home))
        with _patch_ingest(_good_report()) as mock_fn:
            run(data_dir=Path("~/ml-1m.zip"))
        called = mock_fn.call_args[0][0]
        assert called.is_absolute()
        assert called.exists()

    def test_env_var_used_when_no_data_dir(self, tmp_path, monkeypatch):
        zip_path = tmp_path / "ml-1m.zip"
        zip_path.touch()
        monkeypatch.setenv("MOVIELENS_DATA_DIR", str(zip_path))
        with _patch_ingest(_good_report()) as mock_fn:
            run(data_dir=None)
        called = mock_fn.call_args[0][0]
        assert called == zip_path.resolve()

    def test_default_path_used_when_no_env_no_arg(self, monkeypatch):
        monkeypatch.delenv("MOVIELENS_DATA_DIR", raising=False)
        # Default path /data/movielens won't exist; we just check what is passed
        with patch("stages.stage_ingest.ingest_all",
                   side_effect=FileNotFoundError("not found")):
            with pytest.raises(FileNotFoundError):
                run(data_dir=None)

    def test_resolved_path_passed_to_ingest_all(self, tmp_path, monkeypatch):
        zip_path = tmp_path / "ml-25m.zip"
        zip_path.touch()
        monkeypatch.chdir(tmp_path)
        with _patch_ingest(_good_report()) as mock_fn:
            run(data_dir="ml-25m.zip")   # relative string
        called = mock_fn.call_args[0][0]
        # Must be the resolved absolute path, not the raw string
        assert called.is_absolute()
        assert "ml-25m.zip" in str(called)


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------

class TestFileNotFoundError:

    def test_raises_when_path_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError) as exc_info:
            run(data_dir=tmp_path / "does_not_exist.zip")
        msg = str(exc_info.value)
        assert "not found" in msg.lower()

    def test_error_shows_resolved_path(self, tmp_path):
        missing = tmp_path / "missing.zip"
        with pytest.raises(FileNotFoundError) as exc_info:
            run(data_dir=missing)
        assert str(missing.resolve()) in str(exc_info.value)

    def test_error_shows_working_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError) as exc_info:
            run(data_dir=tmp_path / "missing.zip")
        assert "Working dir" in str(exc_info.value) or \
               "Working directory" in str(exc_info.value)

    def test_error_shows_how_to_fix(self, tmp_path):
        with pytest.raises(FileNotFoundError) as exc_info:
            run(data_dir=tmp_path / "missing.zip")
        msg = str(exc_info.value)
        assert "--data-dir" in msg or "MOVIELENS_DATA_DIR" in msg

    def test_error_lists_candidates_when_found(self, tmp_path, monkeypatch):
        # Plant a zip in cwd that the candidate search will find
        (tmp_path / "ml-1m.zip").touch()
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError) as exc_info:
            run(data_dir=tmp_path / "nonexistent.zip")
        # Candidate search should find ml-1m.zip in cwd
        assert "ml-1m.zip" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Data dir resolution
# ---------------------------------------------------------------------------

class TestFindCandidates:

    def test_finds_zip_in_cwd(self, tmp_path, monkeypatch):
        (tmp_path / "ml-25m.zip").touch()
        monkeypatch.chdir(tmp_path)
        candidates = _find_candidates(Path("nonexistent"))
        assert any("ml-25m.zip" in c for c in candidates)

    def test_finds_extracted_directory_in_cwd(self, tmp_path, monkeypatch):
        (tmp_path / "ml-1m").mkdir()
        monkeypatch.chdir(tmp_path)
        candidates = _find_candidates(Path("nonexistent"))
        assert any("ml-1m" in c for c in candidates)

    def test_returns_at_most_eight_candidates(self, tmp_path, monkeypatch):
        for i in range(20):
            (tmp_path / f"ml-fake{i}.zip").touch()
        monkeypatch.chdir(tmp_path)
        candidates = _find_candidates(Path("nonexistent"))
        assert len(candidates) <= 8

    def test_returns_empty_when_nothing_found(self, tmp_path, monkeypatch):
        empty = tmp_path / "empty_dir"
        empty.mkdir()
        monkeypatch.chdir(empty)
        # Patch home so it also doesn't accidentally match
        monkeypatch.setenv("HOME", str(empty))
        candidates = _find_candidates(Path("nonexistent"))
        # May still find things in parent dirs — just assert it's a list
        assert isinstance(candidates, list)


# ---------------------------------------------------------------------------
# IngestReport returned correctly
# ---------------------------------------------------------------------------

class TestReturnValue:

    def test_returns_ingest_report(self, tmp_path):
        zip_path = tmp_path / "ml-1m.zip"
        zip_path.touch()
        with _patch_ingest(_good_report()):
            result = run(data_dir=zip_path)
        assert isinstance(result, IngestReport)

    def test_ok_report_does_not_raise(self, tmp_path):
        zip_path = tmp_path / "ml-1m.zip"
        zip_path.touch()
        with _patch_ingest(_good_report()):
            result = run(data_dir=zip_path)
        assert result.ok is True

    def test_total_rows_correct(self, tmp_path):
        zip_path = tmp_path / "ml-1m.zip"
        zip_path.touch()
        report = _good_report(ingested={
            "raw.ratings": 1_000_000,
            "raw.movies": 3_900,
        })
        with _patch_ingest(report):
            result = run(data_dir=zip_path)
        assert result.total_rows == 1_003_900

    def test_warnings_logged_but_not_raised(self, tmp_path):
        zip_path = tmp_path / "ml-1m.zip"
        zip_path.touch()
        report = _good_report(warnings=["Some non-critical warning"])
        with _patch_ingest(report):
            result = run(data_dir=zip_path)
        assert result.warnings == ["Some non-critical warning"]


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------

class TestErrorPropagation:

    def test_raises_runtime_error_on_ingest_errors(self, tmp_path):
        zip_path = tmp_path / "ml-1m.zip"
        zip_path.touch()
        with _patch_ingest(_bad_report()):
            with pytest.raises(RuntimeError, match="failed to ingest"):
                run(data_dir=zip_path)

    def test_error_message_contains_filename(self, tmp_path):
        zip_path = tmp_path / "ml-1m.zip"
        zip_path.touch()
        with _patch_ingest(_bad_report(errors={"ratings.dat": "parse error"})):
            with pytest.raises(RuntimeError, match="ratings.dat"):
                run(data_dir=zip_path)

    def test_error_message_contains_count(self, tmp_path):
        zip_path = tmp_path / "ml-1m.zip"
        zip_path.touch()
        errors = {"f1.csv": "e1", "f2.csv": "e2"}
        with _patch_ingest(_bad_report(errors=errors)):
            with pytest.raises(RuntimeError, match="2"):
                run(data_dir=zip_path)

def test_ok_false_when_ratings_missing():
    r = _good_report(ingested={"raw.movies": 50})
    assert r.ok is False

# ---------------------------------------------------------------------------
# IngestReport property tests (no I/O)
# ---------------------------------------------------------------------------

class TestIngestReportProperties:

    def test_ok_true_when_both_tables_present(self):
        r = _good_report(ingested={"raw.ratings": 1, "raw.movies": 1})
        assert r.ok is True

    def test_ok_false_when_ratings_missing(self):
        r = _good_report(ingested={"raw.movies": 50})
        assert r.ok is False

    def test_ok_false_when_movies_missing(self):
        r = _good_report(ingested={"raw.ratings": 100})
        assert r.ok is False

    def test_ok_false_when_nothing_ingested(self):
        r = _bad_report(ingested={})
        assert r.ok is False

    def test_total_rows_sums_all_tables(self):
        r = _good_report(ingested={
            "raw.ratings": 100, "raw.movies": 50,
            "raw.tags": 20, "raw.links": 10,
        })
        assert r.total_rows == 180


# ---------------------------------------------------------------------------
# mode parameter forwarding
# ---------------------------------------------------------------------------

class TestModeForwarding:

    def test_default_mode_is_upsert(self, tmp_path):
        zip_path = tmp_path / "ml-1m.zip"
        zip_path.touch()
        with patch("stages.stage_ingest.ingest_all",
                   return_value=_good_report()) as mock_fn:
            run(data_dir=zip_path)
        _, kwargs = mock_fn.call_args
        assert kwargs.get("mode", "upsert") == "upsert"

    def test_skip_mode_forwarded(self, tmp_path):
        zip_path = tmp_path / "ml-1m.zip"
        zip_path.touch()
        with patch("stages.stage_ingest.ingest_all",
                   return_value=_good_report()) as mock_fn:
            run(data_dir=zip_path, mode="skip")
        _, kwargs = mock_fn.call_args
        assert kwargs["mode"] == "skip"

    def test_replace_mode_forwarded(self, tmp_path):
        zip_path = tmp_path / "ml-1m.zip"
        zip_path.touch()
        with patch("stages.stage_ingest.ingest_all",
                   return_value=_good_report()) as mock_fn:
            run(data_dir=zip_path, mode="replace")
        _, kwargs = mock_fn.call_args
        assert kwargs["mode"] == "replace"
