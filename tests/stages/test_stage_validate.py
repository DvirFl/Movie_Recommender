"""Tests: stage_validate."""
import pytest
from unittest.mock import MagicMock, patch
from stages.stage_validate import run, ValidateResult


def _report(passed=True, issues=None, stats=None):
    r = MagicMock()
    r.passed = passed
    r.issues = issues or []
    r.stats  = stats or {"ratings_count": 1000}
    return r


def test_returns_validate_result_on_pass():
    with patch("etl.validate.validate_all", return_value=_report(passed=True)):
        result = run()
    assert isinstance(result, ValidateResult)
    assert result.passed is True


def test_returns_stats_on_pass():
    with patch("etl.validate.validate_all", return_value=_report(stats={"ratings_count": 500})):
        result = run()
    assert result.stats["ratings_count"] == 500


def test_raises_on_failure_strict_mode():
    with patch("etl.validate.validate_all",
               return_value=_report(passed=False, issues=["null user_id"])):
        with pytest.raises(RuntimeError, match="Validation failed"):
            run(strict=True)


def test_no_raise_in_non_strict_mode():
    with patch("etl.validate.validate_all",
               return_value=_report(passed=False, issues=["warn"])):
        result = run(strict=False)
    assert result.passed is False
    assert "warn" in result.issues


def test_issues_captured_on_failure():
    with patch("etl.validate.validate_all",
               return_value=_report(passed=False, issues=["e1", "e2"])):
        with pytest.raises(RuntimeError):
            run(strict=True)
