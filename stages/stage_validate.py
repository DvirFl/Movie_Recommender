"""Stage 2 — Validate: schema, null, and referential integrity checks on raw.* data."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidateResult:
    passed: bool = True
    issues: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def run(strict: bool = True) -> ValidateResult:
    """Run all validation checks against raw.* tables.

    Args:
        strict: if True (default), raise RuntimeError on any failed check so
                the pipeline stops.  Set False to log-and-continue.

    Returns:
        ValidateResult with pass/fail status, issues list, and row-count stats.
    """
    from etl.validate import validate_all

    logger.info("[validate] Running checks...")
    report = validate_all()

    result = ValidateResult(
        passed=report.passed,
        issues=report.issues,
        stats=report.stats,
    )

    if not report.passed:
        for issue in report.issues:
            logger.error("[validate] %s", issue)
        if strict:
            raise RuntimeError(
                f"[validate] Validation failed with {len(report.issues)} issue(s). "
                "Fix raw data or re-run ingest before continuing."
            )
    else:
        logger.info("[validate] All checks passed — stats: %s", report.stats)

    return result
