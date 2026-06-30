"""ETL Stage 2: Validate raw data — schema, nulls, referential integrity."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import func, select, text

from db.connection import get_session
from db.models import RawLink, RawMovie, RawRating, RawTag

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    passed: bool = True
    issues: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.issues.append(msg)
        logger.error("Validation failed: %s", msg)

    def warn(self, msg: str) -> None:
        self.issues.append(f"[WARN] {msg}")
        logger.warning("Validation warning: %s", msg)


def validate_ratings(report: ValidationReport) -> None:
    with get_session() as session:
        total = session.scalar(select(func.count()).select_from(RawRating))
        report.stats["ratings_count"] = total

        if total == 0:
            report.fail("raw.ratings is empty.")
            return

        # Null checks
        null_users = session.scalar(
            select(func.count()).select_from(RawRating).where(RawRating.user_id == None)  # noqa: E711
        )
        if null_users:
            report.fail(f"raw.ratings has {null_users} null user_id values.")

        null_movies = session.scalar(
            select(func.count()).select_from(RawRating).where(RawRating.movie_id == None)  # noqa: E711
        )
        if null_movies:
            report.fail(f"raw.ratings has {null_movies} null movie_id values.")

        # Rating range check [0.5, 5.0]
        out_of_range = session.scalar(
            select(func.count()).select_from(RawRating).where(
                (RawRating.rating < 0.5) | (RawRating.rating > 5.0)
            )
        )
        if out_of_range:
            report.warn(f"raw.ratings has {out_of_range} out-of-range ratings.")

        report.stats["unique_users"] = session.scalar(
            select(func.count(func.distinct(RawRating.user_id))).select_from(RawRating)
        )
        report.stats["unique_movies_rated"] = session.scalar(
            select(func.count(func.distinct(RawRating.movie_id))).select_from(RawRating)
        )


def validate_movies(report: ValidationReport) -> None:
    with get_session() as session:
        total = session.scalar(select(func.count()).select_from(RawMovie))
        report.stats["movies_count"] = total
        if total == 0:
            report.fail("raw.movies is empty.")
            return

        null_titles = session.scalar(
            select(func.count()).select_from(RawMovie).where(RawMovie.title == None)  # noqa: E711
        )
        if null_titles:
            report.fail(f"raw.movies has {null_titles} null title values.")


def validate_referential_integrity(report: ValidationReport) -> None:
    """Ensure all movie_ids in ratings exist in movies."""
    with get_session() as session:
        orphan_count = session.scalar(
            text("""
                SELECT COUNT(*) FROM raw.ratings r
                LEFT JOIN raw.movies m ON r.movie_id = m.movie_id
                WHERE m.movie_id IS NULL
            """)
        )
        if orphan_count:
            report.warn(
                f"{orphan_count} rating rows reference movie_ids not in raw.movies."
            )


def validate_all() -> ValidationReport:
    """Run all validation checks. Returns a ValidationReport."""
    report = ValidationReport()
    validate_ratings(report)
    validate_movies(report)
    validate_referential_integrity(report)
    logger.info(
        "Validation complete. Passed=%s Issues=%d Stats=%s",
        report.passed, len(report.issues), report.stats,
    )
    return report
