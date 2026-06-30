"""Stage 1 — Ingest: load a MovieLens dataset into PostgreSQL raw.* schema.

Accepts a .zip file, an extracted directory, or a single data file.
Relative paths are resolved to absolute from the current working directory.
Auto-detects the dataset variant (ml-latest-small, ml-25m, ml-1m, ml-100k).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from etl.ingest import IngestReport, ingest_all

logger = logging.getLogger(__name__)


def _find_candidates(raw_path: Path) -> list[str]:
    """Search nearby directories for files that look like MovieLens data."""
    candidates = []
    search_dirs = [
        Path.cwd(),
        Path.cwd().parent,
        Path.home(),
        Path.home() / "Downloads",
        Path.home() / "datasets",
        Path.home() / "data",
    ]
    patterns = [
        "ml-*.zip", "movielens*.zip",
        "ml-latest-small", "ml-25m", "ml-1m", "ml-100k",
    ]
    for directory in search_dirs:
        if not directory.exists():
            continue
        for pattern in patterns:
            for match in directory.glob(pattern):
                candidates.append(str(match))
    return candidates[:8]  # cap at 8 to keep output readable


def run(data_dir: str | Path | None = None, mode: str = "upsert") -> IngestReport:
    """Ingest all recognisable MovieLens files from *data_dir*.

    Args:
        data_dir: path to a .zip file, extracted directory, or single file.
                  Relative paths are resolved from the current working
                  directory.  Falls back to env var MOVIELENS_DATA_DIR,
                  then /data/movielens.
        mode:     conflict strategy for duplicate rows:
                    "upsert"  — update existing rows (default, safe to re-run)
                    "skip"    — silently ignore duplicates, keep existing data
                    "replace" — delete existing rows first, full re-load

    Returns:
        IngestReport with per-table row counts, skipped files, and errors.

    Raises:
        FileNotFoundError: if the resolved path does not exist, with a
                           hint listing nearby MovieLens files if any.
        RuntimeError:      if any file failed to ingest.
    """
    # ── Resolve the source path ───────────────────────────────────────────
    raw_path = Path(data_dir) if data_dir else Path(
        os.environ.get("MOVIELENS_DATA_DIR", "/data/movielens")
    )

    # expanduser handles ~ ; resolve converts relative → absolute
    resolved = raw_path.expanduser().resolve()

    # ── Diagnostic logging so the user can see exactly what path is used ──
    logger.info("[ingest] Input path    : %s", raw_path)
    logger.info("[ingest] Resolved path : %s", resolved)
    logger.info("[ingest] Working dir   : %s", Path.cwd())
    logger.info("[ingest] Path exists   : %s", resolved.exists())

    if resolved.exists():
        kind = "file" if resolved.is_file() else "directory"
        logger.info("[ingest] Path type     : %s", kind)
        if resolved.is_file():
            size_mb = resolved.stat().st_size / 1_048_576
            logger.info("[ingest] File size     : %.1f MB", size_mb)
    else:
        # Path does not exist — give an actionable error message
        candidates = _find_candidates(raw_path)
        hint = ""
        if candidates:
            hint = (
                "\n\n  Nearby MovieLens files found:\n"
                + "\n".join(f"    {c}" for c in candidates)
                + "\n\n  Pass one of these with --data-dir or set "
                  "MOVIELENS_DATA_DIR in .env"
            )
        raise FileNotFoundError(
            f"\n[ingest] Path not found: {resolved}"
            f"\n  You passed           : {raw_path!r}"
            f"\n  Working directory    : {Path.cwd()}"
            f"\n"
            f"\n  How to fix:"
            f"\n    python main.py --stages ingest --data-dir /full/path/to/ml-1m.zip"
            f"\n    python main.py --stages ingest --data-dir ~/Downloads/ml-1m.zip"
            f"\n    # or set it permanently:"
            f"\n    echo 'MOVIELENS_DATA_DIR=/full/path/to/ml-1m.zip' >> .env"
            f"{hint}"
        )

    # ── Run ingest ────────────────────────────────────────────────────────
    logger.info("[ingest] Mode          : %s", mode)
    report = ingest_all(resolved, mode=mode)

    if report.errors:
        error_summary = "\n".join(f"  {f}: {e}" for f, e in report.errors.items())
        raise RuntimeError(
            f"[ingest] {len(report.errors)} file(s) failed to ingest:\n{error_summary}"
        )

    if report.warnings:
        for w in report.warnings:
            logger.warning("[ingest] %s", w)

    logger.info(
        "[ingest] Done — variant=%s  total_rows=%d  tables=%s",
        report.variant, report.total_rows, list(report.ingested.keys()),
    )
    return report
