"""ETL Stage 1: Ingest MovieLens data into the raw.* PostgreSQL schema.

Handles every official MovieLens dataset variant automatically via a chunked stream.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.connection import get_session
from db.models import (
    RawGenomeScore, RawLink, RawMovie, RawRating, RawTag, TriggerWatermark,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class IngestReport:
    source:   str             = ""
    variant:  str             = "unknown"
    ingested: dict[str, int]  = field(default_factory=lambda: defaultdict(int))
    skipped:  dict[str, str]  = field(default_factory=dict)
    errors:   dict[str, str]  = field(default_factory=dict)
    warnings: list[str]       = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return sum(self.ingested.values())

    @property
    def ok(self) -> bool:
        """True when both ratings and movies were successfully ingested."""
        return "raw.ratings" in self.ingested and "raw.movies" in self.ingested
    
    @property
    def counts(self) -> dict[str, int]:
        return self.ingested


# ---------------------------------------------------------------------------
# Dataset Manifest & Mapping Config
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    variant: str
    target_table: str
    model_cls: Any
    delimiter: str
    columns: list[str]
    rename_map: dict[str, str] = field(default_factory=dict)
    usecols: list[int] | None = None
    engine: str = "c"
    encoding: str = "utf-8"
    header: int | str | None = "infer"

DATASET_MANIFEST = {
    # ── Modern CSVs (ml-latest-small, ml-25m, ml-latest) ──
    "ratings.csv": DatasetSpec("modern", "raw.ratings", RawRating, ",", ["user_id", "movie_id", "rating", "timestamp"], {"userId": "user_id", "movieId": "movie_id"}),
    "movies.csv": DatasetSpec("modern", "raw.movies", RawMovie, ",", ["movie_id", "title", "genres"], {"movieId": "movie_id"}),
    "tags.csv": DatasetSpec("modern", "raw.tags", RawTag, ",", ["user_id", "movie_id", "tag", "timestamp"], {"userId": "user_id", "movieId": "movie_id"}),
    "links.csv": DatasetSpec("modern", "raw.links", RawLink, ",", ["movie_id", "imdb_id", "tmdb_id"], {"movieId": "movie_id", "imdbId": "imdb_id", "tmdbId": "tmdb_id"}),
    "genome-scores.csv": DatasetSpec("modern", "raw.genome_scores", RawGenomeScore, ",", ["movie_id", "tag_id", "relevance"], {"movieId": "movie_id", "tagId": "tag_id"}),
    
    # ── Legacy ml-1m & ml-10m ──
    "ratings.dat": DatasetSpec("ml-1m", "raw.ratings", RawRating, "::", ["user_id", "movie_id", "rating", "timestamp"], engine="python", encoding="latin-1", header=None),
    "movies.dat": DatasetSpec("ml-1m", "raw.movies", RawMovie, "::", ["movie_id", "title", "genres"], engine="python", encoding="latin-1", header=None),
    
    # ── Legacy ml-100k ──
    "u.data": DatasetSpec("ml-100k", "raw.ratings", RawRating, "\t", ["user_id", "movie_id", "rating", "timestamp"], header=None),
    "u.item": DatasetSpec("ml-100k", "raw.movies", RawMovie, "|", ["movie_id", "title"], usecols=[0, 1], encoding="latin-1", header=None),
}

# The unique constraint targets from your models.py
CONFLICT_TARGET_KEYS = {
    "raw.ratings":       ["user_id", "movie_id"],
    "raw.movies":        ["movie_id"],
    "raw.tags":          ["user_id", "movie_id", "tag", "timestamp"],
    "raw.links":         ["movie_id"],
    "raw.genome_scores": ["movie_id", "tag_id"],
}


# ---------------------------------------------------------------------------
# Core Utilities
# ---------------------------------------------------------------------------

def _detect_variant(files: list[Path]) -> str:
    names = {f.name.lower() for f in files}
    if "u.data" in names: return "ml-100k"
    if "ratings.dat" in names: return "ml-1m / ml-10m"
    if "genome-scores.csv" in names: return "ml-25m"
    if "ratings.csv" in names: return "modern (small/latest)"
    return "unknown"

def _extract_zip(zip_path: Path, dest: Path) -> list[Path]:
    logger.info("[ingest] Extracting %s ...", zip_path.name)
    with zipfile.ZipFile(zip_path) as zf:
        bad = [m for m in zf.namelist() if m.startswith("/") or ".." in m]
        if bad:
            raise ValueError(f"Zip contains unsafe paths: {bad}")
        zf.extractall(dest)
    return [p for p in dest.rglob("*") if p.is_file()]


# ---------------------------------------------------------------------------
# Chunked Parser & Writer
# ---------------------------------------------------------------------------

def _process_file_in_chunks(path: Path, spec: DatasetSpec, mode: str, chunksize: int = 100000) -> int:
    """Streams large dataset files in memory-safe chunks."""
    kwargs = {
        "sep": spec.delimiter,
        "encoding": spec.encoding,
        "engine": spec.engine,
        "header": spec.header,
        "chunksize": chunksize,
    }
    
    if spec.engine == "c":
        kwargs["low_memory"] = False
        
    # Legacy files need explicit column names provided
    if spec.header is None:
        kwargs["names"] = spec.columns
    if spec.usecols:
        kwargs["usecols"] = spec.usecols

    # Ensure link IDs stay strings to preserve leading zeros
    if path.name.lower() == "links.csv":
        kwargs["dtype"] = str

    table_obj = spec.model_cls.__mapper__.local_table
    conflict_cols = CONFLICT_TARGET_KEYS.get(spec.target_table, [])
    primary_keys = CONFLICT_TARGET_KEYS.get(spec.target_table, [])

    total_inserted = 0
    first_chunk = True

    with get_session() as session:
        # If replace mode, clear table once at the beginning
        if mode == "replace":
            session.execute(text(f"DELETE FROM {spec.target_table}"))

        for chunk_idx, df in enumerate(pd.read_csv(path, **kwargs)):
            if df.empty:
                continue

            if spec.rename_map:
                df = df.rename(columns=spec.rename_map)

            if path.name.lower() == "u.item":
                df["genres"] = "(no genres listed)"

            if path.name.lower() == "links.csv":
                df["movie_id"] = df["movie_id"].astype(int)
                for col in ["imdb_id", "tmdb_id"]:
                    if col in df.columns:
                        df[col] = df[col].where(df[col].notna() & (df[col] != "nan"), None)

            if primary_keys:
                df = df.dropna(subset=primary_keys)

            if df.empty:
                continue

            records = df.to_dict(orient="records")
            update_col_names = [
                c.name for c in table_obj.columns 
                if c.name not in set(conflict_cols) and c.name != "id" and not c.primary_key
            ]

            insert_stmt = pg_insert(table_obj).values(records)
            
            if mode == "upsert" and conflict_cols and update_col_names:
                update_dict = {col: getattr(insert_stmt.excluded, col) for col in update_col_names}
                stmt = insert_stmt.on_conflict_do_update(index_elements=conflict_cols, set_=update_dict)
            else:
                stmt = insert_stmt.on_conflict_do_nothing()

            session.execute(stmt)
            total_inserted += len(records)
            
            if chunk_idx % 10 == 0:
                logger.info("[ingest]   → Processed chunk %d (%d total rows so far) for %s", chunk_idx, total_inserted, path.name)

        # Update watermark / metadata
        result = session.execute(text(f"SELECT MAX(inserted_at) FROM {spec.target_table}")).scalar()
        wm = session.get(TriggerWatermark, spec.target_table)
        if not wm:
            wm = TriggerWatermark(table_name=spec.target_table)
            session.add(wm)
        wm.last_inserted_at = result

    return total_inserted


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def ingest_all(source: str | Path, mode: str = "upsert") -> IngestReport:
    """Ingests a MovieLens dataset into PostgreSQL handling all validation."""
    source = Path(source)
    report = IngestReport(source=str(source))

    if not source.exists():
        raise FileNotFoundError(f"[ingest] Source not found: {source}")

    tmp_dir: str | None = None
    try:
        # 1. Single Zip file provided directly
        if source.is_file() and source.suffix.lower() == ".zip":
            tmp_dir = tempfile.mkdtemp(prefix="movielens_ingest_")
            all_files = _extract_zip(source, Path(tmp_dir))

        # 2. Directory provided
        elif source.is_dir():
            all_files = [p for p in source.rglob("*") if p.is_file()]
            known_targets = set(DATASET_MANIFEST.keys())
            
            # Check if directory already contains unzipped dataset files
            has_unzipped = any(p.name.lower() in known_targets for p in all_files)

            # If no unzipped dataset files are found, extract zip archives inside the directory
            if not has_unzipped:
                zip_files = sorted([p for p in source.glob("*.zip")])
                if zip_files:
                    tmp_dir = tempfile.mkdtemp(prefix="movielens_ingest_")
                    all_files = []
                    for zf in zip_files:
                        logger.info("[ingest] Auto-extracting found archive: %s", zf.name)
                        extracted = _extract_zip(zf, Path(tmp_dir))
                        all_files.extend(extracted)
        else:
            all_files = [source]

        report.variant = _detect_variant(all_files)
        logger.info("[ingest] Source: %s  Variant: %s  Files: %d", source.name, report.variant, len(all_files))

        for path in sorted(all_files):
            filename = path.name.lower()
            
            if filename in ["readme.txt", "readme", "users.dat"] or path.suffix.lower() in [".sh", ".py", ".md"]:
                report.skipped[path.name] = "Non-essential metadata"
                continue

            spec = DATASET_MANIFEST.get(filename)
            if not spec:
                report.skipped[path.name] = "Unrecognized data file format"
                logger.debug("[ingest] Unrecognized file: %s", path.name)
                continue

            try:
                logger.info("[ingest] Streaming & parsing %s -> %s", path.name, spec.target_table)
                count = _process_file_in_chunks(path, spec, mode)
                report.ingested[spec.target_table] += count
                logger.info("[ingest]   → Successfully ingested %d rows into %s", count, spec.target_table)

            except Exception as e:
                report.errors[path.name] = str(e)
                logger.error("[ingest] ERROR processing %s: %s", path.name, str(e))

    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if not report.ok:
        report.warnings.append("Required tables ('raw.ratings', 'raw.movies') not fully ingested.")

    return report
