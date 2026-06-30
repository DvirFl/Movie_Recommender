"""ETL Stage 1: Ingest MovieLens data into the raw.* PostgreSQL schema.

Handles every official MovieLens dataset variant automatically:

  Variant          Extension   Delimiter   Files
  ───────────────  ──────────  ──────────  ──────────────────────────────────────
  ml-latest-small  .csv        comma       ratings, movies, tags, links
  ml-25m           .csv        comma       ratings, movies, tags, links,
                                           genome-scores, genome-tags
  ml-latest        .csv        comma       same as ml-25m
  ml-1m            .dat        ::          ratings, movies, users (ignored)
  ml-100k          .data/.item tab/pipe    u.data (ratings), u.item (movies)

Accepted input formats
──────────────────────
  • A .zip file   → extracted to a temp dir, then processed
  • A directory   → scanned for recognised files
  • A single file → treated as whichever file the name matches

Returns an IngestReport detailing what was found, ingested, skipped, rejected.
"""
from __future__ import annotations

import csv
import logging
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
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
    ingested: dict[str, int]  = field(default_factory=dict)   # table -> rows
    skipped:  dict[str, str]  = field(default_factory=dict)   # file  -> reason
    errors:   dict[str, str]  = field(default_factory=dict)   # file  -> error
    warnings: list[str]       = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return sum(self.ingested.values())

    @property
    def ok(self) -> bool:
        """True when both ratings and movies were successfully ingested."""
        return "raw.ratings" in self.ingested and "raw.movies" in self.ingested


# ---------------------------------------------------------------------------
# Known file sets
# ---------------------------------------------------------------------------

# Files present in the download but not needed by the ORM
_SKIP_FILES = {
    "readme.txt", "readme.html", "readme",
    "genome-tags.csv",   # tag vocabulary — not in schema
    "users.dat",         # ml-1m user demographics — not in schema
    "u.user",            # ml-100k user demographics
    "u.genre",           # ml-100k genre vocabulary
    "u.occupation",      # ml-100k occupation list
    "u.info",            # ml-100k dataset info
    "u1.base", "u1.test",  # ml-100k pre-split folds
    "u2.base", "u2.test",
    "u3.base", "u3.test",
    "u4.base", "u4.test",
    "u5.base", "u5.test",
    "ua.base", "ua.test",
    "ub.base", "ub.test",
}

# Extensions that are never data files
_IGNORE_EXTENSIONS = {".md", ".txt", ".html", ".pdf", ".zip", ".md5",
                      ".json", ".sh", ".py", ".npz"}


# ---------------------------------------------------------------------------
# Variant detection
# ---------------------------------------------------------------------------

def _detect_variant(files: list[Path]) -> str:
    names = {f.name.lower() for f in files}
    if "u.data" in names or "u.item" in names:
        return "ml-100k"
    if "ratings.dat" in names or "movies.dat" in names:
        return "ml-1m"
    if "genome-scores.csv" in names:
        return "ml-25m"
    if "ratings.csv" in names:
        return "ml-latest-small"
    return "unknown"


# ---------------------------------------------------------------------------
# Zip extraction
# ---------------------------------------------------------------------------

def _extract_zip(zip_path: Path, dest: Path) -> list[Path]:
    logger.info("[ingest] Extracting %s ...", zip_path.name)
    with zipfile.ZipFile(zip_path) as zf:
        # Safety check — reject zips with path-traversal entries
        bad = [m for m in zf.namelist() if m.startswith("/") or ".." in m]
        if bad:
            raise ValueError(f"Zip contains unsafe paths: {bad}")
        zf.extractall(dest)
    files = [p for p in dest.rglob("*") if p.is_file()]
    logger.info("[ingest] Extracted %d file(s).", len(files))
    return files


# ---------------------------------------------------------------------------
# Column guard
# ---------------------------------------------------------------------------

def _require_columns(path: Path, df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name}: missing columns {sorted(missing)}. "
            f"Found: {sorted(df.columns)}"
        )


# ---------------------------------------------------------------------------
# Readers — one per file type per format variant
# ---------------------------------------------------------------------------

# ── Modern CSV (ml-latest-small / ml-25m / ml-latest) ─────────────────────

def _read_ratings_modern(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    df = df.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    _require_columns(path, df, {"user_id", "movie_id", "rating", "timestamp"})
    return df[["user_id", "movie_id", "rating", "timestamp"]].astype(
        {"user_id": int, "movie_id": int, "rating": float, "timestamp": int}
    )


def _read_movies_modern(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    df = df.rename(columns={"movieId": "movie_id"})
    _require_columns(path, df, {"movie_id", "title", "genres"})
    return df[["movie_id", "title", "genres"]].astype(
        {"movie_id": int, "title": str, "genres": str}
    )


def _read_tags_modern(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    df = df.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    _require_columns(path, df, {"user_id", "movie_id", "tag", "timestamp"})
    df = df.dropna(subset=["tag"])
    return df[["user_id", "movie_id", "tag", "timestamp"]].astype(
        {"user_id": int, "movie_id": int, "tag": str, "timestamp": int}
    )


def _read_links_modern(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", dtype=str, low_memory=False)
    df = df.rename(columns={"movieId": "movie_id", "imdbId": "imdb_id", "tmdbId": "tmdb_id"})
    _require_columns(path, df, {"movie_id"})
    df["movie_id"] = df["movie_id"].astype(int)
    for col in ("imdb_id", "tmdb_id"):
        if col not in df.columns:
            df[col] = None
        else:
            df[col] = df[col].where(df[col].notna() & (df[col] != "nan"), other=None)
    return df[["movie_id", "imdb_id", "tmdb_id"]]


def _read_genome_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    df = df.rename(columns={"movieId": "movie_id", "tagId": "tag_id"})
    _require_columns(path, df, {"movie_id", "tag_id", "relevance"})
    return df[["movie_id", "tag_id", "relevance"]].astype(
        {"movie_id": int, "tag_id": int, "relevance": float}
    )


# ── ml-1m (.dat, :: delimiter) ────────────────────────────────────────────

def _read_ratings_1m(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="::", header=None, engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    return df.astype({"user_id": int, "movie_id": int, "rating": float, "timestamp": int})


def _read_movies_1m(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="::", header=None, engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    return df[["movie_id", "title", "genres"]].astype(
        {"movie_id": int, "title": str, "genres": str}
    )


# ── ml-100k (tab / pipe delimited, no headers) ────────────────────────────

def _read_ratings_100k(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    return df.astype({"user_id": int, "movie_id": int, "rating": float, "timestamp": int})


def _read_movies_100k(path: Path) -> pd.DataFrame:
    """u.item has 24 pipe-delimited columns; only the first two are needed."""
    col_names = ["movie_id", "title", "release_date", "video_date", "imdb_url"] + \
                [f"genre_{i}" for i in range(19)]
    df = pd.read_csv(
        path, sep="|", header=None, encoding="latin-1",
        names=col_names, on_bad_lines="skip",
    )
    df = df[["movie_id", "title"]].copy()
    # 100k genres are one-hot columns — convert to a placeholder string
    df["genres"] = "(no genres listed)"
    return df.astype({"movie_id": int, "title": str, "genres": str})


# ---------------------------------------------------------------------------
# DB writers
# ---------------------------------------------------------------------------

def _update_watermark(session, table_name: str) -> None:
    from sqlalchemy import text
    result = session.execute(
        text(f"SELECT MAX(inserted_at) FROM {table_name}")
    ).scalar()
    wm = session.get(TriggerWatermark, table_name)
    if wm is None:
        wm = TriggerWatermark(table_name=table_name)
        session.add(wm)
    wm.last_inserted_at = result


# Columns that serve as the natural key for each table.
# Used to build  ON CONFLICT (key_cols) DO UPDATE  upsert statements.
_TABLE_CONFLICT_KEYS: dict[str, list[str]] = {
    "raw.ratings":       ["user_id", "movie_id", "timestamp"],
    "raw.movies":        ["movie_id"],
    "raw.tags":          ["user_id", "movie_id", "tag", "timestamp"],
    "raw.links":         ["movie_id"],
    "raw.genome_scores": ["movie_id", "tag_id"],
}


# PostgreSQL hard limit is 65535 parameters per statement.
# We stay well below it: 500 rows * ~6 columns = ~3000 params per batch.
_BATCH_SIZE = 500


def _write(
    model_cls,
    records: list[dict],
    table_name: str,
    mode: str = "upsert",
) -> int:
    """Write *records* to *table_name* in batches using the requested conflict strategy.

    Args:
        model_cls:  SQLAlchemy ORM model class.
        records:    list of dicts — one per row.
        table_name: schema-qualified table name (e.g. "raw.ratings").
        mode:       conflict strategy:
                      "upsert"  — INSERT ... ON CONFLICT DO UPDATE (default).
                                  Safe to re-run; existing rows are updated.
                      "skip"    — INSERT ... ON CONFLICT DO NOTHING.
                                  Keeps existing rows, silently ignores dupes.
                      "replace" — DELETE existing rows first, then INSERT.
                                  Full re-load of the table.

    Returns:
        Total number of records written.
    """
    from sqlalchemy import text

    if not records:
        return 0

    tbl_obj       = model_cls.__mapper__.local_table
    conflict_cols = _TABLE_CONFLICT_KEYS.get(table_name, [])

    # Pre-compute which columns to update (for upsert mode).
    # We use column.name strings here; the actual excluded.* refs are
    # rebuilt per-batch below because they must reference that batch's
    # insert statement.
    key_set = set(conflict_cols)
    update_col_names: list[str] = []
    if conflict_cols and mode == "upsert":
        update_col_names = [
            c.name for c in tbl_obj.columns
            if c.name not in key_set
            and c.name != "id"
            and not c.primary_key
        ]
    update_cols: dict = {}   # populated per-batch

    total = 0
    n_batches = (len(records) + _BATCH_SIZE - 1) // _BATCH_SIZE

    with get_session() as session:

        # replace: DELETE once before the first batch
        if mode == "replace":
            session.execute(text(f"DELETE FROM {table_name}"))

        for batch_num, start in enumerate(range(0, len(records), _BATCH_SIZE)):
            batch = records[start: start + _BATCH_SIZE]
            if n_batches > 1:
                logger.debug(
                    "[ingest] %s batch %d/%d (%d rows)...",
                    table_name, batch_num + 1, n_batches, len(batch),
                )

            try:
                if mode == "replace":
                    # After the initial DELETE, use plain insert for speed
                    session.bulk_insert_mappings(model_cls, batch)  # type: ignore

                elif mode == "skip" or not conflict_cols:
                    stmt = pg_insert(tbl_obj).values(batch).on_conflict_do_nothing()
                    session.execute(stmt)

                else:  # upsert
                    insert_stmt = pg_insert(tbl_obj).values(batch)
                    if update_col_names:
                        # Build excluded.* refs from this batch's insert statement
                        batch_update = {
                            name: getattr(insert_stmt.excluded, name)
                            for name in update_col_names
                        }
                        stmt = insert_stmt.on_conflict_do_update(
                            index_elements=conflict_cols,
                            set_=batch_update,
                        )
                    else:
                        stmt = insert_stmt.on_conflict_do_nothing()
                    session.execute(stmt)

                total += len(batch)

            except Exception as exc:
                # Log the actual error clearly BEFORE SQLAlchemy adds the
                # giant parameter dump to the exception chain
                logger.error(
                    "[ingest] Batch %d/%d failed for %s (rows %d-%d): %s",
                    batch_num + 1, n_batches, table_name,
                    start, start + len(batch) - 1,
                    str(exc).split("\n")[0],   # first line only — no param dump
                )
                raise

        _update_watermark(session, table_name)

    return total


# ---------------------------------------------------------------------------
# File classifier
# ---------------------------------------------------------------------------

class _FileClassifier:
    """Maps a path to (action, format_hint).

    action values:
      'ratings' | 'movies' | 'tags' | 'links' | 'genome_scores'
      'skip'    — known non-data file, silently ignored
      'reject'  — unrecognised file, logged as warning
    """

    def classify(self, path: Path) -> tuple[str, str]:
        name = path.name.lower()
        ext  = path.suffix.lower()

        if path.is_dir():
            return "skip", "directory"
        if ext in _IGNORE_EXTENSIONS:
            return "skip", f"non-data extension ({ext})"
        if name in _SKIP_FILES:
            return "skip", "known non-essential file"

        # Modern CSV
        if name == "ratings.csv":       return "ratings",       "modern"
        if name == "movies.csv":        return "movies",        "modern"
        if name == "tags.csv":          return "tags",          "modern"
        if name == "links.csv":         return "links",         "modern"
        if name == "genome-scores.csv": return "genome_scores", "modern"

        # ml-1m
        if name == "ratings.dat":       return "ratings",       "ml-1m"
        if name == "movies.dat":        return "movies",        "ml-1m"

        # ml-100k
        if name == "u.data":            return "ratings",       "ml-100k"
        if name == "u.item":            return "movies",        "ml-100k"

        return "reject", (
            f"unrecognised file '{path.name}'. "
            f"Expected one of: ratings.csv, movies.csv, tags.csv, links.csv, "
            f"genome-scores.csv, ratings.dat, movies.dat, u.data, u.item."
        )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_READERS = {
    ("ratings",       "modern"):  _read_ratings_modern,
    ("ratings",       "ml-1m"):   _read_ratings_1m,
    ("ratings",       "ml-100k"): _read_ratings_100k,
    ("movies",        "modern"):  _read_movies_modern,
    ("movies",        "ml-1m"):   _read_movies_1m,
    ("movies",        "ml-100k"): _read_movies_100k,
    ("tags",          "modern"):  _read_tags_modern,
    ("links",         "modern"):  _read_links_modern,
    ("genome_scores", "modern"):  _read_genome_scores,
}

_WRITERS = {
    "ratings":       (RawRating,       "raw.ratings"),
    "movies":        (RawMovie,        "raw.movies"),
    "tags":          (RawTag,          "raw.tags"),
    "links":         (RawLink,         "raw.links"),
    "genome_scores": (RawGenomeScore,  "raw.genome_scores"),
}


def _ingest_file(path: Path, action: str, fmt: str, report: IngestReport, mode: str = "upsert") -> None:
    reader_key = (action, fmt)
    if reader_key not in _READERS:
        report.errors[path.name] = f"No reader for ({action}, {fmt})"
        logger.error("[ingest] No reader for %s (%s, %s)", path.name, action, fmt)
        return
    try:
        logger.info("[ingest] Reading %s as (%s, %s)...", path.name, action, fmt)
        df = _READERS[reader_key](path)
        model_cls, table_name = _WRITERS[action]
        count = _write(model_cls, df.to_dict(orient="records"), table_name, mode=mode)
        report.ingested[table_name] = count
        logger.info("[ingest]   → %d rows into %s", count, table_name)
    except Exception as exc:
        report.errors[path.name] = str(exc)
        logger.error("[ingest] ERROR on %s: %s", path.name, exc)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest_all(source: str | Path, mode: str = "upsert") -> IngestReport:
    """Ingest a MovieLens dataset from a zip, directory, or single file.

    Args:
        source: path to a .zip file, an extracted directory, or a single
                data file.  Set MOVIELENS_DATA_DIR to point here.
        mode:   conflict strategy — "upsert" (default), "skip", or "replace".
                See _write() for full semantics.

    Returns:
        IngestReport with per-table row counts, skipped files, and errors.

    Raises:
        FileNotFoundError: if *source* does not exist.
        ValueError:        if the zip contains unsafe paths.
    """
    source = Path(source)
    report = IngestReport(source=str(source))

    if not source.exists():
        raise FileNotFoundError(
            f"[ingest] Source not found: {source}\n"
            f"Set MOVIELENS_DATA_DIR to the path of your downloaded "
            f"MovieLens zip or extracted directory."
        )

    tmp_dir: str | None = None
    try:
        # ── Resolve to a flat list of files ──────────────────────────────
        if source.suffix.lower() == ".zip":
            tmp_dir = tempfile.mkdtemp(prefix="movielens_ingest_")
            all_files = _extract_zip(source, Path(tmp_dir))
        elif source.is_dir():
            all_files = [p for p in source.rglob("*") if p.is_file()]
        else:
            all_files = [source]

        if not all_files:
            raise ValueError(f"No files found in {source}.")

        # ── Detect variant ────────────────────────────────────────────────
        report.variant = _detect_variant(all_files)
        logger.info("[ingest] Source: %s  Variant: %s  Files: %d",
                    source.name, report.variant, len(all_files))

        # ── Classify and process ──────────────────────────────────────────
        classifier = _FileClassifier()
        for path in sorted(all_files):
            action, detail = classifier.classify(path)

            if action == "skip":
                report.skipped[path.name] = detail
                logger.debug("[ingest] Skip %s: %s", path.name, detail)

            elif action == "reject":
                report.skipped[path.name] = f"REJECTED: {detail}"
                logger.warning("[ingest] Rejected %s: %s", path.name, detail)

            else:
                _ingest_file(path, action, detail, report, mode=mode)

    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Sanity check ─────────────────────────────────────────────────────
    if not report.ok:
        missing = [t for t in ("raw.ratings", "raw.movies")
                   if t not in report.ingested]
        msg = f"Required tables not ingested: {missing}. Downstream stages will fail."
        report.warnings.append(msg)
        logger.warning("[ingest] %s", msg)

    logger.info("[ingest] Complete — %s", report.ingested)
    return report
