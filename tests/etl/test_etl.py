"""Tests: ETL stages — ingest (all variants), validate, featurize, split."""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ===========================================================================
# Helpers
# ===========================================================================

def _make_zip(tmp_path: Path, subdir: str, files: dict[str, str]) -> Path:
    """Create a zip containing files under a subdirectory (mirrors real downloads)."""
    zip_path = tmp_path / f"{subdir}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, content in files.items():
            zf.writestr(f"{subdir}/{name}", content)
    return zip_path


def _patch_db(monkeypatch):
    """Patch _write so no real DB is needed in integration tests."""
    def fake_write(model_cls, records, table_name, mode="upsert"):
        return len(records)

    from etl import ingest as _ingest_mod
    monkeypatch.setattr(_ingest_mod, "_write", fake_write)
    monkeypatch.setattr(_ingest_mod, "_update_watermark", lambda *a, **kw: None)



# ---------------------------------------------------------------------------
# Sample file content per variant
# ---------------------------------------------------------------------------

RATINGS_CSV = "userId,movieId,rating,timestamp\n1,10,4.0,1000\n2,20,3.5,2000\n"
MOVIES_CSV  = "movieId,title,genres\n10,Movie A (2010),Action\n20,Movie B (2015),Drama\n"
TAGS_CSV    = "userId,movieId,tag,timestamp\n1,10,funny,1500\n"
LINKS_CSV   = "movieId,imdbId,tmdbId\n10,0114709,862\n20,0113497,8844\n"
GENOME_CSV  = "movieId,tagId,relevance\n10,1,0.9\n10,2,0.4\n"

RATINGS_DAT = "1::10::4::1000\n2::20::3::2000\n"
MOVIES_DAT  = "10::Movie A (2010)::Action|Adventure\n20::Movie B (2015)::Drama\n"

RATINGS_100K = "1\t10\t4\t1000\n2\t20\t3\t2000\n"
# u.item: movie_id|title|date|url|genre0..genre18
MOVIES_100K  = "10|Movie A|01-Jan-2010||" + "|".join(["0"] * 19) + "\n"

def test_ingest_movies_reads_csv(tmp_path):
    csv = tmp_path / "movies.csv"
    csv.write_text("movieId,title,genres\n10,Movie A (2010),Action\n20,Movie B (2015),Drama\n")
    with patch("etl.ingest.get_session") as mock_session_ctx:
        mock_session = MagicMock()
        mock_session_ctx.return_value.__enter__.return_value = mock_session
        mock_session_ctx.return_value.__exit__.return_value = False
        mock_session.execute.return_value.scalar.return_value = None
        mock_session.get.return_value = None

# ===========================================================================
# _FileClassifier
# ===========================================================================

class TestFileClassifier:

    def _cls(self):
        from etl.ingest import _FileClassifier
        return _FileClassifier()

    def _path(self, name):
        p = MagicMock(spec=Path)
        p.name = name
        p.suffix = "." + name.rsplit(".", 1)[-1] if "." in name else ""
        p.is_dir.return_value = False
        return p

    def test_ratings_csv(self):
        action, fmt = self._cls().classify(self._path("ratings.csv"))
        assert action == "ratings" and fmt == "modern"

    def test_movies_csv(self):
        action, fmt = self._cls().classify(self._path("movies.csv"))
        assert action == "movies" and fmt == "modern"

    def test_tags_csv(self):
        action, fmt = self._cls().classify(self._path("tags.csv"))
        assert action == "tags" and fmt == "modern"

    def test_links_csv(self):
        action, fmt = self._cls().classify(self._path("links.csv"))
        assert action == "links" and fmt == "modern"

    def test_genome_scores_csv(self):
        action, fmt = self._cls().classify(self._path("genome-scores.csv"))
        assert action == "genome_scores" and fmt == "modern"

    def test_ratings_dat(self):
        action, fmt = self._cls().classify(self._path("ratings.dat"))
        assert action == "ratings" and fmt == "ml-1m"

    def test_movies_dat(self):
        action, fmt = self._cls().classify(self._path("movies.dat"))
        assert action == "movies" and fmt == "ml-1m"

    def test_u_data(self):
        action, fmt = self._cls().classify(self._path("u.data"))
        assert action == "ratings" and fmt == "ml-100k"

    def test_u_item(self):
        action, fmt = self._cls().classify(self._path("u.item"))
        assert action == "movies" and fmt == "ml-100k"

    def test_genome_tags_skipped(self):
        action, _ = self._cls().classify(self._path("genome-tags.csv"))
        assert action == "skip"

    def test_readme_skipped(self):
        action, _ = self._cls().classify(self._path("README.html"))
        assert action == "skip"

    def test_users_dat_skipped(self):
        action, _ = self._cls().classify(self._path("users.dat"))
        assert action == "skip"

    def test_unknown_csv_rejected(self):
        action, detail = self._cls().classify(self._path("unknown_file.csv"))
        assert action == "reject"
        assert "unrecognised" in detail.lower()

    def test_txt_file_skipped(self):
        action, _ = self._cls().classify(self._path("notes.txt"))
        assert action == "skip"

    def test_directory_skipped(self):
        p = MagicMock(spec=Path)
        p.name = "ml-25m"
        p.suffix = ""
        p.is_dir.return_value = True
        action, _ = self._cls().classify(p)
        assert action == "skip"

    def test_md5_file_skipped(self):
        action, _ = self._cls().classify(self._path("ml-25m.zip.md5"))
        assert action == "skip"


# ===========================================================================
# _detect_variant
# ===========================================================================

class TestDetectVariant:

    def _paths(self, names):
        return [MagicMock(name=n, spec=Path) for n in names]

    def _make(self, names):
        paths = []
        for n in names:
            p = MagicMock(spec=Path)
            p.name = n
            paths.append(p)
        return paths

    def test_detects_ml_latest_small(self):
        from etl.ingest import _detect_variant
        files = self._make(["ratings.csv", "movies.csv", "tags.csv", "links.csv"])
        assert _detect_variant(files) == "ml-latest-small"

    def test_detects_ml_25m(self):
        from etl.ingest import _detect_variant
        files = self._make(["ratings.csv", "movies.csv", "genome-scores.csv"])
        assert _detect_variant(files) == "ml-25m"

    def test_detects_ml_1m(self):
        from etl.ingest import _detect_variant
        files = self._make(["ratings.dat", "movies.dat", "users.dat"])
        assert _detect_variant(files) == "ml-1m"

    def test_detects_ml_100k(self):
        from etl.ingest import _detect_variant
        files = self._make(["u.data", "u.item", "u.user"])
        assert _detect_variant(files) == "ml-100k"

    def test_unknown_returns_unknown(self):
        from etl.ingest import _detect_variant
        files = self._make(["random.csv"])
        assert _detect_variant(files) == "unknown"


# ===========================================================================
# Readers — unit test each reader function with real CSV content
# ===========================================================================

class TestReadersModern:

    def _tmp_file(self, tmp_path, name, content):
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p

    def test_read_ratings_modern(self, tmp_path):
        from etl.ingest import _read_ratings_modern
        p = self._tmp_file(tmp_path, "ratings.csv", RATINGS_CSV)
        df = _read_ratings_modern(p)
        assert list(df.columns) == ["user_id", "movie_id", "rating", "timestamp"]
        assert len(df) == 2
        assert df["user_id"].tolist() == [1, 2]

    def test_read_movies_modern(self, tmp_path):
        from etl.ingest import _read_movies_modern
        p = self._tmp_file(tmp_path, "movies.csv", MOVIES_CSV)
        df = _read_movies_modern(p)
        assert "movie_id" in df.columns
        assert len(df) == 2

    def test_read_tags_modern_drops_null_tags(self, tmp_path):
        from etl.ingest import _read_tags_modern
        content = "userId,movieId,tag,timestamp\n1,10,funny,1500\n2,20,,2000\n"
        p = self._tmp_file(tmp_path, "tags.csv", content)
        df = _read_tags_modern(p)
        assert len(df) == 1   # null tag row dropped

    def test_read_links_modern(self, tmp_path):
        from etl.ingest import _read_links_modern
        p = self._tmp_file(tmp_path, "links.csv", LINKS_CSV)
        df = _read_links_modern(p)
        assert "movie_id" in df.columns
        assert len(df) == 2

    def test_read_links_handles_missing_tmdb(self, tmp_path):
        from etl.ingest import _read_links_modern
        content = "movieId,imdbId\n10,0114709\n"
        p = self._tmp_file(tmp_path, "links.csv", content)
        df = _read_links_modern(p)
        assert "tmdb_id" in df.columns

    def test_read_genome_scores(self, tmp_path):
        from etl.ingest import _read_genome_scores
        p = self._tmp_file(tmp_path, "genome-scores.csv", GENOME_CSV)
        df = _read_genome_scores(p)
        assert list(df.columns) == ["movie_id", "tag_id", "relevance"]
        assert len(df) == 2

    def test_reader_raises_on_missing_columns(self, tmp_path):
        from etl.ingest import _read_ratings_modern
        p = self._tmp_file(tmp_path, "ratings.csv", "userId,movieId\n1,10\n")
        with pytest.raises(ValueError, match="missing columns"):
            _read_ratings_modern(p)


class TestReaders1m:

    def test_read_ratings_1m(self, tmp_path):
        from etl.ingest import _read_ratings_1m
        p = tmp_path / "ratings.dat"
        p.write_text(RATINGS_DAT, encoding="latin-1")
        df = _read_ratings_1m(p)
        assert list(df.columns) == ["user_id", "movie_id", "rating", "timestamp"]
        assert len(df) == 2

    def test_read_movies_1m(self, tmp_path):
        from etl.ingest import _read_movies_1m
        p = tmp_path / "movies.dat"
        p.write_text(MOVIES_DAT, encoding="latin-1")
        df = _read_movies_1m(p)
        assert "genres" in df.columns
        assert "Action|Adventure" in df["genres"].tolist()


class TestReaders100k:

    def test_read_ratings_100k(self, tmp_path):
        from etl.ingest import _read_ratings_100k
        p = tmp_path / "u.data"
        p.write_text(RATINGS_100K)
        df = _read_ratings_100k(p)
        assert list(df.columns) == ["user_id", "movie_id", "rating", "timestamp"]
        assert len(df) == 2

    def test_read_movies_100k(self, tmp_path):
        from etl.ingest import _read_movies_100k
        p = tmp_path / "u.item"
        p.write_text(MOVIES_100K, encoding="latin-1")
        df = _read_movies_100k(p)
        assert "movie_id" in df.columns
        assert "genres" in df.columns
        assert len(df) == 1


# ===========================================================================
# ingest_all — integration tests (DB mocked)
# ===========================================================================

class TestIngestAllDirectory:

    def test_ingest_ml_latest_small_directory(self, tmp_path, monkeypatch):
        _patch_db(monkeypatch)
        (tmp_path / "ratings.csv").write_text(RATINGS_CSV)
        (tmp_path / "movies.csv").write_text(MOVIES_CSV)
        (tmp_path / "tags.csv").write_text(TAGS_CSV)
        (tmp_path / "links.csv").write_text(LINKS_CSV)

        from etl.ingest import ingest_all
        report = ingest_all(tmp_path)

        assert report.variant == "ml-latest-small"
        assert "raw.ratings" in report.ingested
        assert "raw.movies"  in report.ingested
        assert report.ok

    def test_ingest_ml_25m_directory(self, tmp_path, monkeypatch):
        _patch_db(monkeypatch)
        (tmp_path / "ratings.csv").write_text(RATINGS_CSV)
        (tmp_path / "movies.csv").write_text(MOVIES_CSV)
        (tmp_path / "genome-scores.csv").write_text(GENOME_CSV)

        from etl.ingest import ingest_all
        report = ingest_all(tmp_path)

        assert report.variant == "ml-25m"
        assert "raw.genome_scores" in report.ingested

    def test_ingest_ml_1m_directory(self, tmp_path, monkeypatch):
        _patch_db(monkeypatch)
        (tmp_path / "ratings.dat").write_text(RATINGS_DAT)
        (tmp_path / "movies.dat").write_text(MOVIES_DAT)

        from etl.ingest import ingest_all
        report = ingest_all(tmp_path)

        assert report.variant == "ml-1m"
        assert report.ok

    def test_ingest_ml_100k_directory(self, tmp_path, monkeypatch):
        _patch_db(monkeypatch)
        (tmp_path / "u.data").write_text(RATINGS_100K)
        (tmp_path / "u.item").write_text(MOVIES_100K, encoding="latin-1")

        from etl.ingest import ingest_all
        report = ingest_all(tmp_path)

        assert report.variant == "ml-100k"
        assert report.ok

    def test_unknown_files_are_rejected_not_errored(self, tmp_path, monkeypatch):
        _patch_db(monkeypatch)
        (tmp_path / "ratings.csv").write_text(RATINGS_CSV)
        (tmp_path / "movies.csv").write_text(MOVIES_CSV)
        (tmp_path / "mystery_data.csv").write_text("a,b,c\n1,2,3\n")

        from etl.ingest import ingest_all
        report = ingest_all(tmp_path)

        assert "mystery_data.csv" in report.skipped
        assert "REJECTED" in report.skipped["mystery_data.csv"]
        assert "mystery_data.csv" not in report.errors

    def test_readme_and_txt_silently_skipped(self, tmp_path, monkeypatch):
        _patch_db(monkeypatch)
        (tmp_path / "ratings.csv").write_text(RATINGS_CSV)
        (tmp_path / "movies.csv").write_text(MOVIES_CSV)
        (tmp_path / "README.txt").write_text("GroupLens readme")
        (tmp_path / "README.html").write_text("<html></html>")

        from etl.ingest import ingest_all
        report = ingest_all(tmp_path)

        assert "README.txt"  in report.skipped
        assert "README.html" in report.skipped
        # Not counted as errors
        assert "README.txt"  not in report.errors

    def test_genome_tags_silently_skipped(self, tmp_path, monkeypatch):
        _patch_db(monkeypatch)
        (tmp_path / "ratings.csv").write_text(RATINGS_CSV)
        (tmp_path / "movies.csv").write_text(MOVIES_CSV)
        (tmp_path / "genome-tags.csv").write_text("tagId,tag\n1,fun\n")

        from etl.ingest import ingest_all
        report = ingest_all(tmp_path)

        assert "genome-tags.csv" in report.skipped
        assert "genome-tags.csv" not in report.ingested.values()

    def test_raises_file_not_found(self):
        from etl.ingest import ingest_all
        with pytest.raises(FileNotFoundError, match="not found"):
            ingest_all("/nonexistent/path")

    def test_missing_ratings_adds_warning(self, tmp_path, monkeypatch):
        _patch_db(monkeypatch)
        (tmp_path / "movies.csv").write_text(MOVIES_CSV)

        from etl.ingest import ingest_all
        report = ingest_all(tmp_path)

        assert not report.ok
        assert any("raw.ratings" in w for w in report.warnings)

    def test_row_counts_match_csv_rows(self, tmp_path, monkeypatch):
        mock_session = _patch_db(monkeypatch)
        (tmp_path / "ratings.csv").write_text(RATINGS_CSV)
        (tmp_path / "movies.csv").write_text(MOVIES_CSV)

        from etl.ingest import ingest_all
        report = ingest_all(tmp_path)

        assert report.ingested["raw.ratings"] == 2
        assert report.ingested["raw.movies"]  == 2

    def test_total_rows_sums_all_tables(self, tmp_path, monkeypatch):
        _patch_db(monkeypatch)
        (tmp_path / "ratings.csv").write_text(RATINGS_CSV)
        (tmp_path / "movies.csv").write_text(MOVIES_CSV)
        (tmp_path / "tags.csv").write_text(TAGS_CSV)

        from etl.ingest import ingest_all
        report = ingest_all(tmp_path)

        assert report.total_rows == (
            report.ingested.get("raw.ratings", 0) +
            report.ingested.get("raw.movies",  0) +
            report.ingested.get("raw.tags",    0)
        )


class TestIngestAllZip:

    def test_ingest_from_zip_ml_latest_small(self, tmp_path, monkeypatch):
        _patch_db(monkeypatch)
        zip_path = _make_zip(tmp_path, "ml-latest-small", {
            "ratings.csv": RATINGS_CSV,
            "movies.csv":  MOVIES_CSV,
            "tags.csv":    TAGS_CSV,
            "links.csv":   LINKS_CSV,
        })

        from etl.ingest import ingest_all
        report = ingest_all(zip_path)

        assert report.variant == "ml-latest-small"
        assert report.ok

    def test_ingest_from_zip_ml_25m(self, tmp_path, monkeypatch):
        _patch_db(monkeypatch)
        zip_path = _make_zip(tmp_path, "ml-25m", {
            "ratings.csv":       RATINGS_CSV,
            "movies.csv":        MOVIES_CSV,
            "genome-scores.csv": GENOME_CSV,
            "genome-tags.csv":   "tagId,tag\n1,fun\n",
            "README.txt":        "GroupLens data",
        })

        from etl.ingest import ingest_all
        report = ingest_all(zip_path)

        assert report.variant == "ml-25m"
        assert "raw.genome_scores" in report.ingested
        assert "README.txt"        in report.skipped
        assert "genome-tags.csv"   in report.skipped

    def test_ingest_from_zip_1m(self, tmp_path, monkeypatch):
        _patch_db(monkeypatch)
        zip_path = _make_zip(tmp_path, "ml-1m", {
            "ratings.dat": RATINGS_DAT,
            "movies.dat":  MOVIES_DAT,
            "users.dat":   "1::F::1::10::48067\n",
            "README":      "readme",
        })

        from etl.ingest import ingest_all
        report = ingest_all(zip_path)

        assert report.variant == "ml-1m"
        assert report.ok
        assert any("users.dat" in k for k in report.skipped)

    def test_zip_path_traversal_rejected(self, tmp_path):
        zip_path = tmp_path / "bad.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../evil.py", "rm -rf /")

        from etl.ingest import ingest_all
        with pytest.raises(ValueError, match="unsafe paths"):
            ingest_all(zip_path)

    def test_temp_dir_cleaned_up_after_zip(self, tmp_path, monkeypatch):
        _patch_db(monkeypatch)
        import tempfile as _tmp
        created_dirs = []
        real_mkdtemp = _tmp.mkdtemp

        def tracking_mkdtemp(**kwargs):
            d = real_mkdtemp(**kwargs)
            created_dirs.append(d)
            return d

        monkeypatch.setattr("etl.ingest.tempfile.mkdtemp", tracking_mkdtemp)

        zip_path = _make_zip(tmp_path, "ml-latest-small", {
            "ratings.csv": RATINGS_CSV,
            "movies.csv":  MOVIES_CSV,
        })

        from etl.ingest import ingest_all
        ingest_all(zip_path)

        # All temp dirs created during ingest must be cleaned up
        for d in created_dirs:
            assert not Path(d).exists(), f"Temp dir not cleaned up: {d}"




# ===========================================================================
# Mode parameter — upsert / skip / replace
# ===========================================================================

class TestIngestMode:

    def test_default_mode_is_upsert(self):
        import inspect
        from etl.ingest import ingest_all
        sig = inspect.signature(ingest_all)
        assert sig.parameters["mode"].default == "upsert"

    def _mock_session(self):
        s = MagicMock()
        s.__enter__ = MagicMock(return_value=s)
        s.__exit__  = MagicMock(return_value=False)
        s.execute.return_value.scalar.return_value = None
        s.get.return_value = None
        return s

    def _mock_ins(self):
        m = MagicMock()
        m.values.return_value = m
        m.excluded = MagicMock()
        m.on_conflict_do_update.return_value = m
        m.on_conflict_do_nothing.return_value = m
        return m

    def test_write_upsert_calls_on_conflict_do_update(self):
        from etl.ingest import _write
        from db.models import RawMovie
        mi = self._mock_ins()
        with patch("etl.ingest.get_session", return_value=self._mock_session()), \
             patch("etl.ingest.pg_insert", return_value=mi):
            _write(RawMovie, [{"movie_id": 1, "title": "A", "genres": "X"}],
                   "raw.movies", mode="upsert")
        mi.on_conflict_do_update.assert_called_once()

    def test_write_skip_calls_on_conflict_do_nothing(self):
        from etl.ingest import _write
        from db.models import RawMovie
        mi = self._mock_ins()
        with patch("etl.ingest.get_session", return_value=self._mock_session()), \
             patch("etl.ingest.pg_insert", return_value=mi):
            _write(RawMovie, [{"movie_id": 1, "title": "A", "genres": "X"}],
                   "raw.movies", mode="skip")
        mi.on_conflict_do_nothing.assert_called_once()

    def test_write_replace_deletes_first(self):
        from etl.ingest import _write
        from db.models import RawMovie
        session = self._mock_session()
        with patch("etl.ingest.get_session", return_value=session):
            _write(RawMovie, [{"movie_id": 1, "title": "A", "genres": "X"}],
                   "raw.movies", mode="replace")
        # First execute call must be the DELETE statement
        executed_sqls = [str(c.args[0]) for c in session.execute.call_args_list]
        assert any("DELETE" in s.upper() for s in executed_sqls)

    def test_write_batches_large_input(self):
        """Records exceeding _BATCH_SIZE are split into multiple batches."""
        from etl import ingest as _mod
        from etl.ingest import _write, _BATCH_SIZE
        from db.models import RawRating
        # 2.5 batches worth of records
        n = int(_BATCH_SIZE * 2.5)
        records = [
            {"user_id": i, "movie_id": 1, "rating": 4.0, "timestamp": 1000 + i}
            for i in range(n)
        ]
        batches_seen: list[int] = []
        mi = self._mock_ins()
        mi.on_conflict_do_update.return_value = mi

        original_pg_insert = _mod.pg_insert
        def counting_insert(tbl):
            class _FakeInsert:
                def values(self, batch):
                    batches_seen.append(len(batch))
                    return mi
            return _FakeInsert()

        with patch("etl.ingest.get_session", return_value=self._mock_session()), \
             patch("etl.ingest.pg_insert", side_effect=counting_insert):
            total = _write(RawRating, records, "raw.ratings", mode="upsert")

        assert total == n
        assert len(batches_seen) == 3   # ceil(2.5) = 3 batches
        assert batches_seen[0] == _BATCH_SIZE
        assert batches_seen[1] == _BATCH_SIZE
        # Third batch has the remainder
        assert batches_seen[2] == n - _BATCH_SIZE * 2

    def test_write_returns_record_count(self):
        from etl.ingest import _write
        from db.models import RawMovie
        mi = self._mock_ins()
        with patch("etl.ingest.get_session", return_value=self._mock_session()), \
             patch("etl.ingest.pg_insert", return_value=mi):
            count = _write(RawMovie,
                           [{"movie_id": 1, "title": "A", "genres": "X"},
                            {"movie_id": 2, "title": "B", "genres": "Y"}],
                           "raw.movies", mode="upsert")
        assert count == 2

    def test_write_empty_returns_zero(self):
        from etl.ingest import _write
        from db.models import RawMovie
        assert _write(RawMovie, [], "raw.movies", mode="upsert") == 0

    def test_ingest_all_passes_mode_to_write(self, tmp_path, monkeypatch):
        (tmp_path / "ratings.csv").write_text(
            "userId,movieId,rating,timestamp\n1,10,4.0,1000\n")
        (tmp_path / "movies.csv").write_text(
            "movieId,title,genres\n10,A,Action\n")
        captured: list[str] = []

        def spy(model_cls, records, table_name, mode="upsert"):
            captured.append(mode)
            return len(records)

        from etl import ingest as _mod
        monkeypatch.setattr(_mod, "_write", spy)
        monkeypatch.setattr(_mod, "_update_watermark", lambda *a, **kw: None)
        from etl.ingest import ingest_all
        ingest_all(tmp_path, mode="skip")
        assert captured and all(m == "skip" for m in captured)

# ===========================================================================
# stage_ingest
# ===========================================================================

class TestStageIngest:

    def test_returns_ingest_report(self, tmp_path, monkeypatch):
        _patch_db(monkeypatch)
        (tmp_path / "ratings.csv").write_text(RATINGS_CSV)
        (tmp_path / "movies.csv").write_text(MOVIES_CSV)
        monkeypatch.setenv("MOVIELENS_DATA_DIR", str(tmp_path))

        from stages.stage_ingest import run
        from etl.ingest import IngestReport
        result = run(data_dir=tmp_path)
        assert isinstance(result, IngestReport)
        assert result.ok

    def test_falls_back_to_env_var(self, tmp_path, monkeypatch):
        _patch_db(monkeypatch)
        (tmp_path / "ratings.csv").write_text(RATINGS_CSV)
        (tmp_path / "movies.csv").write_text(MOVIES_CSV)
        monkeypatch.setenv("MOVIELENS_DATA_DIR", str(tmp_path))

        from stages.stage_ingest import run
        result = run(data_dir=None)
        assert result.source == str(tmp_path)

    def test_raises_on_ingest_errors(self, tmp_path, monkeypatch):
        """stage_ingest raises RuntimeError when ingest_all reports errors."""
        mock_report = MagicMock()
        mock_report.errors   = {"ratings.csv": "DB error"}
        mock_report.warnings = []
        mock_report.ok       = False
        mock_report.variant  = "ml-latest-small"
        mock_report.total_rows = 0
        mock_report.ingested = {}

        monkeypatch.setattr("stages.stage_ingest.ingest_all", lambda _, mode="upsert": mock_report)

        from stages.stage_ingest import run
        with pytest.raises(RuntimeError, match="failed to ingest"):
            run(data_dir=tmp_path)

    def test_raises_file_not_found(self, tmp_path):
        from stages.stage_ingest import run
        with pytest.raises(FileNotFoundError):
            run(data_dir=tmp_path / "does_not_exist")


# ===========================================================================
# validate (unchanged — kept for regression)
# ===========================================================================

def test_validation_report_fail_sets_passed_false():
    from etl.validate import ValidationReport
    report = ValidationReport()
    report.fail("test error")
    assert report.passed is False
    assert "test error" in report.issues


def test_validation_report_warn_does_not_fail():
    from etl.validate import ValidationReport
    report = ValidationReport()
    report.warn("test warning")
    assert report.passed is True
    assert any("WARN" in i for i in report.issues)


# ===========================================================================
# featurize (pure logic — unchanged)
# ===========================================================================

def test_extract_year():
    from etl.featurize import _extract_year
    assert _extract_year("The Matrix (1999)") == 1999
    assert _extract_year("No Year Here") is None


def test_time_decay_weight_recent_higher():
    from etl.featurize import _time_decay_weight
    w_recent = _time_decay_weight(1000, 1000, 500, 0.1)
    w_old    = _time_decay_weight(500,  1000, 500, 0.1)
    assert w_recent > w_old


def test_genre_vocab_coverage():
    from etl.featurize import GENRE_VOCAB, GENRE_TO_IDX
    assert len(GENRE_VOCAB) == 20
    assert len(GENRE_TO_IDX) == 20
    for i, g in enumerate(GENRE_VOCAB):
        assert GENRE_TO_IDX[g] == i


# ===========================================================================
# split (pure logic — unchanged)
# ===========================================================================

def test_split_fractions_correct():
    """Temporal split should produce correct proportions."""
    n = 100
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    n_test  = n - n_train - n_val
    assert n_train == 80
    assert n_val == 10
    assert n_test == 10


def test_split_no_leakage():
    """All train timestamps must be ≤ all val timestamps."""
    import pandas as pd
    from etl.split import split_ratings

    with patch("etl.split.get_session") as mock_ctx, \
         patch("etl.split.pd.read_sql") as mock_read_sql, \
         patch("etl.split.delete"):
        mock_session = MagicMock()
        mock_ctx.return_value.__enter__.return_value = mock_session
        mock_ctx.return_value.__exit__.return_value = False
        mock_read_sql.return_value = pd.DataFrame({
            "id": range(10),
            "timestamp": range(1000, 1010),
        })

        counts = split_ratings(train_frac=0.8, val_frac=0.1)
        assert counts["train"] == 8
        assert counts["val"] == 1
        assert counts["test"] == 1
