"""Tests: config/settings.py and config/env_loader.py."""
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# env_loader
# ---------------------------------------------------------------------------

class TestEnvLoader:

    def test_loads_simple_key_value(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_KEY=hello\n")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEST_KEY", None)
            from config.env_loader import load_env
            load_env(path=env_file)
            assert os.environ["TEST_KEY"] == "hello"

    def test_does_not_override_existing_env_var(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MY_VAR=from_file\n")
        with patch.dict(os.environ, {"MY_VAR": "from_shell"}):
            from config.env_loader import load_env
            load_env(path=env_file, override=False)
            assert os.environ["MY_VAR"] == "from_shell"

    def test_override_true_replaces_existing(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MY_VAR=from_file\n")
        with patch.dict(os.environ, {"MY_VAR": "from_shell"}):
            from config.env_loader import load_env
            load_env(path=env_file, override=True)
            assert os.environ["MY_VAR"] == "from_file"

    def test_skips_comments_and_blank_lines(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\n\nVALID=yes\n")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VALID", None)
            from config.env_loader import load_env
            load_env(path=env_file)
            assert os.environ.get("VALID") == "yes"

    def test_strips_double_quotes(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('QUOTED="hello world"\n')
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("QUOTED", None)
            from config.env_loader import load_env
            load_env(path=env_file)
            assert os.environ["QUOTED"] == "hello world"

    def test_strips_single_quotes(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("SINGLE='value'\n")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SINGLE", None)
            from config.env_loader import load_env
            load_env(path=env_file)
            assert os.environ["SINGLE"] == "value"

    def test_variable_interpolation(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("BASE=postgres\nURL=${BASE}://localhost\n")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BASE", None)
            os.environ.pop("URL", None)
            from config.env_loader import load_env
            load_env(path=env_file)
            assert os.environ["URL"] == "postgres://localhost"

    def test_returns_none_when_no_env_file(self, tmp_path):
        from config.env_loader import load_env
        result = load_env(path=tmp_path / "does_not_exist.env")
        assert result is None

    def test_returns_path_when_loaded(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("X=1\n")
        from config.env_loader import load_env
        result = load_env(path=env_file)
        assert result == env_file


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class TestSettings:

    def _fresh_settings(self, env: dict):
        """Create a Settings instance with a clean environment."""
        from config.settings import get_settings, Settings
        get_settings.cache_clear()
        with patch.dict(os.environ, env, clear=False):
            s = Settings()
        return s

    def test_reads_db_url_from_env(self):
        s = self._fresh_settings({"RECSYS_DB_URL": "postgresql://custom/db"})
        assert s.db_url == "postgresql://custom/db"

    def test_reads_mlflow_uri_from_env(self):
        s = self._fresh_settings({"MLFLOW_TRACKING_URI": "http://mymlflow:5000"})
        assert s.mlflow_uri == "http://mymlflow:5000"

    def test_reads_data_dir_from_env(self):
        s = self._fresh_settings({"MOVIELENS_DATA_DIR": "/custom/data"})
        assert s.data_dir == "/custom/data"

    def test_reads_minio_endpoint_from_env(self):
        s = self._fresh_settings({"MINIO_ENDPOINT": "minio.company.com:9000"})
        assert s.minio_endpoint == "minio.company.com:9000"

    def test_default_env_is_development(self):
        s = self._fresh_settings({})
        # Only check default if not overridden in current shell
        if "RECSYS_ENV" not in os.environ:
            assert s.env == "development"

    def test_is_test_returns_true_for_test_env(self):
        s = self._fresh_settings({"RECSYS_ENV": "test"})
        assert s.is_test() is True

    def test_is_test_returns_false_for_development(self):
        s = self._fresh_settings({"RECSYS_ENV": "development"})
        assert s.is_test() is False

    def test_effective_db_url_returns_test_url_in_test_env(self):
        s = self._fresh_settings({
            "RECSYS_ENV": "test",
            "RECSYS_DB_URL_TEST": "postgresql://test/db",
        })
        assert s.effective_db_url() == "postgresql://test/db"

    def test_effective_db_url_returns_main_url_in_development(self):
        s = self._fresh_settings({
            "RECSYS_ENV": "development",
            "RECSYS_DB_URL": "postgresql://prod/db",
        })
        assert s.effective_db_url() == "postgresql://prod/db"

    def test_apply_to_environment_sets_mlflow_tracking_uri(self):
        s = self._fresh_settings({"MLFLOW_TRACKING_URI": "http://test:5000"})
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MLFLOW_TRACKING_URI", None)
            s.apply_to_environment()
            assert os.environ.get("MLFLOW_TRACKING_URI") == "http://test:5000"

    def test_apply_to_environment_sets_aws_keys(self):
        # Clear both the AWS and MINIO key env vars so Settings reads defaults,
        # then override the Settings fields directly to test apply_to_environment.
        clean = {k: v for k, v in os.environ.items()
                 if k not in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                              "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY")}
        with patch.dict(os.environ, clean, clear=True):
            from config.settings import get_settings, Settings
            get_settings.cache_clear()
            s = Settings()
            # Directly set the fields we want to push, bypassing env-var init
            s.aws_access_key_id     = "mykey"
            s.aws_secret_access_key = "mysecret"
            s.apply_to_environment()
            assert os.environ.get("AWS_ACCESS_KEY_ID") == "mykey"
            assert os.environ.get("AWS_SECRET_ACCESS_KEY") == "mysecret"

    def test_summary_contains_key_fields(self):
        s = self._fresh_settings({"RECSYS_ENV": "test"})
        summary = s.summary()
        assert "db_url" in summary
        assert "mlflow_uri" in summary
        assert "minio_endpoint" in summary


# ---------------------------------------------------------------------------
# bootstrap_environment (main.py)
# ---------------------------------------------------------------------------

class TestBootstrapEnvironment:

    def _args(self, **kwargs):
        from main import build_parser
        args = build_parser().parse_args([])
        for k, v in kwargs.items():
            setattr(args, k, v)
        return args

    def test_cli_db_url_override_wins(self):
        args = self._args(db_url="postgresql://cli/db", data_dir=None,
                          mlflow_uri=None, minio_endpoint=None, env_file=None)
        with patch("config.env_loader.load_env", return_value=None), \
             patch("config.settings.get_settings") as mock_gs:
            mock_gs.cache_clear = lambda: None
            from main import bootstrap_environment
            bootstrap_environment(args)
        assert os.environ.get("RECSYS_DB_URL") == "postgresql://cli/db"

    def test_data_dir_override(self):
        args = self._args(data_dir="/cli/data", db_url=None,
                          mlflow_uri=None, minio_endpoint=None, env_file=None)
        with patch("config.env_loader.load_env", return_value=None), \
             patch("config.settings.get_settings") as mock_gs:
            mock_gs.cache_clear = lambda: None
            from main import bootstrap_environment
            bootstrap_environment(args)
        assert os.environ.get("MOVIELENS_DATA_DIR") == "/cli/data"

    def test_none_overrides_not_applied(self):
        original = os.environ.get("RECSYS_DB_URL", "original")
        args = self._args(db_url=None, data_dir=None,
                          mlflow_uri=None, minio_endpoint=None, env_file=None)
        with patch("config.env_loader.load_env", return_value=None), \
             patch("config.settings.get_settings") as mock_gs:
            mock_gs.cache_clear = lambda: None
            from main import bootstrap_environment
            bootstrap_environment(args)
        # RECSYS_DB_URL should not have been changed by a None override
        assert os.environ.get("RECSYS_DB_URL") == original
