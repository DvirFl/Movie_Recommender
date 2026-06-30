"""Tests: stage_startup — DB checks, schema creation, MinIO, MLflow."""
import pytest
from unittest.mock import MagicMock, patch, call
import sys
from stages.stage_startup import run, StartupResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mock_settings_obj():
    m = MagicMock()
    m.summary.return_value = ""
    m.apply_to_environment.return_value = None
    m.effective_db_url.return_value = "postgresql://test/db"
    return m


def _mock_engine():
    conn = MagicMock()
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)
    eng = MagicMock()
    eng.connect.return_value = ctx
    return eng


def _migration_ok(result):
    result.migrations_ok = True


def _minio_ok(result):
    result.minio_ok = True


def _mlflow_ok(result):
    result.mlflow_ok = True


# ---------------------------------------------------------------------------
# PostgreSQL checks
# ---------------------------------------------------------------------------

def test_db_ok_set_on_success():
    with patch("db.connection.check_connection", return_value=True), \
         patch("db.connection.get_engine", return_value=_mock_engine()), \
         patch("stages.stage_startup._create_tables", side_effect=_migration_ok), \
         patch("stages.stage_startup._check_minio"), \
         patch("stages.stage_startup._ensure_mlflow"), \
         patch("config.settings.settings", _mock_settings_obj()):
        result = run(skip_minio=True, skip_mlflow=True)
    assert result.db_ok is True


def test_raises_when_postgres_unreachable_strict():
    with patch("db.connection.check_connection", return_value=False), \
         patch("config.settings.settings", _mock_settings_obj()):
        with pytest.raises(RuntimeError, match="PostgreSQL"):
            run(strict=True)


def test_no_raise_when_postgres_unreachable_non_strict():
    with patch("db.connection.check_connection", return_value=False), \
         patch("config.settings.settings", _mock_settings_obj()):
        result = run(strict=False)
    assert result.db_ok is False
    assert result.all_ok is False


def test_schemas_created():
    mock_conn = MagicMock()
    with patch("db.connection.check_connection", return_value=True), \
         patch("db.connection.get_engine", return_value=_mock_engine()), \
         patch("stages.stage_startup._create_tables", side_effect=_migration_ok), \
         patch("stages.stage_startup._check_minio"), \
         patch("stages.stage_startup._ensure_mlflow"), \
         patch("config.settings.settings", _mock_settings_obj()):
        result = run(skip_minio=True, skip_mlflow=True)
    assert set(result.schemas_created) == {"raw", "features", "serving", "pipeline"}


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------

def test_migrations_skipped():
    with patch("db.connection.check_connection", return_value=True), \
         patch("db.connection.get_engine", return_value=_mock_engine()), \
         patch("stages.stage_startup._create_tables") as mock_migrate, \
         patch("stages.stage_startup._check_minio"), \
         patch("stages.stage_startup._ensure_mlflow"), \
         patch("config.settings.settings", _mock_settings_obj()):
        result = run(skip_minio=True, skip_mlflow=True, skip_migrations=True)
    mock_migrate.assert_not_called()
    assert result.migrations_ok is True


# ---------------------------------------------------------------------------
# MinIO
# ---------------------------------------------------------------------------

def test_minio_skipped_when_no_minio():
    with patch("db.connection.check_connection", return_value=True), \
         patch("db.connection.get_engine", return_value=_mock_engine()), \
         patch("stages.stage_startup._create_tables", side_effect=_migration_ok), \
         patch("stages.stage_startup._check_minio") as mock_minio, \
         patch("stages.stage_startup._ensure_mlflow"), \
         patch("config.settings.settings", _mock_settings_obj()):
        result = run(skip_minio=True, skip_mlflow=True)
    mock_minio.assert_not_called()
    assert result.minio_ok is True


def test_minio_check_called_when_not_skipped():
    with patch("db.connection.check_connection", return_value=True), \
         patch("db.connection.get_engine", return_value=_mock_engine()), \
         patch("stages.stage_startup._create_tables", side_effect=_migration_ok), \
         patch("stages.stage_startup._check_minio", side_effect=_minio_ok) as mock_minio, \
         patch("stages.stage_startup._ensure_mlflow", side_effect=_mlflow_ok), \
         patch("config.settings.settings", _mock_settings_obj()):
        result = run(skip_minio=False, skip_mlflow=False)
    mock_minio.assert_called_once()


# ---------------------------------------------------------------------------
# all_ok flag
# ---------------------------------------------------------------------------

def test_all_ok_when_everything_passes():
    with patch("db.connection.check_connection", return_value=True), \
         patch("db.connection.get_engine", return_value=_mock_engine()), \
         patch("stages.stage_startup._create_tables", side_effect=_migration_ok), \
         patch("stages.stage_startup._check_minio",   side_effect=_minio_ok), \
         patch("stages.stage_startup._ensure_mlflow",  side_effect=_mlflow_ok), \
         patch("config.settings.settings", _mock_settings_obj()):
        result = run(skip_minio=False, skip_mlflow=False)
    assert result.all_ok is True


def test_all_ok_false_when_minio_fails():
    def minio_fail(result):
        result.errors.append("minio down")
        # minio_ok stays False

    with patch("db.connection.check_connection", return_value=True), \
         patch("db.connection.get_engine", return_value=_mock_engine()), \
         patch("stages.stage_startup._create_tables", side_effect=_migration_ok), \
         patch("stages.stage_startup._check_minio",   side_effect=minio_fail), \
         patch("stages.stage_startup._ensure_mlflow",  side_effect=_mlflow_ok), \
         patch("config.settings.settings", _mock_settings_obj()):
        result = run(skip_minio=False, skip_mlflow=False, strict=False)
    assert result.all_ok is False
    assert any("minio" in e for e in result.errors)


def test_errors_list_empty_on_full_success():
    with patch("db.connection.check_connection", return_value=True), \
         patch("db.connection.get_engine", return_value=_mock_engine()), \
         patch("stages.stage_startup._create_tables", side_effect=_migration_ok), \
         patch("stages.stage_startup._check_minio",   side_effect=_minio_ok), \
         patch("stages.stage_startup._ensure_mlflow",  side_effect=_mlflow_ok), \
         patch("config.settings.settings", _mock_settings_obj()):
        result = run()
    assert result.errors == []


# ---------------------------------------------------------------------------
# _mlflow_is_healthy
# ---------------------------------------------------------------------------

class TestMlflowIsHealthy:

    def test_returns_true_on_200(self):
        from stages.stage_startup import _mlflow_is_healthy
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__  = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert _mlflow_is_healthy("http://localhost:5000") is True

    def test_returns_false_on_connection_error(self):
        from stages.stage_startup import _mlflow_is_healthy
        import urllib.error
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("connection refused")):
            assert _mlflow_is_healthy("http://localhost:5000") is False

    def test_returns_false_on_generic_exception(self):
        from stages.stage_startup import _mlflow_is_healthy
        with patch("urllib.request.urlopen", side_effect=Exception("boom")):
            assert _mlflow_is_healthy("http://localhost:5000") is False


# ---------------------------------------------------------------------------
# _ensure_mlflow
# ---------------------------------------------------------------------------

class TestEnsureMlflow:

    def _settings(self):
        s = _mock_settings_obj()
        s.mlflow_uri = "http://localhost:5000"
        s.mlflow_port = 5000
        s.mlflow_backend_store_uri = "postgresql://postgres:postgres@localhost/mlflow"
        s.mlflow_artifact_root = "s3://model-checkpoints"
        s.mlflow_s3_endpoint_url = "http://localhost:9000"
        s.aws_access_key_id = "minioadmin"
        s.aws_secret_access_key = "minioadmin"
        return s

    def test_skips_launch_when_already_healthy(self):
        from stages.stage_startup import _ensure_mlflow
        result = StartupResult()
        with patch("stages.stage_startup._mlflow_is_healthy", return_value=True), \
             patch("stages.stage_startup._launch_mlflow") as mock_launch, \
             patch("config.settings.settings", self._settings()):
            _ensure_mlflow(result)
        mock_launch.assert_not_called()
        assert result.mlflow_ok is True
        assert result.mlflow_launched is False

    def test_launches_when_not_running(self):
        from stages.stage_startup import _ensure_mlflow
        result = StartupResult()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345

        # _launch_mlflow sets result.mlflow_launched; replicate that in the mock
        def fake_launch(r, s):
            r.mlflow_launched = True
            r.mlflow_process  = mock_proc
            return mock_proc

        # Not healthy on first probe, healthy after launch
        with patch("stages.stage_startup._mlflow_is_healthy",
                   side_effect=[False, False, True]), \
             patch("stages.stage_startup._launch_mlflow",
                   side_effect=fake_launch) as mock_launch, \
             patch("stages.stage_startup.time.sleep"), \
             patch("config.settings.settings", self._settings()):
            _ensure_mlflow(result)
        mock_launch.assert_called_once()
        assert result.mlflow_ok is True
        assert result.mlflow_launched is True
        assert result.mlflow_process is mock_proc

    def test_records_error_when_process_exits_early(self):
        from stages.stage_startup import _ensure_mlflow
        result = StartupResult()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
        mock_proc.stdout.read.return_value = "port already in use"
        with patch("stages.stage_startup._mlflow_is_healthy", return_value=False), \
             patch("stages.stage_startup._launch_mlflow", return_value=mock_proc), \
             patch("stages.stage_startup.time.sleep"), \
             patch("config.settings.settings", self._settings()):
            _ensure_mlflow(result)
        assert result.mlflow_ok is False
        assert any("exited unexpectedly" in e for e in result.errors)

    def test_records_error_on_timeout(self):
        from stages.stage_startup import _ensure_mlflow, _MLFLOW_STARTUP_TIMEOUT
        result = StartupResult()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        start = [0.0]

        def fake_time():
            v = start[0]
            start[0] += _MLFLOW_STARTUP_TIMEOUT + 1
            return v

        with patch("stages.stage_startup._mlflow_is_healthy", return_value=False), \
             patch("stages.stage_startup._launch_mlflow", return_value=mock_proc), \
             patch("stages.stage_startup.time.sleep"), \
             patch("stages.stage_startup.time.time", side_effect=fake_time), \
             patch("config.settings.settings", self._settings()):
            _ensure_mlflow(result)
        assert result.mlflow_ok is False
        assert any("healthy" in e for e in result.errors)

    def test_launch_failure_recorded(self):
        from stages.stage_startup import _ensure_mlflow
        result = StartupResult()
        with patch("stages.stage_startup._mlflow_is_healthy", return_value=False), \
             patch("stages.stage_startup._launch_mlflow",
                   side_effect=FileNotFoundError("mlflow not found")), \
             patch("config.settings.settings", self._settings()):
            _ensure_mlflow(result)
        assert result.mlflow_ok is False
        assert any("launch" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# _launch_mlflow
# ---------------------------------------------------------------------------

class TestLaunchMlflow:

    def _settings(self):
        s = _mock_settings_obj()
        s.mlflow_port = 5000
        s.mlflow_backend_store_uri = "postgresql://localhost/mlflow"
        s.mlflow_artifact_root = "s3://model-checkpoints"
        s.mlflow_s3_endpoint_url = "http://localhost:9000"
        s.aws_access_key_id = "key"
        s.aws_secret_access_key = "secret"
        return s

    def test_uses_current_python_interpreter(self):
        from stages.stage_startup import _launch_mlflow
        result = StartupResult()
        with patch("subprocess.Popen", return_value=MagicMock()) as mock_popen:
            _launch_mlflow(result, self._settings())
        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == sys.executable

    def test_uses_mlflow_module_flag(self):
        from stages.stage_startup import _launch_mlflow
        result = StartupResult()
        with patch("subprocess.Popen", return_value=MagicMock()) as mock_popen:
            _launch_mlflow(result, self._settings())
        cmd = mock_popen.call_args[0][0]
        assert "-m" in cmd and "mlflow" in cmd

    def test_sets_mlflow_process_on_result(self):
        from stages.stage_startup import _launch_mlflow
        result = StartupResult()
        mock_proc = MagicMock()
        with patch("subprocess.Popen", return_value=mock_proc):
            returned = _launch_mlflow(result, self._settings())
        assert result.mlflow_process is mock_proc
        assert returned is mock_proc

    def test_mlflow_launched_flag_set(self):
        from stages.stage_startup import _launch_mlflow
        result = StartupResult()
        with patch("subprocess.Popen", return_value=MagicMock()):
            _launch_mlflow(result, self._settings())
        assert result.mlflow_launched is True

    def test_env_contains_s3_credentials(self):
        from stages.stage_startup import _launch_mlflow
        result = StartupResult()
        with patch("subprocess.Popen", return_value=MagicMock()) as mock_popen:
            _launch_mlflow(result, self._settings())
        env = mock_popen.call_args[1]["env"]
        assert env["MLFLOW_S3_ENDPOINT_URL"] == "http://localhost:9000"
        assert env["AWS_ACCESS_KEY_ID"]       == "key"
        assert env["AWS_SECRET_ACCESS_KEY"]   == "secret"

    def test_port_in_command(self):
        from stages.stage_startup import _launch_mlflow
        result = StartupResult()
        with patch("subprocess.Popen", return_value=MagicMock()) as mock_popen:
            _launch_mlflow(result, self._settings())
        cmd = mock_popen.call_args[0][0]
        assert "--port" in cmd and "5000" in cmd
