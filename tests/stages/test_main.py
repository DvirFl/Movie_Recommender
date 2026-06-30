"""Tests: main.py — CLI parsing, stage resolution, dispatch, error handling."""
import pytest
from unittest.mock import MagicMock, patch, call

import main as main_module
from main import (
    build_parser, resolve_stages, run_stage,
    ALL_STAGES, main,
)


# ---------------------------------------------------------------------------
# resolve_stages
# ---------------------------------------------------------------------------

class TestResolveStages:

    def _args(self, **kwargs):
        parser = build_parser()
        defaults = parser.parse_args([])
        for k, v in kwargs.items():
            setattr(defaults, k, v)
        return defaults

    def test_default_returns_all_stages(self):
        args = self._args(stages=None, from_stage=None, to_stage=None,
                          skip_tune=False, skip_startup=False)
        assert resolve_stages(args) == ALL_STAGES

    def test_explicit_stages_gets_startup_prepended(self):
        # startup is auto-prepended when not explicitly skipped
        args = self._args(stages=["ingest", "validate"], from_stage=None, to_stage=None,
                          skip_tune=False, skip_startup=False)
        stages = resolve_stages(args)
        assert stages[0] == "startup"
        assert "ingest" in stages
        assert "validate" in stages

    def test_skip_startup_prevents_prepend(self):
        args = self._args(stages=["ingest", "validate"], from_stage=None, to_stage=None,
                          skip_tune=False, skip_startup=True)
        stages = resolve_stages(args)
        assert "startup" not in stages

    def test_from_stage(self):
        args = self._args(stages=None, from_stage="featurize", to_stage=None,
                          skip_tune=False, skip_startup=False)
        stages = resolve_stages(args)
        # startup is prepended
        assert "startup" in stages
        assert "featurize" in stages
        assert "ingest" not in stages

    def test_to_stage_inclusive(self):
        args = self._args(stages=None, from_stage=None, to_stage="split",
                          skip_tune=False, skip_startup=False)
        stages = resolve_stages(args)
        assert stages[-1] == "split"
        assert "tune" not in stages

    def test_from_and_to_stage(self):
        args = self._args(stages=None, from_stage="featurize", to_stage="split",
                          skip_tune=False, skip_startup=False)
        stages = resolve_stages(args)
        assert "featurize" in stages
        assert "split" in stages
        assert "tune" not in stages

    def test_skip_tune_removes_tune(self):
        args = self._args(stages=None, from_stage=None, to_stage=None,
                          skip_tune=True, skip_startup=False)
        stages = resolve_stages(args)
        assert "tune" not in stages
        assert "train" in stages

    def test_skip_tune_with_explicit_stages(self):
        args = self._args(stages=["tune", "train"], from_stage=None, to_stage=None,
                          skip_tune=True, skip_startup=True)
        stages = resolve_stages(args)
        assert "tune" not in stages
        assert "train" in stages

    def test_explicit_stages_with_startup_no_duplicate(self):
        # If startup is already in the explicit list, it should not be duplicated
        args = self._args(stages=["startup", "ingest"], from_stage=None, to_stage=None,
                          skip_tune=False, skip_startup=False)
        stages = resolve_stages(args)
        assert stages.count("startup") == 1


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------

class TestBuildParser:

    def test_default_stages_is_none(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.stages is None

    def test_stages_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--stages", "ingest", "validate"])
        assert args.stages == ["ingest", "validate"]

    def test_losses_default_all(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.losses == ["all"]

    def test_losses_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--losses", "TimedecayMSELoss"])
        assert args.losses == ["TimedecayMSELoss"]

    def test_architectures_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--architectures", "TwoTower"])
        assert args.architectures == ["TwoTower"]

    def test_no_minio_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--no-minio"])
        assert args.no_minio is True

    def test_skip_tune_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--skip-tune"])
        assert args.skip_tune is True

    def test_dry_run_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_port_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--port", "9000"])
        assert args.port == 9000

    def test_reload_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--reload"])
        assert args.reload is True

    def test_decay_lambda_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--decay-lambda", "0.5"])
        assert args.decay_lambda == 0.5

    def test_train_frac_val_frac(self):
        parser = build_parser()
        args = parser.parse_args(["--train-frac", "0.7", "--val-frac", "0.15"])
        assert args.train_frac == 0.7
        assert args.val_frac   == 0.15

    def test_n_trials_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--n-trials", "10"])
        assert args.n_trials == 10

    def test_top_n_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--top-n", "50"])
        assert args.top_n == 50


    def test_ingest_mode_default_upsert(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.ingest_mode == "upsert"

    def test_ingest_mode_skip(self):
        parser = build_parser()
        args = parser.parse_args(["--ingest-mode", "skip"])
        assert args.ingest_mode == "skip"

    def test_ingest_mode_replace(self):
        parser = build_parser()
        args = parser.parse_args(["--ingest-mode", "replace"])
        assert args.ingest_mode == "replace"

    def test_ingest_mode_invalid_rejected(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--ingest-mode", "invalid"])

    def test_invalid_stage_raises(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--stages", "not_a_real_stage"])


# ---------------------------------------------------------------------------
# run_stage dispatch
# ---------------------------------------------------------------------------

class TestRunStageDispatch:

    def _args(self, **overrides):
        parser = build_parser()
        args = parser.parse_args([])
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    def test_dispatches_startup(self):
        with patch("stages.stage_startup.run", return_value=MagicMock(
            all_ok=True, schemas_created=[], buckets_created=[], errors=[]
        )) as mock_run:
            run_stage("startup", self._args(), {})
            mock_run.assert_called_once()

    def test_dispatches_ingest(self):
        with patch("stages.stage_ingest.run", return_value=MagicMock()) as mock_run:
            run_stage("ingest", self._args(), {})
            mock_run.assert_called_once()

    def test_dispatches_validate(self):
        with patch("stages.stage_validate.run", return_value=MagicMock()) as mock_run:
            run_stage("validate", self._args(), {})
            mock_run.assert_called_once()

    def test_dispatches_featurize(self):
        with patch("stages.stage_featurize.run", return_value=MagicMock()) as mock_run:
            run_stage("featurize", self._args(), {})
            mock_run.assert_called_once()

    def test_dispatches_split(self):
        with patch("stages.stage_split.run", return_value=MagicMock()) as mock_run:
            run_stage("split", self._args(), {})
            mock_run.assert_called_once()

    def test_dispatches_tune(self):
        with patch("stages.stage_tune.run", return_value=MagicMock(best_hparams={})) as mock_run:
            run_stage("tune", self._args(), {})
            mock_run.assert_called_once()

    def test_dispatches_train(self):
        with patch("stages.stage_train.run", return_value=MagicMock()) as mock_run:
            run_stage("train", self._args(), {})
            mock_run.assert_called_once()

    def test_dispatches_cross_distill(self):
        with patch("stages.stage_cross_distill.run", return_value=MagicMock()) as mock_run:
            run_stage("cross_distill", self._args(), {})
            mock_run.assert_called_once()

    def test_dispatches_evaluate(self):
        with patch("stages.stage_evaluate.run", return_value=MagicMock()) as mock_run:
            run_stage("evaluate", self._args(), {})
            mock_run.assert_called_once()

    def test_dispatches_precompute(self):
        with patch("stages.stage_precompute.run", return_value=MagicMock()) as mock_run:
            run_stage("precompute", self._args(), {})
            mock_run.assert_called_once()

    def test_train_reads_tune_hparams_from_state(self):
        """train stage pulls best_hparams from state['tune']."""
        tune_result = MagicMock()
        tune_result.best_hparams = {"TwoTower_TimedecayMSELoss": {"lr": 1e-4}}
        state = {"tune": tune_result}
        with patch("stages.stage_train.run", return_value=MagicMock()) as mock_run:
            run_stage("train", self._args(), state)
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["best_hparams"] == tune_result.best_hparams

    def test_no_minio_forwarded_to_train(self):
        args = self._args(no_minio=True)
        with patch("stages.stage_train.run", return_value=MagicMock()) as mock_run:
            run_stage("train", args, {})
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["save_to_minio"] is False

    def test_validate_strict_forwarded(self):
        args = self._args(no_strict_validate=True)
        with patch("stages.stage_validate.run", return_value=MagicMock()) as mock_run:
            run_stage("validate", args, {})
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["strict"] is False

    def test_decay_lambda_forwarded_to_featurize(self):
        args = self._args(decay_lambda=0.5)
        with patch("stages.stage_featurize.run", return_value=MagicMock()) as mock_run:
            run_stage("featurize", args, {})
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["decay_lambda"] == 0.5

    def test_top_n_forwarded_to_precompute(self):
        args = self._args(top_n=50)
        with patch("stages.stage_precompute.run", return_value=MagicMock()) as mock_run:
            run_stage("precompute", args, {})
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["top_n"] == 50


# ---------------------------------------------------------------------------
# main() end-to-end
# ---------------------------------------------------------------------------

class TestMain:

    def _mock_startup(self):
        return patch("stages.stage_startup.run", return_value=MagicMock(
            all_ok=True, schemas_created=["raw"], buckets_created=[], errors=[]
        ))

    def _mock_bootstrap(self):
        return patch("main.bootstrap_environment")

    def test_dry_run_returns_zero(self):
        with self._mock_bootstrap():
            assert main(["--dry-run"]) == 0

    def test_no_stages_returns_error(self):
        with pytest.raises(SystemExit):
            main(["--stages"])

    def test_single_stage_ingest(self):
        with self._mock_bootstrap(), \
             self._mock_startup(), \
             patch("stages.stage_ingest.run",
                   return_value=MagicMock(counts={"r": 10}, data_dir="/d")):
            rc = main(["--stages", "startup", "ingest", "--data-dir", "/d"])
        assert rc == 0

    def test_etl_stages_run_in_order(self):
        call_order = []
        def make_mock(name, retval):
            def mock(**kwargs):
                call_order.append(name)
                return retval
            return mock

        patches = {
            "startup":   patch("stages.stage_startup.run",   side_effect=make_mock("startup",   MagicMock(all_ok=True, schemas_created=[], buckets_created=[], errors=[]))),
            "ingest":    patch("stages.stage_ingest.run",    side_effect=make_mock("ingest",    MagicMock(counts={}, data_dir=""))),
            "validate":  patch("stages.stage_validate.run",  side_effect=make_mock("validate",  MagicMock(passed=True, issues=[], stats={}))),
            "featurize": patch("stages.stage_featurize.run", side_effect=make_mock("featurize", MagicMock(n_users=5, n_items=10))),
            "split":     patch("stages.stage_split.run",     side_effect=make_mock("split",     MagicMock(train=8, val=1, test=1))),
        }
        with self._mock_bootstrap(), patches["startup"], patches["ingest"], \
             patches["validate"], patches["featurize"], patches["split"]:
            main(["--stages", "startup", "ingest", "validate", "featurize", "split"])
        assert call_order == ["startup", "ingest", "validate", "featurize", "split"]

    def test_failed_stage_returns_nonzero(self):
        with self._mock_bootstrap(), self._mock_startup(), \
             patch("stages.stage_ingest.run", side_effect=RuntimeError("disk full")):
            rc = main(["--stages", "startup", "ingest"])
        assert rc == 1

    def test_failed_stage_does_not_run_subsequent(self):
        validate_called = {"n": 0}
        def fake_validate(**kwargs):
            validate_called["n"] += 1
            return MagicMock(passed=True, issues=[], stats={})

        with self._mock_bootstrap(), self._mock_startup(), \
             patch("stages.stage_ingest.run", side_effect=RuntimeError("fail")), \
             patch("stages.stage_validate.run", side_effect=fake_validate):
            main(["--stages", "startup", "ingest", "validate"])
        assert validate_called["n"] == 0

    def test_skip_tune_omits_tune_stage(self):
        tune_called  = {"n": 0}
        train_called = {"n": 0}

        def fake_tune(**kwargs):  tune_called["n"]  += 1; return MagicMock(best_hparams={})
        def fake_train(**kwargs): train_called["n"] += 1; return MagicMock(run_ids={})

        with self._mock_bootstrap(), self._mock_startup(), \
             patch("stages.stage_tune.run",  side_effect=fake_tune), \
             patch("stages.stage_train.run", side_effect=fake_train):
            main(["--stages", "startup", "tune", "train", "--skip-tune"])

        assert tune_called["n"]  == 0
        assert train_called["n"] == 1

    def test_losses_filter_passed_to_train(self):
        with self._mock_bootstrap(), self._mock_startup(), \
             patch("stages.stage_train.run", return_value=MagicMock(run_ids={})) as mock_train:
            main(["--stages", "startup", "train",
                  "--losses", "TimedecayMSELoss", "--skip-tune"])
        call_kwargs = mock_train.call_args[1]
        assert call_kwargs["losses"] == ["TimedecayMSELoss"]

    def test_arch_filter_passed_to_train(self):
        with self._mock_bootstrap(), self._mock_startup(), \
             patch("stages.stage_train.run", return_value=MagicMock(run_ids={})) as mock_train:
            main(["--stages", "startup", "train",
                  "--architectures", "TwoTower", "--skip-tune"])
        call_kwargs = mock_train.call_args[1]
        assert call_kwargs["architectures"] == ["TwoTower"]
