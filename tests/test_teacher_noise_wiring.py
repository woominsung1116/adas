"""Tests for Phase 6 slice 13: teacher perception noise plumbing
through calibration + helper entry points, with matching
observability in reports and checkpoints.

Covers:
  - DefaultEvaluator.teacher_noise_config propagates into orch
  - build_default_evaluator forwards teacher_noise_config
  - build_default_autoresearch_setup forwards teacher_noise_config
  - run_prior_predictive_check forwards and stores the config
  - validation helpers (_run_scenario_once /
    evaluate_config_on_scenarios / build_held_out_report) accept
    and forward it
  - DefaultAutoresearchSetup.validate_best_on_heldout inherits
    the evaluator's teacher_noise_config by default and lets
    explicit override win
  - summary/report surfaces expose the config with compact
    "disabled" / "dropout=X.XXX,confusion=Y.YYY" formatting
  - checkpoint JSON persists the config
  - default behavior preserved when omitted
  - fixed-seed determinism
"""

import json

import pytest

from src.simulation.teacher_noise import TeacherNoiseConfig
from src.simulation.teacher_memory import RetrievalNoiseConfig
from src.calibration import (
    build_default_autoresearch_setup,
    run_prior_predictive_check,
    ValidationScenario,
    evaluate_config_on_scenarios,
    build_held_out_report,
)
from src.calibration.applier import build_default_evaluator
from src.calibration.validation import _run_scenario_once
from src.simulation.orchestrator_v2 import OrchestratorV2


# ---------------------------------------------------------------------------
# Evaluator + setup construction
# ---------------------------------------------------------------------------


def test_build_default_evaluator_forwards_teacher_noise_config():
    cfg = TeacherNoiseConfig(
        observation_dropout_prob=0.3,
        observation_confusion_prob=0.1,
    )
    ev = build_default_evaluator(
        n_classes=1, max_turns=10, seed=1,
        teacher_noise_config=cfg,
    )
    assert ev.teacher_noise_config is cfg


def test_build_default_evaluator_default_teacher_noise_is_none():
    ev = build_default_evaluator(n_classes=1, max_turns=10, seed=1)
    assert ev.teacher_noise_config is None


def test_build_default_autoresearch_setup_forwards_teacher_noise_config(tmp_path):
    cfg = TeacherNoiseConfig(observation_dropout_prob=0.2)
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
        teacher_noise_config=cfg,
    )
    assert setup.evaluator.teacher_noise_config is cfg


def test_default_setup_default_teacher_noise_is_none(tmp_path):
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
    )
    assert setup.evaluator.teacher_noise_config is None


# ---------------------------------------------------------------------------
# DefaultEvaluator.evaluate propagates the config into OrchestratorV2
# ---------------------------------------------------------------------------


def test_default_evaluator_evaluate_runs_with_teacher_noise_config():
    cfg = TeacherNoiseConfig(
        observation_dropout_prob=0.25,
        observation_confusion_prob=0.1,
    )
    ev = build_default_evaluator(
        n_classes=1, max_turns=15, seed=7,
        teacher_noise_config=cfg,
    )
    bundle, loss = ev.evaluate(config={})
    assert loss.total != float("inf")
    assert len(bundle.histories) == 1


# ---------------------------------------------------------------------------
# Prior predictive
# ---------------------------------------------------------------------------


def test_run_prior_predictive_check_accepts_teacher_noise_config():
    cfg = TeacherNoiseConfig(
        observation_dropout_prob=0.15,
        observation_confusion_prob=0.05,
    )
    report = run_prior_predictive_check(
        n_classes=1, max_turns=10, seed=1,
        teacher_noise_config=cfg,
    )
    assert report.teacher_noise_config is cfg
    assert "Teacher noise: dropout=0.150,confusion=0.050" in report.summary()


def test_run_prior_predictive_check_default_teacher_noise_disabled():
    report = run_prior_predictive_check(n_classes=1, max_turns=10, seed=1)
    assert report.teacher_noise_config is None
    assert "Teacher noise: disabled" in report.summary()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def setup_bundle(tmp_path_factory):
    return build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=10,
        enforce_constraints=False,
        results_dir=tmp_path_factory.mktemp("hb"),
    )


def _scenario(name: str, archetype: str, seed: int) -> ValidationScenario:
    return ValidationScenario(
        name=name, archetype=archetype, seed=seed,
        max_turns=10, n_students=5,
    )


def test_run_scenario_once_accepts_teacher_noise_config(setup_bundle):
    cfg = TeacherNoiseConfig(observation_dropout_prob=0.2)
    result = _run_scenario_once(
        scenario=_scenario("t1", "quiet_structured", 1),
        config={},
        naturalness_targets=setup_bundle.evaluator.naturalness_targets,
        epidemiology_targets=setup_bundle.evaluator.epidemiology_targets,
        teacher_noise_config=cfg,
    )
    assert result.error is None


def test_evaluate_config_on_scenarios_forwards_teacher_noise(setup_bundle):
    cfg = TeacherNoiseConfig(observation_dropout_prob=0.2)
    results = evaluate_config_on_scenarios(
        config={},
        scenarios=[_scenario("t1", "quiet_structured", 1)],
        naturalness_targets=setup_bundle.evaluator.naturalness_targets,
        epidemiology_targets=setup_bundle.evaluator.epidemiology_targets,
        teacher_noise_config=cfg,
    )
    assert len(results) == 1
    assert results[0].error is None


def test_build_held_out_report_forwards_teacher_noise(setup_bundle):
    cfg = TeacherNoiseConfig(
        observation_dropout_prob=0.2,
        observation_confusion_prob=0.05,
    )
    training = [_scenario("t1", "quiet_structured", 1)]
    heldout = [_scenario("h1", "chaotic", 1)]
    report = build_held_out_report(
        config={},
        training_scenarios=training,
        heldout_scenarios=heldout,
        naturalness_targets=setup_bundle.evaluator.naturalness_targets,
        epidemiology_targets=setup_bundle.evaluator.epidemiology_targets,
        teacher_noise_config=cfg,
    )
    assert report.teacher_noise_config is cfg
    assert "Teacher noise: dropout=0.200,confusion=0.050" in report.summary()


# ---------------------------------------------------------------------------
# validate_best_on_heldout inheritance
# ---------------------------------------------------------------------------


def test_validate_best_on_heldout_inherits_teacher_noise(tmp_path):
    cfg = TeacherNoiseConfig(observation_dropout_prob=0.3)
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
        teacher_noise_config=cfg,
    )
    heldout = [_scenario("h1", "chaotic", 1)]
    report = setup.validate_best_on_heldout(
        best_config={}, scenarios=heldout,
    )
    assert report.teacher_noise_config is cfg


def test_validate_best_on_heldout_explicit_teacher_override_wins(tmp_path):
    ev_cfg = TeacherNoiseConfig(observation_dropout_prob=0.1)
    override_cfg = TeacherNoiseConfig(observation_dropout_prob=0.5)
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
        teacher_noise_config=ev_cfg,
    )
    heldout = [_scenario("h1", "chaotic", 1)]
    report = setup.validate_best_on_heldout(
        best_config={}, scenarios=heldout,
        teacher_noise_config=override_cfg,
    )
    assert report.teacher_noise_config is override_cfg
    assert "dropout=0.500" in report.summary()


# ---------------------------------------------------------------------------
# Summary / report surface
# ---------------------------------------------------------------------------


def test_setup_summary_shows_teacher_noise_disabled_by_default(tmp_path):
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
    )
    s = setup.summary()
    assert "teacher_noise=disabled" in s


def test_setup_summary_shows_teacher_noise_when_configured(tmp_path):
    cfg = TeacherNoiseConfig(
        observation_dropout_prob=0.3,
        observation_confusion_prob=0.1,
    )
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
        teacher_noise_config=cfg,
    )
    s = setup.summary()
    assert "teacher_noise=dropout=0.300,confusion=0.100" in s


def test_setup_report_shows_both_noise_policies(tmp_path):
    tn = TeacherNoiseConfig(observation_dropout_prob=0.2)
    rn = RetrievalNoiseConfig(dropout_prob=0.15, similarity_jitter=0.05)
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
        teacher_noise_config=tn,
        retrieval_noise_config=rn,
    )
    r = setup.report()
    assert "retrieval_noise_config:" in r
    assert "teacher_noise_config:" in r
    assert "dropout=0.200,confusion=0.000" in r
    assert "dropout=0.150,jitter=0.050" in r


# ---------------------------------------------------------------------------
# Checkpoint persistence
# ---------------------------------------------------------------------------


def test_checkpoint_persists_teacher_noise_config(tmp_path):
    cfg = TeacherNoiseConfig(
        observation_dropout_prob=0.25,
        observation_confusion_prob=0.1,
    )
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
        teacher_noise_config=cfg,
    )
    setup.orchestrator.run()
    payload = json.loads(setup.orchestrator.checkpoint_path.read_text())
    assert "teacher_noise_config" in payload
    assert payload["teacher_noise_config"] == {
        "observation_dropout_prob": 0.25,
        "observation_confusion_prob": 0.1,
    }


def test_checkpoint_teacher_noise_config_is_null_when_not_configured(tmp_path):
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
    )
    setup.orchestrator.run()
    payload = json.loads(setup.orchestrator.checkpoint_path.read_text())
    assert "teacher_noise_config" in payload
    assert payload["teacher_noise_config"] is None


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_teacher_noise_plumbing_is_deterministic_under_fixed_seed():
    def fingerprint():
        cfg = TeacherNoiseConfig(
            observation_dropout_prob=0.2,
            observation_confusion_prob=0.05,
        )
        orch = OrchestratorV2(
            n_students=5, max_classes=1, seed=42,
            teacher_noise_config=cfg,
        )
        orch.classroom.MAX_TURNS = 40
        orch.run_class()
        return [
            (r.student_id, r.turn, r.outcome, tuple(r.observed_behaviors))
            for r in orch.memory.case_base._records
        ]

    a = fingerprint()
    b = fingerprint()
    assert a == b
    assert a
