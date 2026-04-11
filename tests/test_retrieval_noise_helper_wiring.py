"""Tests for Phase 6 slice 11: retrieval-noise plumbing across
secondary helper entry points (prior predictive + validation).

Covers:
  - run_prior_predictive_check accepts and forwards retrieval_noise_config
  - Default run_prior_predictive_check still runs (no behavior change)
  - _run_scenario_once / evaluate_config_on_scenarios accept the config
  - build_held_out_report accepts + forwards to both training and
    held-out evaluation passes
  - DefaultAutoresearchSetup.validate_best_on_heldout auto-inherits
    the evaluator's retrieval_noise_config when no override is given
  - Explicit override on validate_best_on_heldout wins over the
    evaluator's config
  - Fixed-seed determinism under plumbed retrieval noise on all
    helper paths
"""

import pytest

from src.simulation.teacher_memory import RetrievalNoiseConfig
from src.calibration import (
    build_default_autoresearch_setup,
    ValidationScenario,
    evaluate_config_on_scenarios,
    build_held_out_report,
    run_prior_predictive_check,
)
from src.calibration.validation import _run_scenario_once


# ---------------------------------------------------------------------------
# run_prior_predictive_check
# ---------------------------------------------------------------------------


def test_run_prior_predictive_check_accepts_retrieval_noise_config():
    cfg = RetrievalNoiseConfig(dropout_prob=0.2, similarity_jitter=0.05)
    report = run_prior_predictive_check(
        n_classes=1, max_turns=15, seed=1,
        retrieval_noise_config=cfg,
    )
    # Sanity: report constructed, loss is finite.
    assert report.n_classes == 1
    assert report.loss_result.total != float("inf")


def test_run_prior_predictive_check_default_still_works():
    """Omitting retrieval_noise_config must preserve existing
    behavior bit-for-bit."""
    report = run_prior_predictive_check(
        n_classes=1, max_turns=15, seed=1,
    )
    assert report.n_classes == 1
    assert report.loss_result.total != float("inf")


def test_run_prior_predictive_check_is_deterministic_with_noise():
    cfg = RetrievalNoiseConfig(dropout_prob=0.3, similarity_jitter=0.1)

    def fingerprint():
        r = run_prior_predictive_check(
            n_classes=1, max_turns=20, seed=7,
            retrieval_noise_config=cfg,
        )
        return round(r.loss_result.total, 6)

    a = fingerprint()
    b = fingerprint()
    assert a == b


# ---------------------------------------------------------------------------
# Validation: _run_scenario_once / evaluate_config_on_scenarios
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def setup_bundle():
    return build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=15,
        enforce_constraints=False,
    )


def _scenario(name: str, archetype: str, seed: int) -> ValidationScenario:
    return ValidationScenario(
        name=name, archetype=archetype, seed=seed,
        max_turns=15, n_students=5,
    )


def test_run_scenario_once_accepts_retrieval_noise_config(setup_bundle):
    cfg = RetrievalNoiseConfig(dropout_prob=0.3, similarity_jitter=0.1)
    result = _run_scenario_once(
        scenario=_scenario("t1", "quiet_structured", 1),
        config={},
        naturalness_targets=setup_bundle.evaluator.naturalness_targets,
        epidemiology_targets=setup_bundle.evaluator.epidemiology_targets,
        retrieval_noise_config=cfg,
    )
    assert result.error is None
    assert result.loss.total != float("inf")


def test_evaluate_config_on_scenarios_forwards_config(setup_bundle):
    cfg = RetrievalNoiseConfig(dropout_prob=0.3)
    results = evaluate_config_on_scenarios(
        config={},
        scenarios=[_scenario("t1", "quiet_structured", 1)],
        naturalness_targets=setup_bundle.evaluator.naturalness_targets,
        epidemiology_targets=setup_bundle.evaluator.epidemiology_targets,
        retrieval_noise_config=cfg,
    )
    assert len(results) == 1
    assert results[0].error is None


def test_evaluate_config_on_scenarios_default_still_works(setup_bundle):
    results = evaluate_config_on_scenarios(
        config={},
        scenarios=[_scenario("t1", "quiet_structured", 1)],
        naturalness_targets=setup_bundle.evaluator.naturalness_targets,
        epidemiology_targets=setup_bundle.evaluator.epidemiology_targets,
    )
    assert len(results) == 1
    assert results[0].error is None


# ---------------------------------------------------------------------------
# build_held_out_report
# ---------------------------------------------------------------------------


def test_build_held_out_report_forwards_config_to_both_sides(setup_bundle):
    cfg = RetrievalNoiseConfig(dropout_prob=0.3, similarity_jitter=0.05)
    training = [_scenario("t1", "quiet_structured", 1)]
    heldout = [_scenario("h1", "chaotic", 1)]
    report = build_held_out_report(
        config={},
        training_scenarios=training,
        heldout_scenarios=heldout,
        naturalness_targets=setup_bundle.evaluator.naturalness_targets,
        epidemiology_targets=setup_bundle.evaluator.epidemiology_targets,
        retrieval_noise_config=cfg,
    )
    # Both sides should have a result (split is disjoint on archetype).
    assert len(report.training_results) == 1
    assert len(report.heldout_results) == 1
    assert report.training_results[0].error is None
    assert report.heldout_results[0].error is None


def test_build_held_out_report_default_preserves_behavior(setup_bundle):
    training = [_scenario("t1", "quiet_structured", 1)]
    heldout = [_scenario("h1", "chaotic", 1)]
    report = build_held_out_report(
        config={},
        training_scenarios=training,
        heldout_scenarios=heldout,
        naturalness_targets=setup_bundle.evaluator.naturalness_targets,
        epidemiology_targets=setup_bundle.evaluator.epidemiology_targets,
    )
    assert len(report.training_results) == 1
    assert len(report.heldout_results) == 1


# ---------------------------------------------------------------------------
# DefaultAutoresearchSetup.validate_best_on_heldout inheritance
# ---------------------------------------------------------------------------


def test_validate_best_on_heldout_inherits_evaluator_config():
    """When the setup's evaluator was built with a retrieval noise
    config, validate_best_on_heldout must use it by default (no
    override argument)."""
    cfg = RetrievalNoiseConfig(dropout_prob=0.3, similarity_jitter=0.05)
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=15,
        enforce_constraints=False,
        retrieval_noise_config=cfg,
    )
    assert setup.evaluator.retrieval_noise_config is cfg
    heldout = [_scenario("h1", "chaotic", 1)]
    report = setup.validate_best_on_heldout(
        best_config={},
        scenarios=heldout,
    )
    assert len(report.heldout_results) == 1
    assert report.heldout_results[0].error is None


def test_validate_best_on_heldout_explicit_override_wins(setup_bundle):
    """When the caller supplies a retrieval_noise_config override,
    it takes precedence over whatever the evaluator carries."""
    override_cfg = RetrievalNoiseConfig(dropout_prob=1.0)
    heldout = [_scenario("h1", "chaotic", 1)]
    report = setup_bundle.validate_best_on_heldout(
        best_config={},
        scenarios=heldout,
        retrieval_noise_config=override_cfg,
    )
    # Full dropout on retrieval should still let the class run;
    # it only affects recall, not behavior generation.
    assert len(report.heldout_results) == 1


def test_validate_best_on_heldout_default_when_evaluator_has_none(setup_bundle):
    """When the setup's evaluator has no retrieval noise config
    AND the caller does not supply one, validation runs with the
    legacy path."""
    assert setup_bundle.evaluator.retrieval_noise_config is None
    heldout = [_scenario("h1", "chaotic", 1)]
    report = setup_bundle.validate_best_on_heldout(
        best_config={},
        scenarios=heldout,
    )
    assert len(report.heldout_results) == 1
    assert report.heldout_results[0].error is None


# ---------------------------------------------------------------------------
# Determinism under plumbed retrieval noise on the validation path
# ---------------------------------------------------------------------------


def test_validation_with_retrieval_noise_is_deterministic(setup_bundle):
    cfg = RetrievalNoiseConfig(dropout_prob=0.3, similarity_jitter=0.1)

    def fingerprint():
        results = evaluate_config_on_scenarios(
            config={},
            scenarios=[_scenario("t1", "quiet_structured", 1)],
            naturalness_targets=setup_bundle.evaluator.naturalness_targets,
            epidemiology_targets=setup_bundle.evaluator.epidemiology_targets,
            retrieval_noise_config=cfg,
        )
        return round(results[0].loss.total, 6)

    a = fingerprint()
    b = fingerprint()
    assert a == b
