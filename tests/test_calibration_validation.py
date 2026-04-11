"""Tests for Phase 4.5 held-out / cross-scenario validation layer.

Covers:
  - ValidationScenario dataclass construction
  - per-scenario evaluation produces results with real loss
  - deterministic repeated evaluation under fixed seed
  - aggregate held-out loss computation + train-vs-heldout gap
  - DefaultAutoresearchSetup.validate_best_on_heldout convenience path
  - scenarios_overlap helper rejects overlapping splits
"""

import pytest

from src.calibration import (
    ValidationScenario,
    ValidationResult,
    HeldOutValidationReport,
    ScenarioSplitError,
    scenarios_overlap,
    evaluate_config_on_scenarios,
    build_held_out_report,
    build_default_autoresearch_setup,
)


# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------


def test_validation_scenario_defaults():
    sc = ValidationScenario(name="s1")
    assert sc.name == "s1"
    assert sc.archetype is None
    assert sc.seed == 0
    assert sc.max_turns == 200
    assert sc.n_students == 20
    assert sc.n_classes == 1
    assert sc.note == ""


def test_validation_scenario_fields():
    sc = ValidationScenario(
        name="calm_1",
        archetype="quiet_structured",
        seed=100,
        max_turns=30,
        n_students=5,
        n_classes=2,
        note="heldout",
    )
    assert sc.archetype == "quiet_structured"
    assert sc.seed == 100
    assert sc.n_classes == 2
    assert sc.note == "heldout"


# ---------------------------------------------------------------------------
# scenarios_overlap helper
# ---------------------------------------------------------------------------


def test_scenarios_overlap_detects_shared_key():
    a = [ValidationScenario(name="train_1", archetype="quiet_structured", seed=1,
                             max_turns=20, n_students=5)]
    b = [ValidationScenario(name="heldout_1", archetype="quiet_structured", seed=1,
                             max_turns=20, n_students=5)]
    assert scenarios_overlap(a, b)  # names differ but identity key identical


def test_scenarios_overlap_disjoint_by_seed():
    a = [ValidationScenario(name="a", archetype="quiet_structured", seed=1,
                             max_turns=20, n_students=5)]
    b = [ValidationScenario(name="b", archetype="quiet_structured", seed=2,
                             max_turns=20, n_students=5)]
    assert not scenarios_overlap(a, b)


def test_scenarios_overlap_disjoint_by_archetype():
    a = [ValidationScenario(name="a", archetype="quiet_structured", seed=1,
                             max_turns=20, n_students=5)]
    b = [ValidationScenario(name="b", archetype="chaotic", seed=1,
                             max_turns=20, n_students=5)]
    assert not scenarios_overlap(a, b)


# ---------------------------------------------------------------------------
# Shared fixture: default setup for real-simulator-backed tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def setup_module_bundle():
    return build_default_autoresearch_setup(
        n_iterations=1,
        n_starts=1,
        n_classes=1,
        max_turns=20,
        enforce_constraints=False,
    )


# ---------------------------------------------------------------------------
# Per-scenario evaluation
# ---------------------------------------------------------------------------


def test_evaluate_config_on_scenarios_produces_results(setup_module_bundle):
    setup = setup_module_bundle
    scenarios = [
        ValidationScenario(name="calm_1", archetype="quiet_structured",
                            seed=100, max_turns=20, n_students=5),
        ValidationScenario(name="chaos_1", archetype="chaotic",
                            seed=200, max_turns=20, n_students=5),
    ]
    results = evaluate_config_on_scenarios(
        config={},
        scenarios=scenarios,
        naturalness_targets=setup.evaluator.naturalness_targets,
        epidemiology_targets=setup.evaluator.epidemiology_targets,
    )
    assert len(results) == 2
    for r in results:
        assert isinstance(r, ValidationResult)
        assert r.error is None
        assert r.loss.total != float("inf")
        assert r.bundle is not None
        assert len(r.bundle.histories) >= 1


# ---------------------------------------------------------------------------
# Determinism under fixed seeds
# ---------------------------------------------------------------------------


def test_evaluate_config_on_scenarios_is_deterministic(setup_module_bundle):
    setup = setup_module_bundle
    scenarios = [
        ValidationScenario(name="calm_1", archetype="quiet_structured",
                            seed=100, max_turns=20, n_students=5),
    ]
    r1 = evaluate_config_on_scenarios(
        config={},
        scenarios=scenarios,
        naturalness_targets=setup.evaluator.naturalness_targets,
        epidemiology_targets=setup.evaluator.epidemiology_targets,
    )
    r2 = evaluate_config_on_scenarios(
        config={},
        scenarios=scenarios,
        naturalness_targets=setup.evaluator.naturalness_targets,
        epidemiology_targets=setup.evaluator.epidemiology_targets,
    )
    assert r1[0].loss.total == pytest.approx(r2[0].loss.total)


# ---------------------------------------------------------------------------
# Aggregate report: train-vs-heldout
# ---------------------------------------------------------------------------


def test_build_held_out_report_aggregates_and_computes_gap(setup_module_bundle):
    setup = setup_module_bundle
    training = [
        ValidationScenario(name="train_calm", archetype="quiet_structured",
                            seed=1, max_turns=20, n_students=5),
    ]
    heldout = [
        ValidationScenario(name="heldout_chaos", archetype="chaotic",
                            seed=50, max_turns=20, n_students=5),
    ]
    # Splits are actually disjoint
    assert not scenarios_overlap(training, heldout)

    report = build_held_out_report(
        config={},
        training_scenarios=training,
        heldout_scenarios=heldout,
        naturalness_targets=setup.evaluator.naturalness_targets,
        epidemiology_targets=setup.evaluator.epidemiology_targets,
    )
    assert isinstance(report, HeldOutValidationReport)
    assert len(report.training_results) == 1
    assert len(report.heldout_results) == 1

    t = report.aggregate_training_loss()
    h = report.aggregate_heldout_loss()
    gap = report.train_vs_heldout_gap()

    assert t is not None
    assert h is not None
    assert gap is not None
    assert gap == pytest.approx(h - t)

    summary = report.summary()
    assert "Held-out Validation Report" in summary
    assert "train_calm" in summary
    assert "heldout_chaos" in summary


# ---------------------------------------------------------------------------
# DefaultAutoresearchSetup convenience method
# ---------------------------------------------------------------------------


def test_default_setup_validate_best_on_heldout(setup_module_bundle):
    setup = setup_module_bundle
    training = [
        ValidationScenario(name="train_1", archetype="quiet_structured",
                            seed=1, max_turns=20, n_students=5),
    ]
    heldout = [
        ValidationScenario(name="heldout_1", archetype="chaotic",
                            seed=50, max_turns=20, n_students=5),
    ]
    report = setup.validate_best_on_heldout(
        best_config={},
        scenarios=heldout,
        training_scenarios=training,
    )
    assert isinstance(report, HeldOutValidationReport)
    assert len(report.training_results) == 1
    assert len(report.heldout_results) == 1
    assert report.aggregate_heldout_loss() is not None


# ---------------------------------------------------------------------------
# Problem 2 regression: split overlap must be enforced, not just testable
# ---------------------------------------------------------------------------


def test_build_held_out_report_rejects_exact_overlap(setup_module_bundle):
    setup = setup_module_bundle
    shared = ValidationScenario(
        name="same", archetype="quiet_structured",
        seed=1, max_turns=20, n_students=5,
    )
    # Different label, identical identity key — still overlap.
    heldout_alias = ValidationScenario(
        name="same_relabeled", archetype="quiet_structured",
        seed=1, max_turns=20, n_students=5,
    )
    with pytest.raises(ScenarioSplitError) as excinfo:
        build_held_out_report(
            config={},
            training_scenarios=[shared],
            heldout_scenarios=[heldout_alias],
            naturalness_targets=setup.evaluator.naturalness_targets,
            epidemiology_targets=setup.evaluator.epidemiology_targets,
        )
    err = excinfo.value
    assert err.overlapping_keys
    assert "archetype" in str(err) or "quiet_structured" in str(err)


def test_build_held_out_report_accepts_disjoint_splits(setup_module_bundle):
    setup = setup_module_bundle
    training = [
        ValidationScenario(name="t1", archetype="quiet_structured",
                            seed=1, max_turns=20, n_students=5),
    ]
    heldout = [
        ValidationScenario(name="h1", archetype="chaotic",
                            seed=1, max_turns=20, n_students=5),
    ]
    # Should not raise — and should produce real results.
    report = build_held_out_report(
        config={},
        training_scenarios=training,
        heldout_scenarios=heldout,
        naturalness_targets=setup.evaluator.naturalness_targets,
        epidemiology_targets=setup.evaluator.epidemiology_targets,
    )
    assert len(report.training_results) == 1
    assert len(report.heldout_results) == 1


def test_validate_best_on_heldout_rejects_overlap(setup_module_bundle):
    setup = setup_module_bundle
    shared = [
        ValidationScenario(name="s", archetype="chaotic",
                            seed=9, max_turns=20, n_students=5),
    ]
    with pytest.raises(ScenarioSplitError):
        setup.validate_best_on_heldout(
            best_config={},
            scenarios=shared,
            training_scenarios=shared,
        )


def test_build_held_out_report_no_training_skips_check(setup_module_bundle):
    """With an empty training list, there is nothing to overlap with
    — the check is skipped and the report runs normally."""
    setup = setup_module_bundle
    heldout = [
        ValidationScenario(name="h", archetype="chaotic",
                            seed=1, max_turns=20, n_students=5),
    ]
    report = build_held_out_report(
        config={},
        training_scenarios=[],
        heldout_scenarios=heldout,
        naturalness_targets=setup.evaluator.naturalness_targets,
        epidemiology_targets=setup.evaluator.epidemiology_targets,
    )
    assert len(report.heldout_results) == 1
    assert report.aggregate_training_loss() is None


def test_default_setup_validate_best_on_heldout_without_training(setup_module_bundle):
    setup = setup_module_bundle
    heldout = [
        ValidationScenario(name="h1", archetype="chaotic",
                            seed=50, max_turns=20, n_students=5),
    ]
    report = setup.validate_best_on_heldout(
        best_config={},
        scenarios=heldout,
    )
    assert report.aggregate_training_loss() is None
    assert report.train_vs_heldout_gap() is None
    assert report.aggregate_heldout_loss() is not None
