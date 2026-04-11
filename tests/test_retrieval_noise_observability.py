"""Tests for Phase 6 slice 12: retrieval-noise observability in
reports / summaries / checkpoint artifacts.

Covers:
  - DefaultAutoresearchSetup.summary() shows retrieval_noise when
    evaluator has a config, "disabled" otherwise
  - DefaultAutoresearchSetup.report() exposes the same
  - PriorPredictiveReport stores and surfaces retrieval_noise_config
  - HeldOutValidationReport stores and surfaces retrieval_noise_config
  - Orchestrator checkpoint JSON persists retrieval_noise_config
    (compact dict form)
  - Omitted config → "disabled" / None in all surfaces
  - Legacy output shape preserved (no breaking field removal)
"""

import json
from pathlib import Path

import pytest

from src.simulation.teacher_memory import RetrievalNoiseConfig
from src.calibration import (
    build_default_autoresearch_setup,
    run_prior_predictive_check,
    ValidationScenario,
)


# ---------------------------------------------------------------------------
# DefaultAutoresearchSetup summary / report
# ---------------------------------------------------------------------------


def test_setup_summary_shows_retrieval_noise_disabled_by_default(tmp_path):
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
    )
    s = setup.summary()
    assert "retrieval_noise=disabled" in s


def test_setup_summary_shows_retrieval_noise_when_configured(tmp_path):
    cfg = RetrievalNoiseConfig(dropout_prob=0.3, similarity_jitter=0.1)
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
        retrieval_noise_config=cfg,
    )
    s = setup.summary()
    assert "retrieval_noise=dropout=0.300,jitter=0.100" in s


def test_setup_report_shows_retrieval_noise(tmp_path):
    cfg = RetrievalNoiseConfig(dropout_prob=0.25, similarity_jitter=0.05)
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
        retrieval_noise_config=cfg,
    )
    r = setup.report()
    assert "Phase 6 noise policy:" in r
    assert "retrieval_noise_config:" in r
    assert "dropout=0.250,jitter=0.050" in r


def test_setup_report_disabled_without_config(tmp_path):
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
    )
    r = setup.report()
    assert "retrieval_noise_config: disabled" in r


# ---------------------------------------------------------------------------
# PriorPredictiveReport
# ---------------------------------------------------------------------------


def test_prior_predictive_report_stores_retrieval_noise_config():
    cfg = RetrievalNoiseConfig(dropout_prob=0.2, similarity_jitter=0.05)
    rep = run_prior_predictive_check(
        n_classes=1, max_turns=10, seed=1,
        retrieval_noise_config=cfg,
    )
    assert rep.retrieval_noise_config is cfg
    assert "Retrieval noise: dropout=0.200,jitter=0.050" in rep.summary()


def test_prior_predictive_report_disabled_by_default():
    rep = run_prior_predictive_check(n_classes=1, max_turns=10, seed=1)
    assert rep.retrieval_noise_config is None
    assert "Retrieval noise: disabled" in rep.summary()


# ---------------------------------------------------------------------------
# HeldOutValidationReport
# ---------------------------------------------------------------------------


def test_held_out_report_stores_retrieval_noise_config(tmp_path):
    cfg = RetrievalNoiseConfig(dropout_prob=0.15, similarity_jitter=0.02)
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
        retrieval_noise_config=cfg,
    )
    heldout = [ValidationScenario(
        name="h1", archetype="chaotic", seed=1,
        max_turns=10, n_students=5,
    )]
    report = setup.validate_best_on_heldout(
        best_config={}, scenarios=heldout,
    )
    # Inherited from evaluator.
    assert report.retrieval_noise_config is cfg
    s = report.summary()
    assert "Retrieval noise: dropout=0.150,jitter=0.020" in s


def test_held_out_report_disabled_when_evaluator_has_none(tmp_path):
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
    )
    heldout = [ValidationScenario(
        name="h1", archetype="chaotic", seed=1,
        max_turns=10, n_students=5,
    )]
    report = setup.validate_best_on_heldout(
        best_config={}, scenarios=heldout,
    )
    assert report.retrieval_noise_config is None
    s = report.summary()
    assert "Retrieval noise: disabled" in s


def test_held_out_report_explicit_override_surfaces(tmp_path):
    """When the caller overrides the config on
    validate_best_on_heldout, the override shows up on the
    report, not the evaluator's config."""
    ev_cfg = RetrievalNoiseConfig(dropout_prob=0.1)
    override_cfg = RetrievalNoiseConfig(dropout_prob=0.5)
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
        retrieval_noise_config=ev_cfg,
    )
    heldout = [ValidationScenario(
        name="h1", archetype="chaotic", seed=1,
        max_turns=10, n_students=5,
    )]
    report = setup.validate_best_on_heldout(
        best_config={}, scenarios=heldout,
        retrieval_noise_config=override_cfg,
    )
    assert report.retrieval_noise_config is override_cfg
    assert "dropout=0.500" in report.summary()


# ---------------------------------------------------------------------------
# Orchestrator checkpoint JSON
# ---------------------------------------------------------------------------


def test_orchestrator_checkpoint_persists_retrieval_noise_config(tmp_path):
    cfg = RetrievalNoiseConfig(dropout_prob=0.2, similarity_jitter=0.05)
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
        retrieval_noise_config=cfg,
    )
    setup.orchestrator.run()
    checkpoint_path = setup.orchestrator.checkpoint_path
    assert checkpoint_path.exists()
    payload = json.loads(checkpoint_path.read_text())
    assert "retrieval_noise_config" in payload
    assert payload["retrieval_noise_config"] == {
        "dropout_prob": 0.2,
        "similarity_jitter": 0.05,
    }


def test_orchestrator_checkpoint_records_none_when_no_config(tmp_path):
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
    )
    setup.orchestrator.run()
    payload = json.loads(setup.orchestrator.checkpoint_path.read_text())
    # Field present (stable shape) but value is None.
    assert "retrieval_noise_config" in payload
    assert payload["retrieval_noise_config"] is None


def test_orchestrator_checkpoint_preserves_existing_fields(tmp_path):
    """Regression: adding the new field must not break existing
    keys that downstream readers depend on."""
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
    )
    setup.orchestrator.run()
    payload = json.loads(setup.orchestrator.checkpoint_path.read_text())
    for required in (
        "completed_runs",
        "global_best_loss",
        "global_best_config",
        "global_best_start_id",
        "runs",
        "n_iterations",
        "n_starts",
        "proposer_kind",
        "timestamp",
    ):
        assert required in payload
