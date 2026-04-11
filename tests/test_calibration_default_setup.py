"""Tests for the default autoresearch wiring — Phase 4.5 usability layer.

Proves the default path works end-to-end:
    harness YAML → LoadedSearchSpace → DefaultEvaluator → AutoresearchOrchestrator
without any manual ParameterSpec assembly.

Covers:
  - Factory returns a valid bundle
  - Loaded constraints / metadata / unsupported sections remain accessible
  - Override arguments propagate correctly to orchestrator + evaluator
  - End-to-end tiny run produces a valid result
  - results.tsv + checkpoint JSON are written
  - Default search space really comes from the harness file (not a
    hardcoded mini-space)
  - Summary / report strings are informative
"""

from pathlib import Path

import pytest

from src.calibration import (
    build_default_autoresearch_setup,
    DefaultAutoresearchSetup,
    LoadedSearchSpace,
    AutoresearchOrchestrator,
    DefaultEvaluator,
    default_student_ranges_path,
)


# ==========================================================================
# Factory return shape
# ==========================================================================


def test_factory_returns_bundle(tmp_path):
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=30,
        results_dir=tmp_path,
    )
    assert isinstance(setup, DefaultAutoresearchSetup)
    assert isinstance(setup.loaded_space, LoadedSearchSpace)
    assert isinstance(setup.evaluator, DefaultEvaluator)
    assert isinstance(setup.orchestrator, AutoresearchOrchestrator)


def test_factory_uses_real_harness_yaml(tmp_path):
    """Default search space must come from the harness file, not a stub."""
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=30,
        results_dir=tmp_path,
    )
    # Real harness has ≥30 parameters; a stub would have fewer
    assert len(setup.loaded_space.space) >= 30
    # Known harness parameter names must be present
    names = setup.loaded_space.space.names()
    assert "base_cognitive.att_bandwidth" in names
    assert "adhd_inattentive.emotional.frustration" in names


def test_factory_preserves_constraints(tmp_path):
    """Constraints parsed from YAML must remain accessible via the bundle."""
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=30,
        results_dir=tmp_path,
    )
    # Real harness has at least the 5 documented constraints
    assert len(setup.loaded_space.constraints) >= 4
    # Each constraint is a dict with a 'rule' field
    for c in setup.loaded_space.constraints:
        assert "rule" in c
        assert isinstance(c["rule"], str)


def test_factory_preserves_metadata(tmp_path):
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=30,
        results_dir=tmp_path,
    )
    assert "version" in setup.loaded_space.metadata
    assert "philosophy" in setup.loaded_space.metadata


def test_factory_preserves_unsupported_sections(tmp_path):
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=30,
        results_dir=tmp_path,
    )
    # Known unsupported YAML sections are tracked for future work
    unsupported = set(setup.loaded_space.unsupported_sections)
    assert "emotion_decay_rates" in unsupported
    assert "event_reactivity" in unsupported


# ==========================================================================
# Override argument propagation
# ==========================================================================


def test_override_orchestrator_args(tmp_path):
    setup = build_default_autoresearch_setup(
        n_iterations=7,
        n_starts=3,
        seed=99,
        proposer_kind="lhs",
        results_dir=tmp_path,
        early_stop_patience=5,
        proposer_kwargs={"n_initial": 4},
    )
    o = setup.orchestrator
    assert o.n_iterations == 7
    assert o.n_starts == 3
    assert o.seed == 99
    assert o.proposer_kind == "lhs"
    assert o.early_stop_patience == 5
    assert o.proposer_kwargs == {"n_initial": 4}
    assert o.results_dir == tmp_path


def test_override_evaluator_args(tmp_path):
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1,
        n_classes=5,
        max_turns=123,
        n_students=15,
        results_dir=tmp_path,
    )
    e = setup.evaluator
    assert e.n_classes == 5
    assert e.max_turns == 123
    assert e.n_students == 15


def test_default_results_dir_is_harness(tmp_path, monkeypatch):
    """When results_dir is omitted, default is `.harness` relative to cwd."""
    monkeypatch.chdir(tmp_path)
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=30,
    )
    assert setup.orchestrator.results_dir == Path(".harness")


# ==========================================================================
# End-to-end smoke run from harness YAML
# ==========================================================================


def test_default_setup_runs_end_to_end(tmp_path):
    """Tiny calibration pass from real harness file, verify result + artifacts."""
    setup = build_default_autoresearch_setup(
        n_iterations=1,
        n_starts=1,
        n_classes=1,
        max_turns=30,
        seed=7,
        results_dir=tmp_path,
    )
    result = setup.orchestrator.run()

    # Result is valid
    assert len(result.runs) == 1
    run = result.runs[0]
    assert run.iterations == 1
    assert run.best_loss < float("inf")
    assert result.global_best_loss < float("inf")

    # Search space is non-empty (harness-driven)
    assert len(setup.loaded_space.space) > 0

    # Constraints + metadata still accessible post-run
    assert len(setup.loaded_space.constraints) > 0
    assert setup.loaded_space.metadata

    # Artifacts written
    assert (tmp_path / "results.tsv").exists()
    assert (tmp_path / "orchestrator_checkpoint.json").exists()


def test_default_setup_multi_start_run(tmp_path):
    """2-start run produces 2 RunStates."""
    setup = build_default_autoresearch_setup(
        n_iterations=1,
        n_starts=2,
        n_classes=1,
        max_turns=30,
        seed=13,
        results_dir=tmp_path,
    )
    result = setup.orchestrator.run()
    assert len(result.runs) == 2
    assert all(r.best_loss < float("inf") for r in result.runs)


# ==========================================================================
# Summary / report helpers
# ==========================================================================


def test_summary_includes_core_fields(tmp_path):
    setup = build_default_autoresearch_setup(
        n_iterations=3, n_starts=2, n_classes=1, max_turns=30,
        results_dir=tmp_path,
    )
    s = setup.summary()
    assert "n_params=" in s
    assert "n_constraints=" in s
    assert "proposer=" in s
    assert "n_starts=2" in s
    assert "n_iterations=3" in s


def test_report_multiline_diagnostic(tmp_path):
    setup = build_default_autoresearch_setup(
        n_iterations=3, n_starts=2, n_classes=1, max_turns=30,
        results_dir=tmp_path,
    )
    r = setup.report()
    assert "Default Autoresearch Setup" in r
    assert "Search space:" in r
    assert "Constraints:" in r
    assert "Orchestrator:" in r
    assert "Evaluator:" in r
    # Multi-line
    assert r.count("\n") >= 10


# ==========================================================================
# Alternative YAML paths
# ==========================================================================


def test_override_search_space_yaml(tmp_path):
    """Alternative search-space YAML path is honored."""
    fake_yaml = tmp_path / "tiny.yaml"
    fake_yaml.write_text(
        """
base_emotional:
  frustration:
    range: [0.05, 0.20]
    default: 0.1
""",
        encoding="utf-8",
    )
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=30,
        results_dir=tmp_path,
        search_space_yaml=fake_yaml,
    )
    # Only 1 param from the tiny file
    assert len(setup.loaded_space.space) == 1
    assert setup.loaded_space.space.specs[0].name == "base_emotional.frustration"


def test_override_target_yamls_from_real_harness(tmp_path):
    """Overriding naturalness/epidemiology YAMLs still works."""
    from src.calibration import default_harness_paths
    nat_path, epi_path = default_harness_paths()
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=30,
        results_dir=tmp_path,
        naturalness_yaml=nat_path,
        epidemiology_yaml=epi_path,
    )
    # Real harness targets → non-zero lists
    assert len(setup.evaluator.naturalness_targets) > 0
    assert len(setup.evaluator.epidemiology_targets) > 0
