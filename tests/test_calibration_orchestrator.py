"""Tests for autoresearch orchestrator + proposer + applier.

Covers:
  - ParameterSpec / SearchSpace validation & clipping
  - RandomProposer / LatinHypercubeProposer / GridProposer
  - parse_key + parameter_override transient mutation
  - DefaultEvaluator integration
  - AutoresearchOrchestrator multi-start + checkpointing
  - PriorPredictiveReport
"""

from pathlib import Path
import json
import tempfile

import pytest

from src.calibration import (
    ParameterSpec,
    SearchSpace,
    RandomProposer,
    LatinHypercubeProposer,
    GridProposer,
    make_proposer,
    parse_key,
    ConfigKey,
    parameter_override,
    AutoresearchOrchestrator,
    build_default_evaluator,
    run_prior_predictive_check,
    Trial,
    LossResult,
)


# ==========================================================================
# ParameterSpec / SearchSpace
# ==========================================================================


def test_parameter_spec_clip_float():
    spec = ParameterSpec("x", 0.0, 1.0)
    assert spec.clip(-0.5) == 0.0
    assert spec.clip(0.5) == 0.5
    assert spec.clip(1.5) == 1.0


def test_parameter_spec_clip_int():
    spec = ParameterSpec("n", 1, 10, kind="int")
    assert spec.clip(0) == 1
    assert spec.clip(5.7) == 6
    assert spec.clip(15) == 10


def test_parameter_spec_clip_choice():
    spec = ParameterSpec("m", kind="choice", choices=["a", "b", "c"])
    assert spec.clip("b") == "b"
    assert spec.clip("z") == "a"  # falls back to first


def test_search_space_names_and_random():
    import random
    space = SearchSpace([
        ParameterSpec("a", 0, 1),
        ParameterSpec("b", 5, 10, kind="int"),
    ])
    assert space.names() == ["a", "b"]
    assert len(space) == 2
    cfg = space.random_config(random.Random(42))
    assert "a" in cfg and "b" in cfg
    assert 0 <= cfg["a"] <= 1
    assert 5 <= cfg["b"] <= 10


def test_search_space_clip_and_default():
    space = SearchSpace([
        ParameterSpec("a", 0.0, 1.0, default=0.5),
        ParameterSpec("b", 0.0, 1.0, default=0.3),
    ])
    cfg = space.clip_config({"a": 2.0})
    assert cfg["a"] == 1.0
    assert cfg["b"] == 0.3  # filled from default


def test_search_space_validate():
    space = SearchSpace([
        ParameterSpec("a", 0.0, 1.0),
    ])
    ok, errs = space.validate_config({"a": 0.5})
    assert ok
    ok, errs = space.validate_config({"a": 2.0})
    assert not ok
    ok, errs = space.validate_config({})
    assert not ok  # missing


# ==========================================================================
# Proposers
# ==========================================================================


def _simple_space():
    return SearchSpace([
        ParameterSpec("x", 0.0, 1.0, default=0.5),
        ParameterSpec("y", 0.0, 1.0, default=0.5),
    ])


def test_random_proposer_produces_in_range_configs():
    space = _simple_space()
    p = RandomProposer(space, seed=42)
    for _ in range(10):
        cfg = p.propose([])
        assert 0 <= cfg["x"] <= 1
        assert 0 <= cfg["y"] <= 1


def test_random_proposer_determinism():
    space = _simple_space()
    p1 = RandomProposer(space, seed=123)
    p2 = RandomProposer(space, seed=123)
    for _ in range(5):
        assert p1.propose([]) == p2.propose([])


def test_lhs_proposer_produces_initial_block():
    space = _simple_space()
    p = LatinHypercubeProposer(space, n_initial=4, seed=42)
    configs = [p.propose([]) for _ in range(4)]
    # All unique positions in the LHS
    x_vals = sorted(c["x"] for c in configs)
    # x values should span the range
    assert x_vals[0] < 0.5
    assert x_vals[-1] > 0.5


def test_lhs_falls_back_to_random_after_block():
    space = _simple_space()
    p = LatinHypercubeProposer(space, n_initial=2, seed=42)
    # Drain initial block
    p.propose([])
    p.propose([])
    # Third call → random fallback, still valid
    cfg = p.propose([])
    assert 0 <= cfg["x"] <= 1


def test_grid_proposer_covers_corners():
    space = _simple_space()
    p = GridProposer(space, levels=2, max_dims=2, seed=42)
    configs = []
    for _ in range(4):
        configs.append(p.propose([]))
    xs = {c["x"] for c in configs}
    ys = {c["y"] for c in configs}
    assert len(xs) == 2  # min + max
    assert len(ys) == 2


def test_make_proposer_dispatch():
    space = _simple_space()
    for kind in ("random", "lhs", "grid"):
        p = make_proposer(kind, space, seed=0)
        cfg = p.propose([])
        assert "x" in cfg


def test_make_proposer_unknown_raises():
    with pytest.raises(ValueError):
        make_proposer("unknown_kind", _simple_space())


# ==========================================================================
# parse_key + parameter_override
# ==========================================================================


def test_parse_key_base_forms():
    assert parse_key("base_emotional.shame").kind == "base_emotional"
    assert parse_key("base_observable.attention").kind == "base_observable"
    assert parse_key("base_cognitive.att_bandwidth").kind == "base_cognitive"


def test_parse_key_profile_delta():
    ck = parse_key("adhd_inattentive.emotional.frustration")
    assert ck.kind == "profile_delta"
    assert ck.profile == "adhd_inattentive"
    assert ck.section == "emotional"
    assert ck.field == "frustration"


def test_parse_key_invalid_section():
    with pytest.raises(ValueError):
        parse_key("adhd_inattentive.badsection.x")


def test_parse_key_wrong_parts():
    with pytest.raises(ValueError):
        parse_key("just_one_part")
    with pytest.raises(ValueError):
        parse_key("a.b.c.d")


def test_parameter_override_mutates_and_restores():
    from src.simulation import cognitive_agent as ca

    before = ca.EMOTIONAL_PRESETS["adhd_inattentive"]["shame"]
    with parameter_override({"adhd_inattentive.emotional.shame": 0.05}) as errs:
        assert not errs
        after = ca.EMOTIONAL_PRESETS["adhd_inattentive"]["shame"]
        # shame = BASE_EMOTIONAL + delta, clamped to [0, 1]
        # BASE 0.08 + 0.05 = 0.13
        assert abs(after - 0.13) < 1e-9
    # Restored
    assert ca.EMOTIONAL_PRESETS["adhd_inattentive"]["shame"] == before


def test_parameter_override_base_emotional():
    from src.simulation import cognitive_agent as ca

    original = ca.BASE_EMOTIONAL["shame"]
    with parameter_override({"base_emotional.shame": 0.15}):
        assert abs(ca.BASE_EMOTIONAL["shame"] - 0.15) < 1e-9
    assert ca.BASE_EMOTIONAL["shame"] == original


def test_parameter_override_integer_field_coercion():
    from src.simulation import cognitive_agent as ca

    original = ca.BASE_COGNITIVE.att_bandwidth
    with parameter_override({"base_cognitive.att_bandwidth": 2.7}):
        # Coerced to int(round(2.7)) = 3
        assert ca.BASE_COGNITIVE.att_bandwidth == 3
    assert ca.BASE_COGNITIVE.att_bandwidth == original


def test_parameter_override_reports_errors():
    with parameter_override({"unknown.section.x": 0.1}) as errs:
        assert any("Config key" in e or "Unhandled" in e or "section must" in e for e in errs)


# ==========================================================================
# DefaultEvaluator + AutoresearchOrchestrator
# ==========================================================================


def _make_small_evaluator():
    # Very fast evaluator: 1 class × 30 turns
    return build_default_evaluator(n_classes=1, max_turns=30, seed=42)


def test_default_evaluator_returns_finite_loss():
    ev = _make_small_evaluator()
    bundle, loss = ev.evaluate({})  # empty override = defaults
    assert bundle is not None
    assert loss is not None
    assert 0.0 <= loss.total < 100.0


def test_default_evaluator_accepts_override():
    ev = _make_small_evaluator()
    config = {"adhd_inattentive.emotional.frustration": 0.20}
    _, loss = ev.evaluate(config)
    assert isinstance(loss.total, float)


def test_orchestrator_runs_to_completion(tmp_path):
    space = SearchSpace([
        ParameterSpec("adhd_inattentive.emotional.frustration", 0.12, 0.25, default=0.19),
    ])
    ev = _make_small_evaluator()
    orch = AutoresearchOrchestrator(
        space=space,
        evaluator=ev,
        proposer_kind="random",
        n_iterations=2,
        n_starts=2,
        seed=7,
        results_dir=tmp_path,
    )
    result = orch.run()
    assert len(result.runs) == 2
    assert result.global_best_loss < float("inf")
    # Checkpoint file written
    assert (tmp_path / "orchestrator_checkpoint.json").exists()
    # Results tsv written
    assert (tmp_path / "results.tsv").exists()


def test_orchestrator_log_tsv_format(tmp_path):
    space = SearchSpace([
        ParameterSpec("adhd_inattentive.emotional.frustration", 0.12, 0.25, default=0.19),
    ])
    ev = _make_small_evaluator()
    orch = AutoresearchOrchestrator(
        space=space,
        evaluator=ev,
        proposer_kind="random",
        n_iterations=2,
        n_starts=1,
        seed=1,
        results_dir=tmp_path,
    )
    orch.run()

    log_path = tmp_path / "results.tsv"
    with open(log_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
    assert "loss_total" in header
    assert "loss_naturalness" in header
    assert "config_json" in header


def test_orchestrator_checkpoint_roundtrip(tmp_path):
    space = SearchSpace([
        ParameterSpec("adhd_inattentive.emotional.frustration", 0.12, 0.25, default=0.19),
    ])
    ev = _make_small_evaluator()
    orch = AutoresearchOrchestrator(
        space=space,
        evaluator=ev,
        proposer_kind="random",
        n_iterations=1,
        n_starts=2,
        seed=1,
        results_dir=tmp_path,
    )
    orch.run()

    loaded = orch.load_checkpoint()
    assert loaded is not None
    assert loaded["completed_runs"] == 2


def test_orchestrator_resume_skips_completed_and_restores_history(tmp_path):
    """After resume with n_starts equal to completed_runs:
      - no new starts are executed
      - prior run states are reconstructed into result.runs (summary-only)
      - global best metadata is restored
    """
    space = SearchSpace([
        ParameterSpec("adhd_inattentive.emotional.frustration", 0.12, 0.25, default=0.19),
    ])
    ev = _make_small_evaluator()
    orch = AutoresearchOrchestrator(
        space=space,
        evaluator=ev,
        proposer_kind="random",
        n_iterations=1,
        n_starts=2,
        seed=1,
        results_dir=tmp_path,
    )
    orch.run()

    # Resume: should pick up the global best from checkpoint AND rehydrate
    # the prior RunStates (as summary-only reconstructions).
    orch2 = AutoresearchOrchestrator(
        space=space,
        evaluator=ev,
        proposer_kind="random",
        n_iterations=1,
        n_starts=2,
        seed=1,
        results_dir=tmp_path,
    )
    result = orch2.run(resume=True)
    # completed_runs was 2, n_starts is 2 → no new runs, but the 2 old
    # runs are reconstructed into result.runs
    assert len(result.runs) == 2
    assert all(r.restored_from_checkpoint for r in result.runs)
    assert result.global_best_loss < float("inf")


def test_orchestrator_sensitivity_analysis(tmp_path):
    space = SearchSpace([
        ParameterSpec("adhd_inattentive.emotional.frustration", 0.12, 0.25, default=0.19),
        ParameterSpec("adhd_inattentive.emotional.shame", 0.10, 0.25, default=0.17),
    ])
    ev = _make_small_evaluator()
    orch = AutoresearchOrchestrator(
        space=space,
        evaluator=ev,
        proposer_kind="random",
        n_iterations=2,
        n_starts=1,
        seed=2,
        results_dir=tmp_path,
    )
    result = orch.run()
    reports = orch.sensitivity_analysis(
        best_config=result.global_best_config,
        perturbation=0.2,
    )
    # 2 dims × 2 signs = 4 reports
    assert len(reports) == 4
    for r in reports:
        assert "dim" in r and "delta" in r and "loss" in r


# ==========================================================================
# Prior predictive check
# ==========================================================================


def test_prior_predictive_smoke():
    report = run_prior_predictive_check(n_classes=1, max_turns=30, seed=3)
    assert report.n_total() > 0
    assert 0.0 <= report.coverage() <= 1.0
    assert report.summary()
    # per_target entries have the right shape
    for name, v in report.per_target.items():
        assert "in_range" in v
        assert "range" in v


# ==========================================================================
# Regression — invalid override must not equal baseline
# ==========================================================================


def test_invalid_override_raises_parameter_override_error():
    """Evaluator must raise ParameterOverrideError for unapplicable config.

    Previous behavior silently treated bad configs as baseline, contaminating
    search history with meaningless trials.
    """
    from src.calibration import ParameterOverrideError

    ev = _make_small_evaluator()
    with pytest.raises(ParameterOverrideError) as exc_info:
        ev.evaluate({"unknown.section.definitely_not_real": 0.1})
    # Error carries offending config + error list
    assert exc_info.value.errors
    assert "unknown.section.definitely_not_real" in exc_info.value.config


def test_invalid_override_not_equal_baseline():
    """Explicit regression: baseline loss != invalid-config loss.

    Under the old code path they would coincide because apply errors were
    silently swallowed. Now the invalid path must raise before running the
    simulator, so the orchestrator records a penalized trial.
    """
    from src.calibration import ParameterOverrideError

    ev = _make_small_evaluator()
    _, baseline_loss = ev.evaluate({})  # empty override = baseline

    with pytest.raises(ParameterOverrideError):
        ev.evaluate({"bogus.cognitive.att_bandwidth": 2})
    # Prove that *had* they been silently accepted, the baseline path was
    # the only way to reach this loss; now the invalid path explicitly
    # diverges via an exception.
    assert baseline_loss.total >= 0.0


def test_orchestrator_records_invalid_trials_with_penalty(tmp_path):
    """Orchestrator must convert ParameterOverrideError into a penalized
    trial — not crash, not silently use baseline loss.
    """
    from src.calibration.orchestrator import INVALID_CONFIG_PENALTY

    space = SearchSpace([
        # All values in this space will be routed to a profile that does NOT
        # exist, forcing every evaluator call to hit the invalid path.
        ParameterSpec(
            "nonexistent_profile.emotional.frustration",
            0.0,
            1.0,
            default=0.2,
        ),
    ])
    ev = _make_small_evaluator()
    orch = AutoresearchOrchestrator(
        space=space,
        evaluator=ev,
        proposer_kind="random",
        n_iterations=2,
        n_starts=1,
        seed=7,
        results_dir=tmp_path,
    )
    result = orch.run()
    assert len(result.runs) == 1
    run = result.runs[0]
    assert run.iterations == 2
    # Every trial should have been penalized (not baseline loss)
    for trial in run.history:
        assert trial.loss == INVALID_CONFIG_PENALTY
        # Metadata preserves the error info for debugging
        assert trial.metadata.get("error_type") == "parameter_override"
        assert "errors" in trial.metadata
        assert "offending_config" in trial.metadata
    # Best loss should also equal penalty (no valid trials existed)
    assert run.best_loss == INVALID_CONFIG_PENALTY


def test_orchestrator_survives_mix_of_valid_and_invalid(tmp_path):
    """When some configs are valid and some invalid, the loop still
    completes and records both kinds of trials."""
    from src.calibration import ParameterOverrideError
    from src.calibration.orchestrator import INVALID_CONFIG_PENALTY, EvaluatorProtocol
    from src.calibration.loss import LossResult

    # Custom evaluator that raises on odd iteration, returns 0.5 on even
    class ToggleEvaluator(EvaluatorProtocol):
        def __init__(self):
            self._count = 0

        def evaluate(self, config):
            self._count += 1
            if self._count % 2 == 1:
                raise ParameterOverrideError(
                    ["simulated error"], config
                )
            return (None, LossResult(total=0.5))

    space = SearchSpace([
        ParameterSpec("adhd_inattentive.emotional.frustration", 0.12, 0.25, default=0.19),
    ])
    orch = AutoresearchOrchestrator(
        space=space,
        evaluator=ToggleEvaluator(),
        proposer_kind="random",
        n_iterations=4,
        n_starts=1,
        seed=9,
        results_dir=tmp_path,
    )
    result = orch.run()
    run = result.runs[0]
    assert run.iterations == 4
    # 2 penalized, 2 valid
    losses = sorted(t.loss for t in run.history)
    assert losses.count(0.5) == 2
    assert losses.count(INVALID_CONFIG_PENALTY) == 2
    # Best is 0.5
    assert run.best_loss == 0.5


# ==========================================================================
# Regression — resume reconstructs prior run states
# ==========================================================================


def test_resume_reconstructs_prior_runs_and_appends_new(tmp_path):
    """Resume with larger n_starts:
      - first run: 2 starts → 2 completed
      - second run: resume with n_starts=4 → 2 restored + 2 new = 4 total
    """
    space = SearchSpace([
        ParameterSpec(
            "adhd_inattentive.emotional.frustration", 0.12, 0.25, default=0.19
        ),
    ])
    ev = _make_small_evaluator()

    orch = AutoresearchOrchestrator(
        space=space,
        evaluator=ev,
        proposer_kind="random",
        n_iterations=1,
        n_starts=2,
        seed=42,
        results_dir=tmp_path,
    )
    first_result = orch.run()
    assert len(first_result.runs) == 2

    orch2 = AutoresearchOrchestrator(
        space=space,
        evaluator=ev,
        proposer_kind="random",
        n_iterations=1,
        n_starts=4,  # larger
        seed=42,
        results_dir=tmp_path,
    )
    second_result = orch2.run(resume=True)

    assert len(second_result.runs) == 4
    # First 2 should be restored from checkpoint
    restored = [r for r in second_result.runs if r.restored_from_checkpoint]
    fresh = [r for r in second_result.runs if not r.restored_from_checkpoint]
    assert len(restored) == 2
    assert len(fresh) == 2
    # Fresh runs have start_ids 2 and 3
    fresh_ids = sorted(r.start_id for r in fresh)
    assert fresh_ids == [2, 3]
    # Global best should still be tracked (either from restored or new runs)
    assert second_result.global_best_loss < float("inf")
    # Summary string marks restored runs
    summary = second_result.summary()
    assert "[restored]" in summary


def test_run_state_from_dict_roundtrip():
    """RunState.to_dict() + from_dict() preserves summary fields."""
    from src.calibration.orchestrator import RunState

    original = RunState(
        start_id=5,
        seed=1234,
        iterations=10,
        best_loss=0.123456,
        best_config={"x": 0.5, "y": 0.3},
        elapsed_seconds=12.34,
    )
    payload = original.to_dict()
    restored = RunState.from_dict(payload)

    assert restored.start_id == 5
    assert restored.seed == 1234
    assert restored.iterations == 10
    assert abs(restored.best_loss - 0.123456) < 1e-9
    assert restored.best_config == {"x": 0.5, "y": 0.3}
    assert abs(restored.elapsed_seconds - 12.34) < 1e-9
    # history is intentionally empty after reconstruction
    assert restored.history == []
    assert restored.restored_from_checkpoint is True


def test_resume_restored_runs_have_empty_history(tmp_path):
    """Restored runs should have empty history (documented limitation)."""
    space = SearchSpace([
        ParameterSpec(
            "adhd_inattentive.emotional.frustration", 0.12, 0.25, default=0.19
        ),
    ])
    ev = _make_small_evaluator()

    orch = AutoresearchOrchestrator(
        space=space,
        evaluator=ev,
        proposer_kind="random",
        n_iterations=2,
        n_starts=1,
        seed=1,
        results_dir=tmp_path,
    )
    orch.run()

    orch2 = AutoresearchOrchestrator(
        space=space,
        evaluator=ev,
        proposer_kind="random",
        n_iterations=2,
        n_starts=1,
        seed=1,
        results_dir=tmp_path,
    )
    result = orch2.run(resume=True)

    assert len(result.runs) == 1
    assert result.runs[0].restored_from_checkpoint
    # Per-trial detail is NOT in the JSON checkpoint; it lives in results.tsv
    assert result.runs[0].history == []
