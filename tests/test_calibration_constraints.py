"""Tests for constraint-aware filtering in autoresearch (Phase 4.5).

Covers:
  - parse_rule pattern matching for supported grammar
  - parse_rule rejection of unsupported grammar
  - check_constraints positive / negative / unresolvable cases
  - ConstraintViolationError payload
  - Orchestrator integration (pre-evaluation filtering)
  - TSV error_type == "constraint_violation"
  - Trial metadata preserves violation details
  - Default setup path wires loaded constraints automatically
  - enforce_constraints=False disables the filter
"""

from pathlib import Path

import pytest

from src.calibration import (
    SupportedRule,
    UnsupportedRule,
    ConstraintCheckResult,
    ConstraintViolationError,
    parse_rule,
    parse_constraints,
    check_constraints,
    ParameterSpec,
    SearchSpace,
    AutoresearchOrchestrator,
    build_default_autoresearch_setup,
)
from src.calibration.orchestrator import (
    CONSTRAINT_VIOLATION_PENALTY,
    EvaluatorProtocol,
)
from src.calibration.loss import LossResult


# ==========================================================================
# parse_rule: supported patterns
# ==========================================================================


def test_parse_rule_abs_gt():
    rule = parse_rule(
        "abs(delta.cognitive.att_bandwidth) > abs(delta.cognitive.impulse_override)",
        profile="adhd_inattentive",
        rationale="DSM-5",
    )
    assert isinstance(rule, SupportedRule)
    assert rule.kind == "abs_gt"
    assert rule.left == ("cognitive", "att_bandwidth")
    assert rule.right == ("cognitive", "impulse_override")
    assert rule.profile == "adhd_inattentive"
    assert rule.rationale == "DSM-5"


def test_parse_rule_gt_zero():
    rule = parse_rule(
        "delta.emotional.anxiety > 0",
        profile="anxiety",
    )
    assert isinstance(rule, SupportedRule)
    assert rule.kind == "gt_zero"
    assert rule.left == ("emotional", "anxiety")


def test_parse_rule_lt_zero():
    rule = parse_rule(
        "delta.cognitive.plan_consistency < 0",
        profile="adhd_inattentive",
    )
    assert isinstance(rule, SupportedRule)
    assert rule.kind == "lt_zero"


def test_parse_rule_ge_zero_and_le_zero():
    ge = parse_rule("delta.emotional.shame >= 0", profile="adhd_inattentive")
    le = parse_rule("delta.cognitive.impulse_override <= 0", profile="anxiety")
    assert isinstance(ge, SupportedRule)
    assert ge.kind == "ge_zero"
    assert isinstance(le, SupportedRule)
    assert le.kind == "le_zero"


def test_parse_rule_handles_extra_whitespace():
    rule = parse_rule(
        "  abs(  delta.cognitive.att_bandwidth  )  >  abs(delta.cognitive.impulse_override)  ",
        profile="adhd_inattentive",
    )
    assert isinstance(rule, SupportedRule)
    assert rule.kind == "abs_gt"


# ==========================================================================
# parse_rule: unsupported patterns
# ==========================================================================


def test_parse_rule_missing_profile():
    rule = parse_rule("delta.emotional.anxiety > 0", profile=None)
    assert isinstance(rule, UnsupportedRule)
    assert "profile" in rule.reason


def test_parse_rule_compound_expression_rejected():
    rule = parse_rule(
        "delta.emotional.anxiety > 0 and delta.emotional.anger > 0",
        profile="anxiety",
    )
    assert isinstance(rule, UnsupportedRule)


def test_parse_rule_max_severity_clause_unsupported():
    """The adhd_combined max-severity rule is explicitly out of scope."""
    rule = parse_rule(
        "delta magnitude >= max(adhd_inattentive, adhd_hyperactive_impulsive)",
        profile="adhd_combined",
    )
    assert isinstance(rule, UnsupportedRule)


def test_parse_rule_unknown_operator():
    rule = parse_rule("delta.emotional.anxiety != 0", profile="anxiety")
    assert isinstance(rule, UnsupportedRule)


# ==========================================================================
# parse_constraints: split into supported / unsupported
# ==========================================================================


def test_parse_constraints_splits_lists():
    raw = [
        {
            "profile": "adhd_inattentive",
            "rule": "abs(delta.cognitive.att_bandwidth) > abs(delta.cognitive.impulse_override)",
            "rationale": "DSM-5",
        },
        {
            "profile": "anxiety",
            "rule": "delta.emotional.anxiety > 0",
        },
        {
            "profile": "adhd_combined",
            "rule": "delta magnitude >= max(adhd_inattentive, adhd_hyperactive_impulsive)",
        },
    ]
    supported, unsupported = parse_constraints(raw)
    assert len(supported) == 2
    assert len(unsupported) == 1
    assert unsupported[0].profile == "adhd_combined"


def test_parse_constraints_empty():
    supported, unsupported = parse_constraints([])
    assert supported == []
    assert unsupported == []


# ==========================================================================
# check_constraints: positive / negative / unresolvable
# ==========================================================================


def test_check_constraints_abs_gt_satisfied():
    rule = parse_rule(
        "abs(delta.cognitive.att_bandwidth) > abs(delta.cognitive.impulse_override)",
        profile="adhd_inattentive",
    )
    config = {
        "adhd_inattentive.cognitive.att_bandwidth": -3,
        "adhd_inattentive.cognitive.impulse_override": 0.1,
    }
    result = check_constraints(config, [rule], [])
    assert result.valid
    assert result.violations == []


def test_check_constraints_abs_gt_violated():
    rule = parse_rule(
        "abs(delta.cognitive.att_bandwidth) > abs(delta.cognitive.impulse_override)",
        profile="adhd_inattentive",
    )
    config = {
        "adhd_inattentive.cognitive.att_bandwidth": 0,
        "adhd_inattentive.cognitive.impulse_override": 0.5,
    }
    result = check_constraints(config, [rule], [])
    assert not result.valid
    assert len(result.violations) == 1


def test_check_constraints_gt_zero_satisfied_and_violated():
    rule = parse_rule("delta.emotional.anxiety > 0", profile="anxiety")
    # Satisfied
    config_ok = {"anxiety.emotional.anxiety": 0.3}
    assert check_constraints(config_ok, [rule], []).valid
    # Violated (zero is not strictly > 0)
    config_bad = {"anxiety.emotional.anxiety": 0.0}
    assert not check_constraints(config_bad, [rule], []).valid
    config_neg = {"anxiety.emotional.anxiety": -0.1}
    assert not check_constraints(config_neg, [rule], []).valid


def test_check_constraints_lt_zero():
    rule = parse_rule("delta.cognitive.plan_consistency < 0", profile="adhd_inattentive")
    assert check_constraints(
        {"adhd_inattentive.cognitive.plan_consistency": -0.3}, [rule], []
    ).valid
    assert not check_constraints(
        {"adhd_inattentive.cognitive.plan_consistency": 0.0}, [rule], []
    ).valid


def test_check_constraints_fallback_to_simulator():
    """If config is missing a field, fall back to cognitive_agent PROFILE_DELTAS."""
    rule = parse_rule(
        "abs(delta.cognitive.att_bandwidth) > abs(delta.cognitive.impulse_override)",
        profile="adhd_inattentive",
    )
    # Empty config — should use live simulator values
    result = check_constraints({}, [rule], [])
    # adhd_inattentive's live deltas: att_bandwidth=-2, impulse_override=+0.075
    # |−2| = 2 > |0.075| → valid
    assert result.valid


def test_check_constraints_unresolvable():
    """Unknown profile → unresolvable, not violation."""
    rule = SupportedRule(
        kind="gt_zero",
        profile="nonexistent_profile",
        left=("emotional", "anxiety"),
        raw="delta.emotional.anxiety > 0",
    )
    result = check_constraints({}, [rule], [])
    # Not violated because we could not resolve the field
    assert result.valid
    assert len(result.unresolvable) == 1


def test_check_constraints_unsupported_surfaced():
    rule = UnsupportedRule(
        profile="adhd_combined",
        raw="delta magnitude >= max(...)",
        reason="unsupported pattern",
    )
    result = check_constraints({}, [], [rule])
    assert result.valid  # not enforced
    assert len(result.unsupported) == 1


def test_describe_violations_readable():
    rule = SupportedRule(
        kind="gt_zero",
        profile="anxiety",
        left=("emotional", "anxiety"),
        raw="delta.emotional.anxiety > 0",
    )
    result = ConstraintCheckResult(valid=False, violations=[rule])
    messages = result.describe_violations()
    assert len(messages) == 1
    assert "anxiety" in messages[0]
    assert "> 0" in messages[0]


# ==========================================================================
# ConstraintViolationError payload
# ==========================================================================


def test_constraint_violation_error_carries_details():
    rule = parse_rule(
        "delta.emotional.anxiety > 0", profile="anxiety"
    )
    config = {"anxiety.emotional.anxiety": -0.2}
    exc = ConstraintViolationError([rule], config)
    assert exc.config == config
    assert len(exc.violations) == 1
    assert "anxiety" in str(exc)


# ==========================================================================
# Orchestrator integration
# ==========================================================================


def _passthrough_evaluator():
    """Evaluator that would always return 0.5, used to verify
    constraint check runs BEFORE the evaluator."""
    class PassEval(EvaluatorProtocol):
        def __init__(self):
            self.call_count = 0
        def evaluate(self, config):
            self.call_count += 1
            return (None, LossResult(total=0.5))
    return PassEval()


def test_orchestrator_blocks_violating_config(tmp_path):
    """A config that violates a supported constraint must NOT reach the
    evaluator; it gets a CONSTRAINT_VIOLATION_PENALTY trial."""
    rule = parse_rule(
        "delta.emotional.anxiety > 0", profile="anxiety"
    )
    # Search space forces anxiety delta to be <= 0 (violation)
    space = SearchSpace([
        ParameterSpec(
            "anxiety.emotional.anxiety",
            -0.3, -0.1,
            default=-0.2,
        ),
    ])
    ev = _passthrough_evaluator()
    orch = AutoresearchOrchestrator(
        space=space,
        evaluator=ev,
        proposer_kind="random",
        n_iterations=3,
        n_starts=1,
        seed=5,
        results_dir=tmp_path,
        supported_constraints=[rule],
    )
    result = orch.run()
    # Evaluator never called
    assert ev.call_count == 0
    # All trials penalized
    run = result.runs[0]
    for trial in run.history:
        assert trial.loss == CONSTRAINT_VIOLATION_PENALTY
        assert trial.metadata["error_type"] == "constraint_violation"
        assert "violations" in trial.metadata
        assert trial.metadata["violations"][0]["kind"] == "gt_zero"


def test_orchestrator_allows_valid_config(tmp_path):
    """A config that satisfies all supported constraints reaches the evaluator."""
    rule = parse_rule(
        "delta.emotional.anxiety > 0", profile="anxiety"
    )
    space = SearchSpace([
        ParameterSpec(
            "anxiety.emotional.anxiety",
            0.1, 0.3,  # always positive → valid
            default=0.2,
        ),
    ])
    ev = _passthrough_evaluator()
    orch = AutoresearchOrchestrator(
        space=space,
        evaluator=ev,
        proposer_kind="random",
        n_iterations=2,
        n_starts=1,
        seed=5,
        results_dir=tmp_path,
        supported_constraints=[rule],
    )
    result = orch.run()
    # Evaluator called for every iteration
    assert ev.call_count == 2
    run = result.runs[0]
    for trial in run.history:
        assert trial.loss == 0.5
        assert not trial.metadata.get("error_type")


def test_orchestrator_mixed_valid_and_violating(tmp_path):
    """Mix of valid and violating trials — orchestrator handles both paths."""
    # Two dimensions: one forcing violation, one harmless
    rule = parse_rule(
        "delta.emotional.anxiety > 0", profile="anxiety"
    )
    # The violating rule looks at "anxiety.emotional.anxiety"
    # We'll deliberately vary that dim across iterations via grid
    space = SearchSpace([
        ParameterSpec(
            "anxiety.emotional.anxiety",
            -0.2, 0.2,
            default=0.0,
        ),
    ])
    ev = _passthrough_evaluator()
    orch = AutoresearchOrchestrator(
        space=space,
        evaluator=ev,
        proposer_kind="random",
        n_iterations=20,
        n_starts=1,
        seed=42,
        results_dir=tmp_path,
        supported_constraints=[rule],
    )
    result = orch.run()
    run = result.runs[0]
    penalized = [t for t in run.history if t.loss == CONSTRAINT_VIOLATION_PENALTY]
    valid = [t for t in run.history if t.loss == 0.5]
    # Both kinds should exist with 20 random draws from a symmetric range
    assert penalized  # at least one
    assert valid  # at least one
    assert len(penalized) + len(valid) == 20
    # Evaluator only called for valid
    assert ev.call_count == len(valid)


def test_orchestrator_tsv_constraint_violation_row(tmp_path):
    """A constraint violation is logged with error_type=constraint_violation
    and detail JSON containing the violated rule info."""
    import csv as _csv
    import json as _json

    rule = parse_rule(
        "delta.emotional.anxiety > 0", profile="anxiety"
    )
    space = SearchSpace([
        ParameterSpec("anxiety.emotional.anxiety", -0.3, -0.1, default=-0.2),
    ])
    ev = _passthrough_evaluator()
    orch = AutoresearchOrchestrator(
        space=space,
        evaluator=ev,
        proposer_kind="random",
        n_iterations=1,
        n_starts=1,
        seed=1,
        results_dir=tmp_path,
        supported_constraints=[rule],
    )
    orch.run()

    with open(tmp_path / "results.tsv", "r", encoding="utf-8") as f:
        reader = _csv.reader(f, delimiter="\t")
        header = next(reader)
        rows = list(reader)

    assert len(rows) == 1
    row = dict(zip(header, rows[0]))
    assert row["error_type"] == "constraint_violation"
    assert row["error_details_json"]
    detail = _json.loads(row["error_details_json"])
    assert "violations" in detail
    assert detail["violations"][0]["profile"] == "anxiety"
    assert detail["violations"][0]["kind"] == "gt_zero"


# ==========================================================================
# Default setup path automatically uses loaded constraints
# ==========================================================================


def test_default_setup_loads_and_parses_harness_constraints(tmp_path):
    """The default factory must parse harness constraints into supported /
    unsupported lists and forward supported ones to the orchestrator."""
    setup = build_default_autoresearch_setup(
        n_iterations=1,
        n_starts=1,
        n_classes=1,
        max_turns=30,
        results_dir=tmp_path,
    )
    # Real harness has 5 constraints; at least some are supported
    assert len(setup.supported_rules) >= 3
    # At least the max-severity clause is unsupported
    assert len(setup.unsupported_rules) >= 1

    # Orchestrator received the supported rules
    assert setup.orchestrator.supported_constraints
    assert len(setup.orchestrator.supported_constraints) == len(setup.supported_rules)


def test_default_setup_enforce_constraints_false(tmp_path):
    """enforce_constraints=False leaves orchestrator without supported rules."""
    setup = build_default_autoresearch_setup(
        n_iterations=1,
        n_starts=1,
        n_classes=1,
        max_turns=30,
        results_dir=tmp_path,
        enforce_constraints=False,
    )
    # Rules still parsed and accessible on the bundle
    assert setup.supported_rules  # non-empty (parsed)
    # But orchestrator received empty list
    assert setup.orchestrator.supported_constraints == []


def test_default_setup_summary_shows_constraint_counts(tmp_path):
    setup = build_default_autoresearch_setup(
        n_iterations=1,
        n_starts=1,
        n_classes=1,
        max_turns=30,
        results_dir=tmp_path,
    )
    s = setup.summary()
    assert "supported_rules=" in s
    assert "unsupported_rules=" in s
    r = setup.report()
    assert "constraint enforcement: active" in r


def test_default_setup_report_marks_enforcement_inactive(tmp_path):
    setup = build_default_autoresearch_setup(
        n_iterations=1,
        n_starts=1,
        n_classes=1,
        max_turns=30,
        results_dir=tmp_path,
        enforce_constraints=False,
    )
    r = setup.report()
    assert "constraint enforcement: inactive" in r
