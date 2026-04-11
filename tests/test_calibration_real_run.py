"""Real-run integration smoke tests for the calibration bridge.

Validates that `run_real_bundle(OrchestratorV2)` produces a
`CalibrationResultBundle` on which the currently-supported metrics can be
evaluated end-to-end — without any synthetic fixtures.

These tests are slower than the unit tests (they run a real mini-simulation)
but remain bounded by reducing MAX_TURNS.
"""

import pytest

from src.simulation.orchestrator_v2 import OrchestratorV2
from src.calibration import (
    run_real_bundle,
    class_result_to_history,
    compute_combined_loss,
    resolve_metric,
    EpidemiologyTarget,
    NaturalnessTarget,
    ClassHistory,
    CalibrationResultBundle,
)


def _make_short_orchestrator(n_students=12, seed=17, max_turns=200):
    """Create an orchestrator configured for fast smoke runs.

    Uses 200 turns to ensure phase 2 (suspicion) and phase 3 fire at
    least briefly. Real 950-turn runs are too slow for unit tests.
    """
    orch = OrchestratorV2(n_students=n_students, max_classes=1, seed=seed)
    orch.classroom.MAX_TURNS = max_turns
    return orch


# ==========================================================================
# Adapter basics
# ==========================================================================


def test_real_run_produces_nonempty_bundle():
    orch = _make_short_orchestrator()
    bundle = run_real_bundle(orch, n_classes=1)
    assert len(bundle) == 1
    h = bundle.histories[0]
    assert len(h.students_roster) == 12
    assert len(h.events) > 0
    # Roster entries have the required keys
    for s in h.students_roster:
        assert set(s.keys()) >= {"id", "profile_type", "gender", "is_adhd"}
    # Metrics pass-through
    assert h.metrics is not None


def test_real_run_patience_log_populated():
    orch = _make_short_orchestrator(max_turns=50)
    bundle = run_real_bundle(orch, n_classes=1)
    h = bundle.histories[0]
    assert h.teacher_patience_log is not None
    assert len(h.teacher_patience_log) == 50
    for v in h.teacher_patience_log:
        assert 0.0 <= v <= 1.0


def test_real_run_intervention_outcomes_captured():
    """With a long-enough run, the teacher should issue at least one
    individual_intervention, producing intervention outcomes."""
    orch = _make_short_orchestrator(max_turns=250)
    bundle = run_real_bundle(orch, n_classes=1)
    h = bundle.histories[0]
    # At least one intervention (phase 4 starts ~turn 100 with default
    # phase boundaries, and 250 turns gives plenty of headroom).
    assert len(h.intervention_outcomes) >= 1
    for outcome in h.intervention_outcomes:
        assert "strategy" in outcome
        assert "pre_compliance" in outcome
        assert "post_compliance" in outcome
        assert "student_id" in outcome
        assert 0.0 <= outcome["pre_compliance"] <= 1.0
        assert 0.0 <= outcome["post_compliance"] <= 1.0


def test_real_run_first_suspicion_turns_populated():
    """With phase 2 reached, at least some students should be suspected."""
    orch = _make_short_orchestrator(max_turns=250)
    bundle = run_real_bundle(orch, n_classes=1)
    h = bundle.histories[0]
    # Phase 2 starts at turn 100 (default); 250 turns should yield
    # at least one suspicion entry.
    assert isinstance(h.first_suspicion_turns, dict)
    assert len(h.first_suspicion_turns) >= 1
    for sid, turn in h.first_suspicion_turns.items():
        assert isinstance(turn, int)
        assert turn >= 1


# ==========================================================================
# Epidemiology metrics on real runs
# ==========================================================================


def test_real_run_prevalence_metrics_compute():
    orch = _make_short_orchestrator(n_students=20, seed=42, max_turns=50)
    bundle = run_real_bundle(orch, n_classes=1)

    prev_total = resolve_metric("prevalence.adhd_total")(bundle)
    assert 0.0 <= prev_total <= 1.0

    prev_inatt = resolve_metric("prevalence.adhd_inattentive")(bundle)
    prev_hyper = resolve_metric("prevalence.adhd_hyperactive")(bundle)
    prev_comb = resolve_metric("prevalence.adhd_combined")(bundle)
    assert 0.0 <= prev_inatt <= 1.0
    assert 0.0 <= prev_hyper <= 1.0
    assert 0.0 <= prev_comb <= 1.0

    # Sum of subtypes should equal (or closely match) total ADHD,
    # since all adhd_* profiles fall into one of the three buckets.
    assert abs((prev_inatt + prev_hyper + prev_comb) - prev_total) < 1e-9


def test_real_run_male_female_ratio_metric():
    orch = _make_short_orchestrator(n_students=30, seed=3, max_turns=50)
    bundle = run_real_bundle(orch, n_classes=1)
    # Even small classes can edge-case to zero female ADHD students;
    # just verify the call succeeds and returns a finite scalar >= 0.
    ratio = resolve_metric("prevalence.adhd_male_to_female_ratio")(bundle)
    assert ratio >= 0.0


def test_real_run_on_task_behavior_metrics():
    """Behavior-frequency metrics require populated student behaviors
    in events; real runs emit these via student.step()."""
    orch = _make_short_orchestrator(max_turns=80)
    bundle = run_real_bundle(orch, n_classes=1)
    # Both metrics should compute without raising
    normal_on_task = resolve_metric("behavior.normal_on_task_percentage")(bundle)
    adhd_rel = resolve_metric("behavior.adhd_on_task_relative_to_normal")(bundle)
    assert 0.0 <= normal_on_task <= 1.0
    assert 0.0 <= adhd_rel <= 2.0  # relative ratio, usually < 1


# ==========================================================================
# Naturalness metrics on real runs
# ==========================================================================


def test_real_run_patience_ratio_metric():
    """teacher.patience_end_of_day_vs_start_ratio requires the new
    teacher_patience_log field populated by the orchestrator."""
    # 2 days = 10 periods, guaranteed to have start/end pairs
    orch = _make_short_orchestrator(max_turns=10)
    bundle = run_real_bundle(orch, n_classes=1)
    ratio = resolve_metric("teacher.patience_end_of_day_vs_start_ratio")(bundle)
    assert ratio >= 0.0


def test_real_run_empathic_compliance_gain_metric():
    """intervention.empathic_compliance_gain requires intervention_outcomes.

    Long-enough run needed so that phase 4 fires and at least one empathic
    strategy is used. Skipped if no empathic interventions occur.
    """
    orch = _make_short_orchestrator(max_turns=300)
    bundle = run_real_bundle(orch, n_classes=1)
    h = bundle.histories[0]

    empathic_count = sum(
        1 for o in h.intervention_outcomes
        if "empath" in (o.get("strategy") or "").lower()
    )
    if empathic_count == 0:
        pytest.skip("No empathic interventions in this run (rng dependent)")

    gain = resolve_metric("intervention.empathic_compliance_gain")(bundle)
    # Gain should be a finite number in [-1, 1]
    assert -1.0 <= gain <= 1.0


def test_real_run_first_suspicion_uses_real_timing():
    """Verify that the metric prefers real suspicion timing over the
    identify_adhd proxy when first_suspicion_turns is populated.
    """
    orch = _make_short_orchestrator(max_turns=250)
    bundle = run_real_bundle(orch, n_classes=1)
    h = bundle.histories[0]

    if not h.first_suspicion_turns:
        pytest.skip("Phase 2 did not produce any suspicions in this run")

    median_turn = resolve_metric("teacher.first_suspicion_turn_median")(bundle)
    # The median should be at least one of the recorded suspicion turns
    # (it literally picks from first_suspicion_turns values).
    adhd_ids = {
        s["id"] for s in h.students_roster
        if s.get("is_adhd")
    }
    adhd_suspicion_turns = [
        t for sid, t in h.first_suspicion_turns.items() if sid in adhd_ids
    ]
    if adhd_suspicion_turns:
        assert min(adhd_suspicion_turns) <= median_turn <= max(adhd_suspicion_turns)


def test_real_run_first_suspicion_falls_back_to_identify():
    """If a history lacks suspicion data, the metric should fall back to
    the identify_adhd proxy path."""
    fake_event = {
        "turn": 150,
        "students": [],
        "teacher_action": {
            "action_type": "identify_adhd",
            "student_id": "S01",
        },
    }
    roster = [
        {"id": "S01", "profile_type": "adhd_inattentive", "is_adhd": True},
    ]
    history = ClassHistory(
        students_roster=roster,
        events=[fake_event],
        first_suspicion_turns={},  # empty — force fallback
    )
    bundle = CalibrationResultBundle(histories=[history])
    median_turn = resolve_metric("teacher.first_suspicion_turn_median")(bundle)
    assert median_turn == 150.0


# ==========================================================================
# Combined loss end-to-end on a real run
# ==========================================================================


def test_combined_loss_on_real_bundle():
    """End-to-end: real simulation → bundle → loss scalar."""
    orch = _make_short_orchestrator(n_students=20, max_turns=100)
    bundle = run_real_bundle(orch, n_classes=1)

    # Minimal set of targets that real runs can satisfy without extra data
    nat_targets = []
    epi_targets = [
        EpidemiologyTarget(
            name="adhd_total_prevalence",
            metric=resolve_metric("prevalence.adhd_total"),
            target_range=(0.0, 1.0),  # accept anything — smoke test
            weight=1.0,
        ),
        EpidemiologyTarget(
            name="normal_on_task",
            metric=resolve_metric("behavior.normal_on_task_percentage"),
            target_range=(0.0, 1.0),
            weight=1.0,
        ),
    ]

    result = compute_combined_loss(bundle, nat_targets, epi_targets)
    assert result.total >= 0.0
    assert result.epidemiology_loss == 0.0  # all in-range
    assert result.summary()


def test_multi_class_real_bundle_aggregates_correctly():
    """Run 2 classes and verify the bundle contains both histories."""
    orch = _make_short_orchestrator(n_students=10, max_turns=30)
    # Remove the max_classes=1 restriction so run_real_bundle can continue
    orch.max_classes = None
    bundle = run_real_bundle(orch, n_classes=2)
    assert len(bundle) == 2
    assert all(len(h.students_roster) == 10 for h in bundle.histories)
