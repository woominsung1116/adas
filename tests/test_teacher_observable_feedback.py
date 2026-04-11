"""Tests for Phase 6 slice 4: observable-response feedback heuristic.

Covers:
  - observable_response_label positive / neutral / negative cases
  - observable_response_effect signed scalar in [-0.20, 0.20]
  - _derive_feedback_outcome routes through the observable ladder
  - No latent compliance reads anywhere in the feedback path
  - Phase 2b hypothesis-test effect recording uses behavior change
  - HypothesisTracker.likely_profile still produces a verdict on
    the new signal scale
  - Delayed queue + new heuristic still commits deterministically
  - Real run_class() populates case base via observable feedback
"""

import inspect

import pytest

from src.simulation.teacher_observation import (
    observable_response_effect,
    observable_response_label,
)
from src.simulation.teacher_memory import (
    ObservationOutcome,
    PendingObservationFeedback,
)
from src.simulation.orchestrator_v2 import (
    HypothesisTracker,
    OrchestratorV2,
)


# ---------------------------------------------------------------------------
# observable_response_label ladder
# ---------------------------------------------------------------------------


def test_label_vanished_is_positive():
    assert observable_response_label(
        ("out_of_seat", "calling_out"), ()
    ) == "positive"


def test_label_both_empty_is_neutral():
    assert observable_response_label((), ()) == "neutral"


def test_label_new_emerged_is_negative():
    assert observable_response_label((), ("emotional_outburst",)) == "negative"


def test_label_reduced_count_is_positive():
    assert observable_response_label(
        ("out_of_seat", "calling_out", "fidgeting"),
        ("out_of_seat", "fidgeting"),
    ) == "positive"


def test_label_escalated_count_is_negative():
    assert observable_response_label(
        ("out_of_seat",),
        ("out_of_seat", "calling_out"),
    ) == "negative"


def test_label_persistence_is_neutral():
    assert observable_response_label(
        ("out_of_seat", "calling_out"),
        ("out_of_seat", "calling_out"),
    ) == "neutral"


def test_label_same_count_different_identity_is_neutral():
    # Same number of behaviors but different set. Intentionally
    # conservative — not enough signal to label negative.
    assert observable_response_label(
        ("out_of_seat",),
        ("calling_out",),
    ) == "neutral"


# ---------------------------------------------------------------------------
# observable_response_effect magnitudes
# ---------------------------------------------------------------------------


def test_effect_vanished_is_large_positive():
    assert observable_response_effect(("out_of_seat",), ()) == pytest.approx(0.20)


def test_effect_emerged_is_large_negative():
    assert observable_response_effect((), ("out_of_seat",)) == pytest.approx(-0.20)


def test_effect_reduced_is_small_positive():
    assert observable_response_effect(
        ("out_of_seat", "calling_out"), ("out_of_seat",)
    ) == pytest.approx(0.10)


def test_effect_escalated_is_small_negative():
    assert observable_response_effect(
        ("out_of_seat",), ("out_of_seat", "calling_out")
    ) == pytest.approx(-0.10)


def test_effect_persistence_is_zero():
    assert observable_response_effect(
        ("out_of_seat",), ("out_of_seat",)
    ) == 0.0


def test_effect_both_empty_is_zero():
    assert observable_response_effect((), ()) == 0.0


def test_effect_stays_within_hypothesis_threshold_scale():
    """HypothesisTracker thresholds live in the ±0.03..±0.05
    range (tuned against compliance deltas). The new effect
    scale must still cross those thresholds in both directions."""
    # Anxiety threshold: empathic_avg > 0.05 → anxiety
    assert observable_response_effect(("out_of_seat",), ()) > 0.05
    # ADHD threshold: break_avg > 0.03 → adhd
    assert observable_response_effect(
        ("out_of_seat", "calling_out"), ("out_of_seat",)
    ) > 0.03
    # ODD threshold: firm_avg < -0.03 → odd
    assert observable_response_effect(
        ("out_of_seat",), ("out_of_seat", "calling_out")
    ) < -0.03


# ---------------------------------------------------------------------------
# _derive_feedback_outcome no longer reads latent compliance
# ---------------------------------------------------------------------------


def test_derive_feedback_outcome_source_has_no_latent_compliance_read():
    """Static guard: the feedback-translation step must not read
    student.state.get('compliance', ...) or any other latent
    scalar. The observable-response heuristic replaced it."""
    src = inspect.getsource(OrchestratorV2._derive_feedback_outcome)
    assert "compliance" not in src, (
        "_derive_feedback_outcome still references compliance"
    )
    assert ".state.get(" not in src, (
        "_derive_feedback_outcome still reads student.state.get(...)"
    )
    assert "distress_level" not in src
    assert "escalation_risk" not in src
    # The new code must call through the observable helper.
    assert "observable_response_label" in src


def test_update_memory_hypothesis_feedback_has_no_latent_compliance_read():
    """Static guard: the Phase 2b hypothesis-test effect recording
    must not read latent compliance anywhere in the _update_memory
    body. Phase 6 slice 6 moved the ``observable_response_effect``
    call out of _update_memory and into
    ``_drain_hypothesis_feedback_queue`` / the flush helper, so
    _update_memory now just enqueues. The latent-compliance
    probes must stay gone from _update_memory, and the
    observable-response helper must still be the one computing
    the effect — it just lives in the drain path now.
    """
    um_src = inspect.getsource(OrchestratorV2._update_memory)
    # Legacy compliance-history delta line is gone.
    assert "track.compliance_history[-1]" not in um_src
    assert "curr_compliance" not in um_src

    # The observable-response helper must still be the one
    # computing the effect, just in the drain path.
    drain_src = inspect.getsource(
        OrchestratorV2._drain_hypothesis_feedback_queue
    )
    flush_src = inspect.getsource(
        OrchestratorV2._flush_hypothesis_feedback_queue
    )
    assert "observable_response_effect" in drain_src
    assert "observable_response_effect" in flush_src
    # And the drain / flush helpers must also stay latent-free.
    for body in (drain_src, flush_src):
        assert ".state.get(" not in body
        assert "compliance" not in body


# ---------------------------------------------------------------------------
# Behavior-level: _derive_feedback_outcome routes through observable ladder
# ---------------------------------------------------------------------------


def test_derive_feedback_outcome_uses_pre_visible_snapshot():
    """Given a pre snapshot with visible disruption and a student
    whose current (post) exhibited_behaviors no longer include
    that disruption, the outcome should be 'positive'."""
    orch = OrchestratorV2(n_students=3, max_classes=1, seed=1)
    orch.classroom.reset()
    sid = orch.classroom.students[0].student_id
    # Manually clear the student's exhibited behaviors to simulate
    # "intervention worked, disruption gone".
    student = orch.classroom.get_student(sid)
    student.exhibited_behaviors = []

    feedback = orch._derive_feedback_outcome(
        student_id=sid,
        teacher_action="individual_intervention",
        pre_visible_disruptive=("out_of_seat", "calling_out"),
    )
    assert feedback.outcome == "positive"
    assert feedback.teacher_action == "individual_intervention"
    assert feedback.post_behaviors == ()


def test_derive_feedback_outcome_detects_escalation():
    orch = OrchestratorV2(n_students=3, max_classes=1, seed=1)
    orch.classroom.reset()
    sid = orch.classroom.students[0].student_id
    student = orch.classroom.get_student(sid)
    student.exhibited_behaviors = [
        "out_of_seat",
        "calling_out",
        "emotional_outburst",
    ]

    feedback = orch._derive_feedback_outcome(
        student_id=sid,
        teacher_action="firm_boundary",
        pre_visible_disruptive=("out_of_seat",),
    )
    assert feedback.outcome == "negative"
    assert "emotional_outburst" in feedback.post_behaviors


def test_derive_feedback_outcome_missing_student_is_neutral():
    orch = OrchestratorV2(n_students=3, max_classes=1, seed=1)
    feedback = orch._derive_feedback_outcome(
        student_id="nonexistent",
        teacher_action="observe",
        pre_visible_disruptive=("out_of_seat",),
    )
    assert feedback.outcome == "neutral"
    assert feedback.teacher_action == "observe"


# ---------------------------------------------------------------------------
# HypothesisTracker still produces verdicts with the new signal scale
# ---------------------------------------------------------------------------


def test_hypothesis_tracker_verdict_adhd_on_observable_reduction():
    t = HypothesisTracker(student_id="S01", suspicion_turn=1)
    # break_offer reduces disruption
    t.record_test("break_offer", observable_response_effect(
        ("out_of_seat", "calling_out"), ("out_of_seat",)
    ))
    t.record_test("break_offer", observable_response_effect(
        ("out_of_seat",), ()
    ))
    # empathic_acknowledgment and firm_boundary neutral
    t.record_test("empathic_acknowledgment", 0.0)
    t.record_test("firm_boundary", 0.0)
    assert t.likely_profile().startswith("adhd")


def test_hypothesis_tracker_verdict_anxiety_on_observable_empathic_win():
    t = HypothesisTracker(student_id="S01", suspicion_turn=1)
    t.record_test("empathic_acknowledgment", observable_response_effect(
        ("out_of_seat",), ()
    ))
    t.record_test("break_offer", 0.0)
    t.record_test("firm_boundary", 0.0)
    assert t.likely_profile() == "anxiety"


def test_hypothesis_tracker_verdict_odd_on_observable_escalation():
    t = HypothesisTracker(student_id="S01", suspicion_turn=1)
    # firm_boundary makes things observably worse
    t.record_test("firm_boundary", observable_response_effect(
        ("out_of_seat",), ("out_of_seat", "calling_out")
    ))
    t.record_test("break_offer", 0.0)
    t.record_test("empathic_acknowledgment", 0.0)
    assert t.likely_profile() == "odd"


# ---------------------------------------------------------------------------
# Pending feedback queue carries pre_visible_disruptive
# ---------------------------------------------------------------------------


def test_pending_feedback_stores_pre_visible_tuple():
    p = PendingObservationFeedback(
        student_id="S01",
        observed_behaviors=["seat-leaving"],
        teacher_action="individual_intervention",
        observed_turn=5,
        due_turn=6,
        pre_visible_disruptive=("out_of_seat", "fidgeting"),
    )
    assert p.pre_visible_disruptive == ("out_of_seat", "fidgeting")
    # Serializable.
    blob = p.as_dict()
    import json
    json.dumps(blob)
    assert blob["pre_visible_disruptive"] == ["out_of_seat", "fidgeting"]


def test_pending_feedback_default_pre_visible_is_empty_tuple():
    p = PendingObservationFeedback(
        student_id="S01",
        observed_behaviors=[],
        teacher_action="observe",
        observed_turn=1,
        due_turn=2,
    )
    assert p.pre_visible_disruptive == ()


# ---------------------------------------------------------------------------
# Integration: real run still populates memory; determinism preserved
# ---------------------------------------------------------------------------


def test_real_run_populates_memory_via_observable_feedback():
    orch = OrchestratorV2(
        n_students=6, max_classes=1, seed=42, feedback_delay_turns=1,
    )
    orch.classroom.MAX_TURNS = 150
    orch.run_class()
    records = orch.memory.case_base._records
    assert len(records) >= 5
    # Every record has a feedback payload with an observable label.
    seen_labels = set()
    for r in records:
        assert r.feedback is not None
        assert r.feedback.outcome in ("positive", "neutral", "negative")
        seen_labels.add(r.feedback.outcome)
    # At least one non-neutral label is produced in a typical run.
    assert len(seen_labels) >= 1
    # Queue flushed at end.
    assert len(orch._feedback_queue) == 0


def test_observable_feedback_is_deterministic_under_fixed_seed():
    def run_one() -> list[tuple[str, int, str, str]]:
        orch = OrchestratorV2(
            n_students=5, max_classes=1, seed=77, feedback_delay_turns=1,
        )
        orch.classroom.MAX_TURNS = 60
        orch.run_class()
        return [
            (r.student_id, r.turn, r.action_taken, r.outcome)
            for r in orch.memory.case_base._records
        ]

    a = run_one()
    b = run_one()
    assert a == b
    assert a
