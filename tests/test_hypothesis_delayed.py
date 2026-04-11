"""Tests for Phase 6 slice 6: delayed hypothesis-test feedback.

Covers:
  - PendingHypothesisFeedback dataclass shape + serializability
  - FeedbackDelayQueue holds both payload types
  - `_update_memory` no longer calls `tracker.record_test` directly
  - Hypothesis-test effect matures via the drain step
  - End-of-class flush applies tail-end staged effects
  - Missing tracker / missing student at drain time is a silent skip
  - Fixed-seed determinism
  - Memory-delay behavior still works (regression)
"""

import inspect
import json

import pytest

from src.simulation.teacher_memory import (
    FeedbackDelayQueue,
    PendingHypothesisFeedback,
    PendingObservationFeedback,
)
from src.simulation.orchestrator_v2 import (
    HypothesisTracker,
    OrchestratorV2,
)


# ---------------------------------------------------------------------------
# PendingHypothesisFeedback structural invariants
# ---------------------------------------------------------------------------


def test_pending_hypothesis_feedback_fields_are_observable_only():
    p = PendingHypothesisFeedback(
        student_id="S01",
        strategy="break_offer",
        observed_turn=10,
        due_turn=12,
        pre_visible_disruptive=("out_of_seat",),
    )
    assert p.student_id == "S01"
    assert p.strategy == "break_offer"
    assert p.observed_turn == 10
    assert p.due_turn == 12
    fields = set(PendingHypothesisFeedback.__dataclass_fields__.keys())
    for latent in (
        "compliance",
        "distress_level",
        "attention",
        "escalation_risk",
        "state_snapshot",
    ):
        assert latent not in fields


def test_pending_hypothesis_feedback_is_json_serializable():
    p = PendingHypothesisFeedback(
        student_id="S01",
        strategy="empathic_acknowledgment",
        observed_turn=3,
        due_turn=4,
        pre_visible_disruptive=("fidgeting", "calling_out"),
    )
    blob = p.as_dict()
    json.dumps(blob)
    assert blob["strategy"] == "empathic_acknowledgment"
    assert blob["pre_visible_disruptive"] == ["fidgeting", "calling_out"]


def test_feedback_queue_holds_both_payload_types():
    q = FeedbackDelayQueue()
    q.enqueue(PendingObservationFeedback(
        student_id="A", observed_behaviors=[],
        teacher_action="observe", observed_turn=1, due_turn=2,
    ))
    q.enqueue(PendingHypothesisFeedback(
        student_id="A", strategy="break_offer",
        observed_turn=1, due_turn=3,
    ))
    due = q.pop_due(current_turn=2)
    assert len(due) == 1
    assert isinstance(due[0], PendingObservationFeedback)
    # The hypothesis item is still pending.
    due2 = q.pop_due(current_turn=3)
    assert len(due2) == 1
    assert isinstance(due2[0], PendingHypothesisFeedback)


# ---------------------------------------------------------------------------
# _update_memory no longer calls record_test directly
# ---------------------------------------------------------------------------


def test_update_memory_does_not_call_record_test_directly():
    src = inspect.getsource(OrchestratorV2._update_memory)
    # The legacy inline call is gone.
    assert "tracker.record_test(" not in src
    assert "record_test(action.strategy" not in src
    # The enqueue path must be present.
    assert "PendingHypothesisFeedback" in src
    assert "_hypothesis_feedback_queue.enqueue" in src


def test_update_memory_does_not_update_tracker_synchronously():
    """Drive one orchestrator turn manually with a hypothesis-test
    strategy and assert the tracker's tests_applied dict is still
    empty immediately after _update_memory — the effect must only
    land later, after the drain step runs."""
    orch = OrchestratorV2(n_students=3, max_classes=1, seed=11)
    obs = orch.classroom.reset()
    orch.memory.new_class()
    sid = orch.classroom.students[0].student_id

    # Create a hypothesis tracker manually and drop a test action.
    orch._hypothesis_trackers[sid] = HypothesisTracker(
        student_id=sid, suspicion_turn=1,
    )
    from src.simulation.classroom_env_v2 import TeacherAction
    action = TeacherAction(
        action_type="individual_intervention",
        student_id=sid,
        strategy="break_offer",
        reasoning="test",
    )

    # Step once so the environment advances.
    obs, reward, done, info = orch.classroom.step(action)

    # Build pre_step_visible for the student (empty — we don't
    # actually care about the value, just that the enqueue runs).
    pre_step_visible = {s.student_id: () for s in orch.classroom.students}
    from src.simulation.orchestrator_v2 import _StudentTrack
    tracks = {
        s.student_id: _StudentTrack(student_id=s.student_id)
        for s in orch.classroom.students
    }
    orch.memory.advance_turn()
    orch._update_memory(
        obs, action, info, tracks, turn=1,
        pre_step_visible=pre_step_visible,
    )

    # tracker.tests_applied must still be empty — the effect is queued.
    assert orch._hypothesis_trackers[sid].tests_applied == {}
    # The hypothesis queue must have one item.
    assert len(orch._hypothesis_feedback_queue) == 1

    # Flush it and confirm the tracker now sees the effect.
    orch._flush_hypothesis_feedback_queue()
    assert "break_offer" in orch._hypothesis_trackers[sid].tests_applied


# ---------------------------------------------------------------------------
# Delayed drain applies the effect on the matching turn
# ---------------------------------------------------------------------------


def test_drain_applies_effect_at_due_turn():
    orch = OrchestratorV2(n_students=3, max_classes=1, seed=1)
    orch.classroom.reset()
    sid = orch.classroom.students[0].student_id
    orch._hypothesis_trackers[sid] = HypothesisTracker(
        student_id=sid, suspicion_turn=1,
    )
    orch._hypothesis_feedback_queue.enqueue(
        PendingHypothesisFeedback(
            student_id=sid,
            strategy="break_offer",
            observed_turn=5,
            due_turn=6,
            pre_visible_disruptive=("out_of_seat",),
        )
    )

    # Draining before due_turn is a no-op.
    applied = orch._drain_hypothesis_feedback_queue(current_turn=5)
    assert applied == 0
    assert len(orch._hypothesis_feedback_queue) == 1
    assert orch._hypothesis_trackers[sid].tests_applied == {}

    # Draining at due_turn applies the effect.
    applied = orch._drain_hypothesis_feedback_queue(current_turn=6)
    assert applied == 1
    assert len(orch._hypothesis_feedback_queue) == 0
    assert "break_offer" in orch._hypothesis_trackers[sid].tests_applied


# ---------------------------------------------------------------------------
# Missing tracker / student at drain time is a silent skip
# ---------------------------------------------------------------------------


def test_drain_skips_missing_tracker():
    orch = OrchestratorV2(n_students=3, max_classes=1, seed=1)
    orch.classroom.reset()
    sid = orch.classroom.students[0].student_id
    orch._hypothesis_feedback_queue.enqueue(
        PendingHypothesisFeedback(
            student_id=sid,
            strategy="break_offer",
            observed_turn=1,
            due_turn=2,
            pre_visible_disruptive=(),
        )
    )
    # No tracker for this sid — drain should silently skip.
    applied = orch._drain_hypothesis_feedback_queue(current_turn=2)
    assert applied == 0
    assert len(orch._hypothesis_feedback_queue) == 0  # item was popped


def test_drain_skips_missing_student():
    orch = OrchestratorV2(n_students=3, max_classes=1, seed=1)
    orch.classroom.reset()
    # Tracker exists but student id is bogus.
    orch._hypothesis_trackers["ghost"] = HypothesisTracker(
        student_id="ghost", suspicion_turn=0,
    )
    orch._hypothesis_feedback_queue.enqueue(
        PendingHypothesisFeedback(
            student_id="ghost",
            strategy="break_offer",
            observed_turn=1,
            due_turn=2,
            pre_visible_disruptive=(),
        )
    )
    applied = orch._drain_hypothesis_feedback_queue(current_turn=2)
    assert applied == 0
    assert orch._hypothesis_trackers["ghost"].tests_applied == {}


# ---------------------------------------------------------------------------
# End-of-class flush
# ---------------------------------------------------------------------------


def test_flush_applies_tail_end_hypothesis_effects():
    orch = OrchestratorV2(n_students=3, max_classes=1, seed=1)
    orch.classroom.reset()
    sid = orch.classroom.students[0].student_id
    orch._hypothesis_trackers[sid] = HypothesisTracker(
        student_id=sid, suspicion_turn=0,
    )
    # Stage an item with a far-future due_turn.
    orch._hypothesis_feedback_queue.enqueue(
        PendingHypothesisFeedback(
            student_id=sid,
            strategy="break_offer",
            observed_turn=10,
            due_turn=1000,
            pre_visible_disruptive=("out_of_seat",),
        )
    )
    applied = orch._flush_hypothesis_feedback_queue()
    assert applied == 1
    assert len(orch._hypothesis_feedback_queue) == 0
    assert "break_offer" in orch._hypothesis_trackers[sid].tests_applied


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_run_class_hypothesis_feedback_is_deterministic():
    def run_one() -> list:
        orch = OrchestratorV2(n_students=5, max_classes=1, seed=42)
        orch.classroom.MAX_TURNS = 400
        orch.run_class()
        return [
            (sid, dict(t.tests_applied), t.diagnosis_hypothesis)
            for sid, t in sorted(orch._hypothesis_trackers.items())
        ]

    a = run_one()
    b = run_one()
    assert a == b


def test_run_class_leaves_hypothesis_queue_empty():
    orch = OrchestratorV2(n_students=5, max_classes=1, seed=42)
    orch.classroom.MAX_TURNS = 200
    orch.run_class()
    assert len(orch._hypothesis_feedback_queue) == 0


def test_memory_delay_behavior_still_works():
    """Regression: slice 3 memory-delay behavior must still hold
    after slice 6 adds a parallel hypothesis queue."""
    orch = OrchestratorV2(
        n_students=5, max_classes=1, seed=42, feedback_delay_turns=1,
    )
    orch.classroom.MAX_TURNS = 150
    orch.run_class()
    # Memory queue drained by end of class.
    assert len(orch._feedback_queue) == 0
    # Case base populated.
    assert len(orch.memory.case_base._records) >= 1


def test_per_class_hypothesis_queue_reset():
    orch = OrchestratorV2(n_students=3, max_classes=2, seed=1)
    orch.classroom.MAX_TURNS = 40
    orch.run_class()
    assert len(orch._hypothesis_feedback_queue) == 0
    orch.run_class()
    assert len(orch._hypothesis_feedback_queue) == 0
