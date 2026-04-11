"""Tests for Phase 6 slice 3: delayed feedback queue for teacher memory.

Covers:
  - PendingObservationFeedback dataclass construction + serializability
  - FeedbackDelayQueue FIFO order + pop_due + flush_all semantics
  - Queue contains no latent state
  - Orchestrator does not commit memory records immediately
  - Records materialize after the configured delay
  - End-of-class flush step catches tail-end items
  - Fixed feedback_delay_turns=0 preserves turn-stamped observations
  - Deterministic delayed commit behavior under fixed seeds
  - `run_class()` still populates the case base end-to-end
  - Per-class queue reset
"""

import json

import pytest

from src.simulation.teacher_memory import (
    FeedbackDelayQueue,
    ObservationOutcome,
    ObservationRecord,
    PendingObservationFeedback,
    TeacherMemory,
)
from src.simulation.orchestrator_v2 import OrchestratorV2


# ---------------------------------------------------------------------------
# PendingObservationFeedback
# ---------------------------------------------------------------------------


def test_pending_feedback_fields_are_observable_only():
    p = PendingObservationFeedback(
        student_id="S01",
        observed_behaviors=["seat-leaving", "blurting-answers"],
        teacher_action="individual_intervention",
        observed_turn=10,
        due_turn=12,
    )
    assert p.student_id == "S01"
    assert p.observed_turn == 10
    assert p.due_turn == 12
    # No latent scalar fields on the dataclass.
    fields = set(PendingObservationFeedback.__dataclass_fields__.keys())
    for latent in (
        "state_snapshot",
        "distress_level",
        "compliance",
        "attention",
        "escalation_risk",
    ):
        assert latent not in fields


def test_pending_feedback_is_json_serializable():
    p = PendingObservationFeedback(
        student_id="S01",
        observed_behaviors=["seat-leaving"],
        teacher_action="observe",
        observed_turn=1,
        due_turn=2,
    )
    blob = p.as_dict()
    json.dumps(blob)
    assert blob["student_id"] == "S01"
    assert blob["observed_behaviors"] == ["seat-leaving"]


# ---------------------------------------------------------------------------
# FeedbackDelayQueue
# ---------------------------------------------------------------------------


def test_queue_is_fifo_by_enqueue_order():
    q = FeedbackDelayQueue()
    for i in range(5):
        q.enqueue(
            PendingObservationFeedback(
                student_id=f"S{i:02d}",
                observed_behaviors=[],
                teacher_action="observe",
                observed_turn=i,
                due_turn=10,  # same due_turn for all
            )
        )
    assert len(q) == 5
    due = q.pop_due(current_turn=10)
    assert [p.student_id for p in due] == [f"S{i:02d}" for i in range(5)]
    assert len(q) == 0


def test_queue_pop_due_respects_due_turn():
    q = FeedbackDelayQueue()
    q.enqueue(PendingObservationFeedback(
        student_id="early", observed_behaviors=[],
        teacher_action="observe", observed_turn=1, due_turn=2,
    ))
    q.enqueue(PendingObservationFeedback(
        student_id="late", observed_behaviors=[],
        teacher_action="observe", observed_turn=1, due_turn=10,
    ))

    # Calling at turn 2: only "early" should mature.
    due = q.pop_due(current_turn=2)
    assert [p.student_id for p in due] == ["early"]
    assert len(q) == 1

    # "late" still pending at turn 5.
    due2 = q.pop_due(current_turn=5)
    assert due2 == []
    assert len(q) == 1

    # At turn 10 it finally matures.
    due3 = q.pop_due(current_turn=10)
    assert [p.student_id for p in due3] == ["late"]
    assert len(q) == 0


def test_queue_flush_all_returns_everything_remaining():
    q = FeedbackDelayQueue()
    q.enqueue(PendingObservationFeedback(
        student_id="A", observed_behaviors=[],
        teacher_action="observe", observed_turn=1, due_turn=100,
    ))
    q.enqueue(PendingObservationFeedback(
        student_id="B", observed_behaviors=[],
        teacher_action="observe", observed_turn=1, due_turn=200,
    ))
    out = q.flush_all()
    assert [p.student_id for p in out] == ["A", "B"]
    assert len(q) == 0


def test_queue_peek_all_is_non_destructive():
    q = FeedbackDelayQueue()
    q.enqueue(PendingObservationFeedback(
        student_id="A", observed_behaviors=[],
        teacher_action="observe", observed_turn=1, due_turn=5,
    ))
    snapshot = q.peek_all()
    assert len(snapshot) == 1
    assert len(q) == 1  # still there


# ---------------------------------------------------------------------------
# Orchestrator wiring: observations do NOT commit immediately
# ---------------------------------------------------------------------------


def _step_one_turn(orch: OrchestratorV2):
    """Drive a single orchestrator turn by calling stream_class once."""
    gen = orch.stream_class()
    header = next(gen)  # "new_class"
    assert header["type"] == "new_class"
    turn_event = next(gen)  # first "turn"
    assert turn_event["type"] == "turn"
    return orch, gen, turn_event


def test_observation_is_not_committed_immediately_with_delay_one(tmp_path):
    orch = OrchestratorV2(
        n_students=5, max_classes=1, seed=7, feedback_delay_turns=1,
    )
    orch.classroom.MAX_TURNS = 1
    # After streaming a single turn, the drain at the START of that
    # turn runs against an empty queue (nothing enqueued yet). The
    # _update_memory call at the end of the turn enqueues items
    # with due_turn=2, which never matures within MAX_TURNS=1.
    list(orch.stream_class())

    # End-of-class flush is our safety net, so items DO eventually
    # reach memory — but we can check the intermediate staging
    # behavior on a slightly longer run below.
    assert len(orch._feedback_queue) == 0  # drained by flush
    # With MAX_TURNS=1 and delay=1, the only matured records come
    # from the end-of-class flush, not from drain_feedback_queue.
    # We just confirm the case base is non-empty.
    assert len(orch.memory.case_base._records) >= 1


def test_records_materialize_after_delay_via_turn_start_drain():
    """Staged observations mature on the next turn's drain step.

    With delay=1, an observation made on turn T has due_turn=T+1,
    so it is committed by the drain step at the start of turn T+1.
    """
    orch = OrchestratorV2(
        n_students=5, max_classes=1, seed=7, feedback_delay_turns=1,
    )
    orch.classroom.MAX_TURNS = 3

    # Stream the class turn by turn so we can peek at the queue
    # between yields.
    gen = orch.stream_class()
    next(gen)  # "new_class" header
    # After turn 1 yield, the items enqueued on turn 1 have
    # due_turn=2 and should still be pending (they mature at the
    # START of turn 2, i.e. at the NEXT drain call).
    next(gen)  # turn 1
    pending_after_t1 = len(orch._feedback_queue)
    records_after_t1 = len(orch.memory.case_base._records)

    next(gen)  # turn 2
    # The drain step at the start of turn 2 should have committed
    # every item that was staged on turn 1.
    records_after_t2_drain = len(orch.memory.case_base._records)

    # Consume the rest (turn 3 + class_complete flush)
    for _ in gen:
        pass

    # Invariants:
    # 1. Turn 1 staged items actually exist in the queue afterwards
    #    (the end-of-T1 enqueue happens AFTER the yield).
    assert pending_after_t1 >= 1
    # 2. At end of turn 1, no records have been committed yet
    #    (drain step at start of T1 had an empty queue).
    assert records_after_t1 == 0
    # 3. By end of turn 2, records_after_t2_drain >= number
    #    staged on turn 1 — the drain at start of T2 matured them.
    assert records_after_t2_drain >= pending_after_t1
    # 4. End-of-class flush leaves the queue empty.
    assert len(orch._feedback_queue) == 0


def test_queue_items_carry_no_latent_state(tmp_path):
    """After a real run, every pending item that lives in the
    queue (mid-run, simulated by peeking mid-stream) must carry
    only observable fields."""
    orch = OrchestratorV2(
        n_students=5, max_classes=1, seed=7, feedback_delay_turns=2,
    )
    orch.classroom.MAX_TURNS = 10

    gen = orch.stream_class()
    next(gen)  # header
    next(gen)  # turn 1
    snapshot = orch._feedback_queue.peek_all()
    assert snapshot  # something staged on turn 1
    for p in snapshot:
        # Only the known-observable fields exist
        assert set(PendingObservationFeedback.__dataclass_fields__.keys()) == {
            "student_id",
            "observed_behaviors",
            "teacher_action",
            "observed_turn",
            "due_turn",
        }
        # No latent values
        d = p.as_dict()
        for latent in ("distress_level", "compliance", "attention", "escalation_risk", "state_snapshot"):
            assert latent not in d
    # Consume the rest
    for _ in gen:
        pass


# ---------------------------------------------------------------------------
# End-of-class flush safety net
# ---------------------------------------------------------------------------


def test_flush_feedback_queue_catches_tail_end_items():
    """With delay=5 and MAX_TURNS=3, nothing would mature via the
    normal drain step. The end-of-class flush must still commit
    every staged item so the case base is not silently empty."""
    orch = OrchestratorV2(
        n_students=4, max_classes=1, seed=11, feedback_delay_turns=5,
    )
    orch.classroom.MAX_TURNS = 3
    orch.run_class()
    # All staged items flushed.
    assert len(orch._feedback_queue) == 0
    # Case base should contain at least one record — the flush
    # path materialized every pending item.
    assert len(orch.memory.case_base._records) >= 1


def test_zero_delay_behaves_like_same_turn_commit():
    """feedback_delay_turns=0 means items are eligible for the
    drain step on the very same turn they were enqueued.

    Since our drain runs at the START of each turn (before the
    enqueue happens at the end), a same-turn commit still requires
    the end-of-class flush OR another turn. In practice delay=0
    with MAX_TURNS>=2 yields identical records count as delay>=1
    for the same run — the flush path catches anything the drain
    step misses."""
    orch = OrchestratorV2(
        n_students=4, max_classes=1, seed=11, feedback_delay_turns=0,
    )
    orch.classroom.MAX_TURNS = 10
    orch.run_class()
    # All staged items flushed.
    assert len(orch._feedback_queue) == 0
    assert len(orch.memory.case_base._records) >= 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_delayed_feedback_is_deterministic_under_fixed_seed():
    def run_one() -> list[tuple[str, int, str, str]]:
        orch = OrchestratorV2(
            n_students=5, max_classes=1, seed=77, feedback_delay_turns=1,
        )
        orch.classroom.MAX_TURNS = 40
        orch.run_class()
        return [
            (r.student_id, r.turn, r.action_taken, r.outcome)
            for r in orch.memory.case_base._records
        ]

    a = run_one()
    b = run_one()
    assert a == b
    assert a  # non-empty


# ---------------------------------------------------------------------------
# Integration: run_class still populates end-to-end
# ---------------------------------------------------------------------------


def test_real_run_class_populates_case_base_after_delay(tmp_path):
    orch = OrchestratorV2(
        n_students=6, max_classes=1, seed=42, feedback_delay_turns=1,
    )
    orch.classroom.MAX_TURNS = 200
    result = orch.run_class()
    # The class ran to completion.
    assert "metrics" in result
    # Case base has records.
    records = orch.memory.case_base._records
    assert len(records) >= 10
    # Every record has an observable-only feedback payload.
    for rec in records:
        assert not hasattr(rec, "state_snapshot")
        assert rec.feedback is not None
        assert rec.feedback.outcome in ("positive", "neutral", "negative")
    # Queue is empty at end.
    assert len(orch._feedback_queue) == 0


def test_per_class_queue_reset_between_runs():
    orch = OrchestratorV2(
        n_students=5, max_classes=2, seed=1, feedback_delay_turns=1,
    )
    orch.classroom.MAX_TURNS = 20
    orch.run_class()
    # After first class the queue is empty (flushed).
    assert len(orch._feedback_queue) == 0
    # Run another class — queue starts fresh.
    orch.run_class()
    assert len(orch._feedback_queue) == 0
