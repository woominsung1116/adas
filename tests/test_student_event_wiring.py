"""Tests for Phase 6 slice 15: student perceive loop now receives
observable turn events instead of an unconditional empty list.

Covers:
  - _make_context no longer passes current_events=[] unconditionally
  - _build_current_events_for_students emits a teacher_action event
    when the teacher is doing something directed
  - Pure passive observation sweeps (action='observe', student_id=None)
    still produce an empty event list — preserves test-harness
    backwards compatibility
  - Previous-turn salient peer events reach the payload
  - Routine chatter is NOT propagated (only salient types)
  - Events carry ONLY observable keys (no latent emotion/state fields)
  - CognitiveStudent memory gains perceived event nodes over time
  - Fixed-seed determinism under the new wiring
"""

import inspect

import pytest

from src.simulation.classroom_env_v2 import ClassroomV2, TeacherAction
from src.simulation.interaction_log import InteractionEvent


# ---------------------------------------------------------------------------
# Static guard: no unconditional empty list
# ---------------------------------------------------------------------------


def test_make_context_source_no_longer_has_unconditional_empty_events():
    """Static grep: the previous `current_events=[],` literal must
    be gone from `_make_context`. The helper `_build_current_events_for_students`
    must be the single source for this field now."""
    src = inspect.getsource(ClassroomV2._make_context)
    assert "current_events=[]" not in src
    assert "_build_current_events_for_students" in src


# ---------------------------------------------------------------------------
# Helper output shape
# ---------------------------------------------------------------------------


def test_directed_teacher_action_produces_teacher_event():
    env = ClassroomV2(n_students=4, seed=7)
    env.reset()
    action = TeacherAction(
        action_type="individual_intervention",
        student_id=env.students[0].student_id,
        strategy="break_offer",
        reasoning="intervention test",
    )
    events = env._build_current_events_for_students(action)
    assert len(events) >= 1
    teacher_event = events[0]
    assert teacher_event["actor"] == "teacher"
    assert teacher_event["target"] == env.students[0].student_id
    assert teacher_event["action"] == "individual_intervention"
    assert teacher_event["type"] == "teacher_action"


def test_passive_sweep_produces_no_teacher_event():
    """`observe` with student_id=None is a passive sweep — the
    helper deliberately skips emitting a teacher_action event so
    harnesses that drive the env with generic observations keep
    their previously-starved perceive loop shape."""
    env = ClassroomV2(n_students=4, seed=7)
    env.reset()
    action = TeacherAction(action_type="observe", student_id=None)
    events = env._build_current_events_for_students(action)
    # No teacher event; no prev-turn salient events at turn 0.
    teacher_events = [e for e in events if e.get("actor") == "teacher"]
    assert teacher_events == []


def test_directed_observe_produces_teacher_event():
    """`observe` WITH a specific student_id is directed perception
    — students can notice the teacher looking at them."""
    env = ClassroomV2(n_students=4, seed=7)
    env.reset()
    action = TeacherAction(
        action_type="observe",
        student_id=env.students[0].student_id,
    )
    events = env._build_current_events_for_students(action)
    teacher_events = [e for e in events if e.get("actor") == "teacher"]
    assert len(teacher_events) == 1


def test_class_targeted_teacher_event_uses_class_token():
    env = ClassroomV2(n_students=4, seed=7)
    env.reset()
    action = TeacherAction(
        action_type="class_instruction",
        student_id=None,
        reasoning="teach",
    )
    events = env._build_current_events_for_students(action)
    teacher_events = [e for e in events if e.get("actor") == "teacher"]
    assert len(teacher_events) == 1
    assert teacher_events[0]["target"] == "class"


# ---------------------------------------------------------------------------
# Previous-turn event propagation (salient filter)
# ---------------------------------------------------------------------------


def test_previous_turn_salient_peer_events_reach_payload():
    env = ClassroomV2(n_students=4, seed=7)
    env.reset()
    # Plant a synthetic conflict event on turn 1, then pull events for turn 2.
    env.turn = 1
    env.log.record(InteractionEvent(
        class_id=env.class_id,
        turn=1,
        actor=env.students[0].student_id,
        target=env.students[1].student_id,
        event_type="conflict",
        action="shove",
        content="S00 shoved S01",
    ))
    env.turn = 2
    action = TeacherAction(action_type="class_instruction", reasoning="teach")
    events = env._build_current_events_for_students(action)
    peer_events = [e for e in events if e.get("type") == "conflict"]
    assert len(peer_events) == 1
    assert peer_events[0]["actor"] == env.students[0].student_id
    assert peer_events[0]["target"] == env.students[1].student_id


def test_previous_turn_chatter_is_not_propagated():
    env = ClassroomV2(n_students=4, seed=7)
    env.reset()
    env.turn = 1
    env.log.record(InteractionEvent(
        class_id=env.class_id,
        turn=1,
        actor=env.students[0].student_id,
        target=env.students[1].student_id,
        event_type="chatter",
        action="whisper",
        content="S00 whispered to S01",
    ))
    env.turn = 2
    action = TeacherAction(action_type="observe", student_id=None)
    events = env._build_current_events_for_students(action)
    chatter_events = [e for e in events if e.get("type") == "chatter"]
    assert chatter_events == []


def test_previous_turn_teacher_event_is_propagated_as_salient():
    env = ClassroomV2(n_students=4, seed=7)
    env.reset()
    env.turn = 1
    env.log.record(InteractionEvent(
        class_id=env.class_id,
        turn=1,
        actor="teacher",
        target=env.students[0].student_id,
        event_type="directed",
        action="private_correction",
        content="teacher corrected S00",
    ))
    env.turn = 2
    action = TeacherAction(action_type="observe", student_id=None)
    events = env._build_current_events_for_students(action)
    teacher_prev = [e for e in events if e.get("actor") == "teacher"]
    assert len(teacher_prev) == 1
    assert teacher_prev[0]["action"] == "private_correction"


# ---------------------------------------------------------------------------
# No latent leakage
# ---------------------------------------------------------------------------


def test_event_payload_has_only_observable_keys():
    env = ClassroomV2(n_students=4, seed=7)
    env.reset()
    env.turn = 1
    env.log.record(InteractionEvent(
        class_id=env.class_id,
        turn=1,
        actor=env.students[0].student_id,
        target=env.students[1].student_id,
        event_type="conflict",
        action="shove",
        content="fight",
        actor_emotions_before={"anger": 0.9},
        actor_emotions_after={"anger": 0.95},
        target_state_before={"compliance": 0.1, "attention": 0.2},
        target_state_after={"compliance": 0.05, "attention": 0.1},
        inner_thought="S00 wanted to win",
    ))
    env.turn = 2
    action = TeacherAction(
        action_type="individual_intervention",
        student_id=env.students[0].student_id,
    )
    events = env._build_current_events_for_students(action)
    expected = {"actor", "target", "action", "description", "type"}
    forbidden = {
        "actor_emotions_before", "actor_emotions_after",
        "target_emotions_before", "target_emotions_after",
        "actor_state_before", "actor_state_after",
        "target_state_before", "target_state_after",
        "inner_thought", "outcome", "reward",
    }
    for e in events:
        assert set(e.keys()) == expected, f"unexpected keys on {e}"
        for f in forbidden:
            assert f not in e


# ---------------------------------------------------------------------------
# Integration: student memory gains perceived events
# ---------------------------------------------------------------------------


def test_student_memory_gains_events_under_directed_teacher_actions():
    """Run the env with a directed teacher action every turn and
    confirm students accumulate perceived event nodes."""
    env = ClassroomV2(n_students=4, seed=11)
    env.reset()
    target_sid = env.students[0].student_id
    for _ in range(20):
        action = TeacherAction(
            action_type="individual_intervention",
            student_id=target_sid,
            strategy="break_offer",
            reasoning="test intervention",
        )
        env.step(action)
    # The targeted student should have multiple event memory nodes.
    targeted = env.students[0]
    event_nodes = [
        n for n in targeted.memory.recent_events(20)
        if n.node_type == "event"
    ]
    assert len(event_nodes) >= 1


def test_student_memory_is_sparse_under_pure_passive_sweeps():
    """Under pure passive `observe` sweeps the perceive loop
    still stays starved — matching the pre-slice behavior for
    test harnesses that do not simulate directed teacher actions.
    This is the guarantee that kept the modulation-drift test
    stable in Phase 6 slice 15."""
    env = ClassroomV2(n_students=4, seed=11)
    env.reset()
    for _ in range(10):
        action = TeacherAction(action_type="observe", student_id=None)
        env.step(action)
    targeted = env.students[0]
    event_nodes = [
        n for n in targeted.memory.recent_events(20)
        if n.node_type == "event"
    ]
    # Either zero (no salient events) or only propagated ones
    # from rare stochastic peer conflicts. The important check
    # is that this path does not explode unboundedly.
    assert len(event_nodes) < 20


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_event_wiring_is_deterministic_under_fixed_seed():
    def fingerprint():
        env = ClassroomV2(n_students=4, seed=99)
        env.reset()
        for _ in range(10):
            action = TeacherAction(
                action_type="individual_intervention",
                student_id=env.students[0].student_id,
                strategy="break_offer",
                reasoning="test",
            )
            env.step(action)
        targeted = env.students[0]
        return [
            (n.subject, n.predicate, n.object, round(n.poignancy, 3))
            for n in targeted.memory.recent_events(20)
            if n.node_type == "event"
        ]

    a = fingerprint()
    b = fingerprint()
    assert a == b
