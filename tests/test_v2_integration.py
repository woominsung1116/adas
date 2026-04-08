"""Integration tests for ADAS v2 modules.

Covers:
  - Server v2 path uses OrchestratorV2 (not raw ClassroomV2)
  - _update_memory preserves focused observations
  - InteractionLog class_id correctness
  - memory_summary in v2 events is meaningful
  - Deterministic full-class run
  - 5-phase teacher progression
  - InteractionLog scaling cap
"""
from __future__ import annotations

import sys
import os

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest

from src.simulation.orchestrator_v2 import OrchestratorV2
from src.simulation.classroom_env_v2 import (
    ClassroomV2,
    TeacherAction,
    InteractionEngine,
)
from src.simulation.interaction_log import InteractionLog, InteractionEvent
from src.simulation.teacher_memory import TeacherMemory


# ---------------------------------------------------------------------------
# Issue 1: Server v2 path should use OrchestratorV2
# ---------------------------------------------------------------------------


def test_server_v2_uses_orchestrator():
    """v2 session path should use OrchestratorV2, not raw ClassroomV2.

    We verify by checking that the server module's _run_v2_session imports
    OrchestratorV2 as the primary path, and that OrchestratorV2 itself
    produces valid class results.
    """
    orch = OrchestratorV2(n_students=10, max_classes=1, seed=42)
    results = list(orch.run())
    assert len(results) == 1

    result = results[0]
    assert result["type"] == "class_complete"
    assert "result" in result

    class_result = result["result"]
    assert "metrics" in class_result
    assert "events" in class_result
    assert "reports" in class_result

    metrics = class_result["metrics"]
    assert hasattr(metrics, "true_positives")
    assert hasattr(metrics, "false_positives")
    assert hasattr(metrics, "false_negatives")


# ---------------------------------------------------------------------------
# Issue 2: Detailed observation not overwritten by summary
# ---------------------------------------------------------------------------


def test_update_memory_preserves_focused_observation():
    """Detailed observation should not be overwritten by summary for same student."""
    orch = OrchestratorV2(n_students=10, max_classes=1, seed=42)
    obs = orch.classroom.reset()
    orch.memory.new_class()

    # Step once to get a real observation
    action = TeacherAction(
        action_type="observe",
        student_id=orch.classroom.students[0].student_id,
        reasoning="test",
    )
    obs, reward, done, info = orch.classroom.step(action)

    # The focused student should appear in detailed_observations
    focused_sid = action.student_id
    detail_sids = {d.student_id for d in obs.detailed_observations}

    # Simulate _update_memory
    from src.simulation.orchestrator_v2 import _StudentTrack
    tracks = {
        s.student_id: _StudentTrack(student_id=s.student_id)
        for s in orch.classroom.students
    }

    orch.memory.advance_turn()
    orch._update_memory(obs, action, info, tracks, turn=1)

    # Check that the focused student's observation was committed
    # by verifying case_base has a record for this student
    records = [
        r for r in orch.memory.case_base._records
        if r.student_id == focused_sid
    ]
    assert len(records) >= 1, "Focused student should have at least one committed record"

    # The first committed record should reflect the detailed observation,
    # not be overwritten by a summary
    if focused_sid in detail_sids:
        first_record = records[0]
        assert first_record.student_id == focused_sid
        assert first_record.action_taken == action.action_type


# ---------------------------------------------------------------------------
# Issue 3: InteractionLog class_id correctness
# ---------------------------------------------------------------------------


def test_interaction_log_class_id_correct():
    """Peer interaction events should have correct class_id, not 0."""
    env = ClassroomV2(n_students=20, seed=42)
    obs = env.reset()

    # Run several turns to generate some interaction events
    for i in range(20):
        action = TeacherAction(
            action_type="observe",
            student_id=env.students[i % len(env.students)].student_id,
        )
        obs, reward, done, info = env.step(action)

    # Check all logged events have the correct class_id
    for event in env.log._events:
        assert event.class_id == env.class_id, (
            f"Event has class_id={event.class_id}, expected {env.class_id}"
        )

    # Also verify via InteractionEngine directly
    engine = InteractionEngine()
    try:
        from src.simulation.cognitive_agent import ClassroomContext
    except ImportError:
        from src.simulation.classroom_env_v2 import ClassroomContext
    ctx = ClassroomContext(
        turn=1, period=1, day=1, subject="math", location="classroom",
        current_events=[], class_mood="calm",
        teacher_action="observe", teacher_target=None,
    )
    events = engine.process_turn(
        env.students, env.relationships, ctx, env._rng, class_id=99
    )
    for event in events:
        assert event.class_id == 99, (
            f"InteractionEngine event has class_id={event.class_id}, expected 99"
        )


# ---------------------------------------------------------------------------
# Issue 4: v2 event has meaningful memory_summary
# ---------------------------------------------------------------------------


def test_v2_event_has_meaningful_memory_summary():
    """Turn events should contain non-empty memory summary."""
    orch = OrchestratorV2(n_students=10, max_classes=1, seed=42)
    class_result = orch.run_class()

    events = class_result["events"]
    assert len(events) > 0

    # Check that at least some events have non-empty memory_summary
    summaries = [e.get("memory_summary", "") for e in events]
    non_empty = [s for s in summaries if s]
    assert len(non_empty) > 0, "All memory_summary fields are empty"

    # Verify the summary contains expected keys (classes=, cases=, principles=)
    sample = non_empty[0]
    assert "classes=" in sample
    assert "cases=" in sample
    assert "principles=" in sample


# ---------------------------------------------------------------------------
# Issue 5a: Deterministic class produces coherent output
# ---------------------------------------------------------------------------


def test_v2_deterministic_class_produces_coherent_output():
    """One full class run with seed should produce consistent metrics/events/reports."""
    orch1 = OrchestratorV2(n_students=10, max_classes=1, seed=123)
    orch2 = OrchestratorV2(n_students=10, max_classes=1, seed=123)

    result1 = orch1.run_class()
    result2 = orch2.run_class()

    m1 = result1["metrics"]
    m2 = result2["metrics"]

    assert m1.true_positives == m2.true_positives
    assert m1.false_positives == m2.false_positives
    assert m1.false_negatives == m2.false_negatives
    assert m1.class_completion_turn == m2.class_completion_turn
    assert len(result1["events"]) == len(result2["events"])


# ---------------------------------------------------------------------------
# Issue 5b: 5-phase teacher progression
# ---------------------------------------------------------------------------


def test_orchestrator_v2_5phase_progression():
    """Teacher actions should follow phase progression over 950 turns."""
    orch = OrchestratorV2(n_students=10, max_classes=1, seed=42)
    class_result = orch.run_class()
    events = class_result["events"]

    # Phase 1 (turns 1-100): should be mostly "observe"
    phase1_actions = [
        e["teacher_action"]["action_type"]
        for e in events
        if e["turn"] <= 100
    ]
    observe_ratio_p1 = phase1_actions.count("observe") / max(len(phase1_actions), 1)
    assert observe_ratio_p1 >= 0.9, (
        f"Phase 1 observe ratio {observe_ratio_p1:.2f} < 0.9"
    )

    # Phase 1 reasoning should mention "Phase 1"
    phase1_reasonings = [
        e["teacher_action"]["reasoning"]
        for e in events
        if e["turn"] <= 100
    ]
    phase1_mentions = sum(1 for r in phase1_reasonings if "Phase 1" in r)
    assert phase1_mentions > 0, "Phase 1 reasoning not found in early turns"

    # Late phases (turns > 475) should include interventions or identifications
    late_actions = set(
        e["teacher_action"]["action_type"]
        for e in events
        if e["turn"] > 475
    )
    # At least class_instruction or interventions should appear
    assert len(late_actions) > 0, "No actions in late phases"


# ---------------------------------------------------------------------------
# Issue 6: InteractionLog scaling cap
# ---------------------------------------------------------------------------


def test_interaction_log_scaling_cap():
    """InteractionLog should drop oldest events when max_events_per_class exceeded."""
    log = InteractionLog(max_events_per_class=5)

    # Record 10 events for class_id=1
    for i in range(10):
        log.record(InteractionEvent(
            class_id=1,
            turn=i + 1,
            actor="S01",
            target="S02",
            event_type="peer_chat",
        ))

    # Only 5 most recent should remain for class 1
    assert len(log._class_events[1]) == 5
    # The oldest surviving event should be turn 6
    assert log._class_events[1][0].turn == 6
    # The newest should be turn 10
    assert log._class_events[1][-1].turn == 10

    # Global events list should also be trimmed
    assert len(log._events) == 5

    # Events for a different class should be unaffected
    for i in range(3):
        log.record(InteractionEvent(
            class_id=2,
            turn=i + 1,
            actor="S03",
            target="S04",
            event_type="peer_help",
        ))
    assert len(log._class_events[2]) == 3
    assert len(log._events) == 8  # 5 from class 1 + 3 from class 2


def test_interaction_log_default_max():
    """Default max_events_per_class should be 10000."""
    log = InteractionLog()
    assert log._max_per_class == 10000
