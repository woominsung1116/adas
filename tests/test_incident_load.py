"""Tests for Phase 6 slice 8: behavior-derived teacher incident load.

Covers:
  - derive_incident_load reads only observable visible_behaviors
  - empty / non-disruptive / single / multiple student cases
  - TeacherObservationBatch exposes .incident_load populated by the builder
  - Teacher emotion update path reads teacher_batch.incident_load,
    NOT `len(info["interactions"])`
  - Fixed-seed determinism
"""

import inspect

import pytest

from src.simulation.classroom_env_v2 import (
    ClassroomObservation,
    StudentSummary,
)
from src.simulation.teacher_observation import (
    StudentObservation,
    TeacherObservationBatch,
    build_observations_from_classroom,
    derive_incident_load,
)
from src.simulation.orchestrator_v2 import OrchestratorV2


def _obs(sid: str, *behaviors: str) -> StudentObservation:
    return StudentObservation(
        student_id=sid,
        turn=1,
        visible_behaviors=tuple(behaviors),
        profile_hint="unknown",
        seat_row=0,
        seat_col=0,
        is_identified=False,
        is_managed=False,
    )


# ---------------------------------------------------------------------------
# derive_incident_load ladder
# ---------------------------------------------------------------------------


def test_empty_observations_load_is_zero():
    assert derive_incident_load([]) == 0


def test_all_calm_students_load_is_zero():
    assert derive_incident_load([_obs(f"S{i}") for i in range(5)]) == 0


def test_single_disruptive_student_load_is_one():
    batch = [_obs("S01", "out_of_seat"), _obs("S02"), _obs("S03")]
    assert derive_incident_load(batch) == 1


def test_multiple_disruptive_students_load_equals_count():
    batch = [
        _obs("S01", "out_of_seat"),
        _obs("S02", "calling_out", "interrupting"),
        _obs("S03", "fidgeting"),
        _obs("S04"),
    ]
    assert derive_incident_load(batch) == 3


def test_non_disruptive_vocabulary_does_not_count():
    batch = [_obs("S01", "on_task"), _obs("S02", "listening")]
    assert derive_incident_load(batch) == 0


def test_one_disruptive_in_mixed_behaviors_still_counts_once():
    """A student with one disruptive + one non-disruptive behavior
    counts as a single incident, not two."""
    batch = [_obs("S01", "out_of_seat", "on_task")]
    assert derive_incident_load(batch) == 1


def test_derive_incident_load_source_has_no_latent_reads():
    src = inspect.getsource(derive_incident_load)
    for latent in (
        "distress_level",
        "compliance",
        "attention",
        "escalation_risk",
        "state_snapshot",
        ".state.get(",
    ):
        assert latent not in src


def test_derive_incident_load_reads_no_classroom_interactions():
    """Structural: the helper must not touch
    classroom-side interaction counts in any form."""
    src = inspect.getsource(derive_incident_load)
    assert "interactions" not in src
    assert "info[" not in src


# ---------------------------------------------------------------------------
# Builder populates incident_load
# ---------------------------------------------------------------------------


def _classroom_obs_with(summaries):
    return ClassroomObservation(
        turn=1, day=1, period=1,
        subject="korean", location="classroom",
        student_summaries=summaries,
        detailed_observations=[],
        class_mood="neutral",
        identified_adhd_ids=[],
        managed_ids=[],
    )


def test_builder_populates_incident_load_zero():
    summaries = [
        StudentSummary(
            student_id=f"S{i:02d}", profile_hint="typical",
            behaviors=[], is_identified=False, is_managed=False,
            seat_row=0, seat_col=0,
        )
        for i in range(5)
    ]
    batch = build_observations_from_classroom(_classroom_obs_with(summaries))
    assert batch.incident_load == 0


def test_builder_populates_incident_load_matches_disruptive_count():
    summaries = [
        StudentSummary(
            student_id="S01", profile_hint="disruptive",
            behaviors=["out_of_seat"], is_identified=False, is_managed=False,
            seat_row=0, seat_col=0,
        ),
        StudentSummary(
            student_id="S02", profile_hint="disruptive",
            behaviors=["calling_out", "fidgeting"],
            is_identified=False, is_managed=False,
            seat_row=0, seat_col=0,
        ),
        StudentSummary(
            student_id="S03", profile_hint="typical",
            behaviors=[], is_identified=False, is_managed=False,
            seat_row=0, seat_col=0,
        ),
    ]
    batch = build_observations_from_classroom(_classroom_obs_with(summaries))
    assert batch.incident_load == 2


def test_incident_load_default_is_zero_on_batch_dataclass():
    batch = TeacherObservationBatch(
        turn=1, class_mood="neutral", observations=(),
    )
    assert batch.incident_load == 0


# ---------------------------------------------------------------------------
# Teacher emotion update path wiring
# ---------------------------------------------------------------------------


def test_stream_class_feeds_emotions_from_batch_incident_load():
    """Static guard: the emotion update call must pass
    ``teacher_batch.incident_load`` and must NOT look at
    ``len(info["interactions"])`` or ``info.get("interactions", ...)``
    as the incident argument."""
    src = inspect.getsource(OrchestratorV2.stream_class)
    # Old classroom-side count path is gone from the emotion call.
    assert "len(info.get(\"interactions\"" not in src
    assert 'n_incidents = len(info' not in src
    # New teacher-visible path is wired.
    assert "teacher_batch.incident_load" in src


def test_orchestrator_runs_without_crash_with_new_incident_input():
    """Behavior test: run a class and verify it completes
    cleanly using the new incident-load signal."""
    orch = OrchestratorV2(n_students=5, max_classes=1, seed=7)
    orch.classroom.MAX_TURNS = 50
    result = orch.run_class()
    assert "metrics" in result


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_incident_load_is_deterministic_under_fixed_seed():
    def run_one() -> list:
        orch = OrchestratorV2(n_students=5, max_classes=1, seed=77)
        orch.classroom.MAX_TURNS = 60
        orch.run_class()
        return sorted(orch.teacher_emotions.asdict().items())

    a = run_one()
    b = run_one()
    assert a == b
