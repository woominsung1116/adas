"""Tests for Phase 6 slice 7: behavior-derived class climate signal.

Covers:
  - derive_class_climate reads only observable visible_behaviors
  - calm / mixed / chaotic thresholds classify sensibly
  - Empty / single-student edge cases
  - TeacherObservationBatch exposes .climate populated by the builder
  - Builder-produced climate label matches derive_class_climate directly
  - Teacher emotion update path reads teacher_batch.climate, NOT
    self._stream_obs.class_mood
  - Determinism under fixed seed
  - No major regression in existing simulator runs
"""

import inspect

import pytest

from src.simulation.classroom_env_v2 import (
    ClassroomObservation,
    StudentSummary,
)
from src.simulation.teacher_observation import (
    CLASS_CLIMATE_LABELS,
    StudentObservation,
    TeacherObservationBatch,
    build_observations_from_classroom,
    derive_class_climate,
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
# derive_class_climate ladder
# ---------------------------------------------------------------------------


def test_empty_observations_is_calm():
    assert derive_class_climate([]) == "calm"


def test_all_calm_students_is_calm():
    batch = [_obs(f"S{i:02d}") for i in range(10)]
    assert derive_class_climate(batch) == "calm"


def test_one_in_ten_disruptive_is_calm():
    batch = [_obs("S01", "out_of_seat")] + [_obs(f"S{i:02d}") for i in range(9)]
    assert derive_class_climate(batch) == "calm"


def test_two_in_ten_disruptive_is_mixed():
    batch = (
        [_obs("S01", "out_of_seat"), _obs("S02", "calling_out")]
        + [_obs(f"S{i:02d}") for i in range(8)]
    )
    assert derive_class_climate(batch) == "mixed"


def test_three_in_ten_disruptive_is_mixed():
    batch = (
        [_obs(f"S{i}", "out_of_seat") for i in range(3)]
        + [_obs(f"S{i:02d}") for i in range(7)]
    )
    assert derive_class_climate(batch) == "mixed"


def test_five_in_ten_disruptive_is_chaotic():
    batch = (
        [_obs(f"S{i}", "calling_out") for i in range(5)]
        + [_obs(f"S{i:02d}") for i in range(5)]
    )
    assert derive_class_climate(batch) == "chaotic"


def test_all_disruptive_is_chaotic():
    batch = [_obs(f"S{i:02d}", "emotional_outburst") for i in range(6)]
    assert derive_class_climate(batch) == "chaotic"


def test_non_disruptive_behaviors_dont_count():
    """Profile hint alone (without visible disruptive behaviors)
    does not push a student into the disruptive bucket. The
    ladder only considers actual visible disruptive tokens."""
    # Students with seemingly concerning hints but nothing visible
    batch = [
        StudentObservation(
            student_id=f"S{i:02d}",
            turn=1,
            visible_behaviors=(),
            profile_hint="disruptive",  # hint ignored by ladder
            seat_row=0,
            seat_col=0,
            is_identified=False,
            is_managed=False,
        )
        for i in range(10)
    ]
    assert derive_class_climate(batch) == "calm"


def test_climate_labels_vocabulary_is_locked():
    # Ladder must only emit these three labels.
    for scenario in ("calm", "mixed", "chaotic"):
        assert scenario in CLASS_CLIMATE_LABELS
    assert len(CLASS_CLIMATE_LABELS) == 3


# ---------------------------------------------------------------------------
# derive_class_climate reads only visible_behaviors / profile_hint
# ---------------------------------------------------------------------------


def test_derive_class_climate_source_has_no_latent_reads():
    """Static guard: the helper body must only reach into
    visible_behaviors. Any attribute that smells latent (attention,
    compliance, distress_level, ...) should not appear."""
    src = inspect.getsource(derive_class_climate)
    for latent in (
        "distress_level",
        "compliance",
        "attention",
        "escalation_risk",
        ".state.get(",
        "state_snapshot",
    ):
        assert latent not in src


# ---------------------------------------------------------------------------
# Builder populates the climate field
# ---------------------------------------------------------------------------


def _classroom_obs_with(summaries: list[StudentSummary]) -> ClassroomObservation:
    return ClassroomObservation(
        turn=1,
        day=1,
        period=1,
        subject="korean",
        location="classroom",
        student_summaries=summaries,
        detailed_observations=[],
        class_mood="neutral",  # legacy latent-derived, must be ignored
        identified_adhd_ids=[],
        managed_ids=[],
    )


def test_builder_populates_climate_calm():
    summaries = [
        StudentSummary(
            student_id=f"S{i:02d}",
            profile_hint="typical",
            behaviors=[],
            is_identified=False,
            is_managed=False,
            seat_row=0, seat_col=0,
        )
        for i in range(6)
    ]
    batch = build_observations_from_classroom(_classroom_obs_with(summaries))
    assert batch.climate == "calm"


def test_builder_populates_climate_chaotic():
    summaries = [
        StudentSummary(
            student_id=f"S{i:02d}",
            profile_hint="typical",
            behaviors=["out_of_seat", "calling_out"],
            is_identified=False,
            is_managed=False,
            seat_row=0, seat_col=0,
        )
        for i in range(6)
    ]
    batch = build_observations_from_classroom(_classroom_obs_with(summaries))
    assert batch.climate == "chaotic"


def test_builder_climate_ignores_latent_class_mood_field():
    """Even if the classroom labels class_mood as 'tense' based on
    latent aggregates, the builder must emit the behavior-derived
    climate, which for an all-calm summary list is 'calm'."""
    summaries = [
        StudentSummary(
            student_id="S01",
            profile_hint="typical",
            behaviors=[],
            is_identified=False,
            is_managed=False,
            seat_row=0, seat_col=0,
        ),
    ]
    obs = _classroom_obs_with(summaries)
    # Manually tamper with class_mood to the worst latent label
    obs.class_mood = "tense"
    batch = build_observations_from_classroom(obs)
    assert batch.climate == "calm"
    # The legacy field is still preserved unchanged on the batch
    # for any consumer that explicitly asks for it.
    assert batch.class_mood == "tense"


# ---------------------------------------------------------------------------
# Teacher emotion update path reads the new signal
# ---------------------------------------------------------------------------


def test_stream_class_feeds_teacher_emotions_from_batch_climate():
    """Static guard: the stream loop must pass
    ``teacher_batch.climate`` (not ``self._stream_obs.class_mood``)
    into the emotion update call."""
    from src.simulation import orchestrator_v2 as m
    src = inspect.getsource(m.OrchestratorV2.stream_class)
    # The old latent path must be gone.
    assert "self._stream_obs.class_mood" not in src
    # The new behavior-derived path must be wired.
    assert "teacher_batch.climate" in src


def test_orchestrator_uses_climate_in_update_after_turn():
    """Behavior test: run a class and confirm the emotion
    update path can be driven through the new climate signal
    by observing that the teacher's patience value changed
    during the run (or stayed at the calm baseline) — either
    outcome is valid as long as the call path did not crash."""
    orch = OrchestratorV2(n_students=5, max_classes=1, seed=7)
    orch.classroom.MAX_TURNS = 50
    orch.run_class()
    # No crash is the primary assertion. Sanity-check that the
    # teacher's state dict still contains the expected keys.
    blob = orch.teacher_emotions.asdict()
    assert "patience" in blob
    assert "emotional_exhaustion" in blob


# ---------------------------------------------------------------------------
# Determinism + regression
# ---------------------------------------------------------------------------


def test_run_class_climate_is_deterministic_under_fixed_seed():
    def run_one() -> list[tuple]:
        orch = OrchestratorV2(n_students=5, max_classes=1, seed=77)
        orch.classroom.MAX_TURNS = 60
        orch.run_class()
        # Final teacher emotion snapshot acts as a fingerprint.
        return sorted(orch.teacher_emotions.asdict().items())

    a = run_one()
    b = run_one()
    assert a == b
