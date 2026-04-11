"""Tests for the Phase 6 slice 1 teacher observation + hypothesis layer.

Covers:
  - StudentObservation dataclass excludes latent-only fields
  - build_observations_from_classroom derives from StudentSummary only
  - TeacherHypothesis accumulates evidence across observations
  - first_suspicion_turn is recorded through the new explicit path
  - Working label restricted to HYPOTHESIS_LABELS
  - Determinism: fixed seed → same observation + hypothesis state
  - Existing simulator runs integrate without regressing metrics
"""

import pytest

from src.simulation.teacher_observation import (
    HYPOTHESIS_LABELS,
    LATENT_FIELD_BLACKLIST,
    StudentObservation,
    TeacherHypothesis,
    TeacherHypothesisBoard,
    TeacherObservationBatch,
    build_observations_from_classroom,
)
from src.simulation.classroom_env_v2 import (
    ClassroomObservation,
    DetailedObservation,
    StudentSummary,
)
from src.simulation.orchestrator_v2 import OrchestratorV2


# ---------------------------------------------------------------------------
# StudentObservation structural invariants
# ---------------------------------------------------------------------------


def _make_obs():
    return StudentObservation(
        student_id="S01",
        turn=5,
        visible_behaviors=("out_of_seat", "calling_out"),
        profile_hint="disruptive",
        seat_row=1,
        seat_col=2,
        is_identified=False,
        is_managed=False,
    )


def test_student_observation_excludes_latent_fields():
    obs = _make_obs()
    keys = set(obs.as_dict().keys())
    leaked = keys & LATENT_FIELD_BLACKLIST
    assert not leaked, f"observation leaked latent fields: {leaked}"


def test_student_observation_is_frozen():
    obs = _make_obs()
    with pytest.raises(Exception):
        obs.student_id = "S99"  # type: ignore[misc]


def test_student_observation_as_dict_round_trip_serializable():
    obs = _make_obs()
    d = obs.as_dict()
    # Ensure every value is JSON-serializable-ish (no custom objects)
    import json
    json.dumps(d)


# ---------------------------------------------------------------------------
# build_observations_from_classroom ignores detailed_observations
# ---------------------------------------------------------------------------


def _make_classroom_obs_with_detailed_leak():
    """A synthetic ClassroomObservation whose detailed_observations
    block carries latent state, to prove the builder ignores it."""
    summaries = [
        StudentSummary(
            student_id="S01",
            profile_hint="inattentive",
            behaviors=["out_of_seat", "fidgeting"],
            is_identified=False,
            is_managed=False,
            seat_row=0,
            seat_col=0,
        ),
        StudentSummary(
            student_id="S02",
            profile_hint="typical",
            behaviors=["on_task"],
            is_identified=False,
            is_managed=False,
            seat_row=0,
            seat_col=1,
        ),
    ]
    # This block would leak latent state if the builder read it.
    detailed = [
        DetailedObservation(
            student_id="S01",
            behaviors=["out_of_seat"],
            state_snapshot={
                "attention": 0.12,
                "compliance": 0.33,
                "distress_level": 0.88,
                "escalation_risk": 0.7,
            },
            emotional_cues={
                "anxiety": 0.9,
                "frustration": 0.8,
                "shame": 0.2,
            },
            recent_interactions=[],
        ),
    ]
    return ClassroomObservation(
        turn=42,
        day=1,
        period=2,
        subject="korean",
        location="classroom_A",
        student_summaries=summaries,
        detailed_observations=detailed,
        class_mood="tense",
        identified_adhd_ids=[],
        managed_ids=[],
    )


def test_build_observations_ignores_detailed_observations():
    classroom_obs = _make_classroom_obs_with_detailed_leak()
    batch = build_observations_from_classroom(classroom_obs)

    assert isinstance(batch, TeacherObservationBatch)
    assert batch.turn == 42
    assert batch.class_mood == "tense"
    assert len(batch) == 2

    all_keys: set[str] = set()
    for obs in batch:
        all_keys |= set(obs.as_dict().keys())
    leaked = all_keys & LATENT_FIELD_BLACKLIST
    assert not leaked, (
        f"builder leaked latent fields into observations: {leaked}. "
        "build_observations_from_classroom must read only "
        "student_summaries, never detailed_observations.state_snapshot / "
        "emotional_cues."
    )


def test_build_observations_preserves_visible_behaviors_in_order():
    classroom_obs = _make_classroom_obs_with_detailed_leak()
    batch = build_observations_from_classroom(classroom_obs)
    by_id = batch.by_student_id()
    assert by_id["S01"].visible_behaviors == ("out_of_seat", "fidgeting")
    assert by_id["S02"].visible_behaviors == ("on_task",)
    assert by_id["S01"].profile_hint == "inattentive"


# ---------------------------------------------------------------------------
# TeacherHypothesis evidence accumulation
# ---------------------------------------------------------------------------


def test_hypothesis_accumulates_behavior_counts_across_observations():
    h = TeacherHypothesis(student_id="S01")
    for turn in [1, 2, 3]:
        h.record_observation(
            StudentObservation(
                student_id="S01",
                turn=turn,
                visible_behaviors=("out_of_seat", "fidgeting"),
                profile_hint="disruptive",
                seat_row=0,
                seat_col=0,
                is_identified=False,
                is_managed=False,
            )
        )
    assert h.n_observations == 3
    assert h.last_observation_turn == 3
    assert h.evidence_behaviors["out_of_seat"] == 3
    assert h.evidence_behaviors["fidgeting"] == 3
    # Observations alone do NOT move suspicion_score or working_label
    assert h.suspicion_score == 0.0
    assert h.working_label == "unknown"
    assert h.first_suspicion_turn is None


def test_hypothesis_record_suspicion_sets_first_crossing_once():
    h = TeacherHypothesis(student_id="S01")
    first1 = h.record_suspicion(turn=10, score=0.2, threshold=0.15)
    first2 = h.record_suspicion(turn=20, score=0.5, threshold=0.15)
    assert first1 is True
    assert first2 is False
    assert h.first_suspicion_turn == 10
    assert h.suspicion_score == 0.5
    # History should contain both
    assert len(h.history) == 2


def test_hypothesis_set_working_label_rejects_unknown_label():
    h = TeacherHypothesis(student_id="S01")
    with pytest.raises(ValueError):
        h.set_working_label("schizophrenia")
    assert h.working_label == "unknown"


def test_hypothesis_record_suspicion_accepts_valid_label():
    h = TeacherHypothesis(student_id="S01")
    h.record_suspicion(
        turn=5, score=0.3, threshold=0.15,
        working_label="adhd_inattentive",
    )
    assert h.working_label == "adhd_inattentive"


def test_hypothesis_labels_contains_expected_set():
    # Locking the canonical set so later passes must deliberately
    # extend it rather than silently invent new labels.
    for required in ("unknown", "typical", "adhd_inattentive",
                     "adhd_hyperactive_impulsive", "adhd_combined",
                     "anxiety", "odd"):
        assert required in HYPOTHESIS_LABELS


# ---------------------------------------------------------------------------
# TeacherHypothesisBoard bulk update
# ---------------------------------------------------------------------------


def test_board_record_batch_creates_hypotheses_lazily():
    board = TeacherHypothesisBoard()
    classroom_obs = _make_classroom_obs_with_detailed_leak()
    batch = build_observations_from_classroom(classroom_obs)
    board.record_batch(batch)
    assert "S01" in board
    assert "S02" in board
    assert board.get("S01").n_observations == 1
    assert board.get("S02").n_observations == 1
    assert board.get("S01").evidence_behaviors["out_of_seat"] == 1


def test_board_first_suspicion_turns_empty_until_recorded():
    board = TeacherHypothesisBoard()
    board.get_or_create("S01")
    assert board.first_suspicion_turns() == {}
    board.record_suspicion("S01", turn=7, score=0.2, threshold=0.15)
    assert board.first_suspicion_turns() == {"S01": 7}


def test_board_as_dict_serializable():
    board = TeacherHypothesisBoard()
    board.get_or_create("S01")
    board.record_suspicion(
        "S01", turn=3, score=0.4, threshold=0.15,
        working_label="adhd_inattentive",
    )
    import json
    blob = board.as_dict()
    json.dumps(blob)
    assert blob["S01"]["working_label"] == "adhd_inattentive"
    assert blob["S01"]["first_suspicion_turn"] == 3


# ---------------------------------------------------------------------------
# Integration: run_class updates the board end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def short_class_result():
    """Run a tiny class and keep the orchestrator around for inspection."""
    orch = OrchestratorV2(n_students=5, max_classes=1, seed=7)
    orch.classroom.MAX_TURNS = 150  # Phase 1 = 100, spill into Phase 2
    result = orch.run_class()
    return orch, result


def test_orchestrator_populates_hypothesis_board(short_class_result):
    orch, _ = short_class_result
    # Every student in the classroom should have a hypothesis entry
    # after at least one turn of observation.
    assert len(orch.hypothesis_board) == len(orch.classroom.students)
    for sid in [s.student_id for s in orch.classroom.students]:
        h = orch.hypothesis_board.get(sid)
        assert h is not None
        assert h.n_observations > 0
        assert h.last_observation_turn is not None


def test_orchestrator_records_first_suspicion_through_board(short_class_result):
    orch, _ = short_class_result
    # If any student became suspicious, the board must know about it.
    # Compare against the legacy dict: they should agree on keys.
    legacy = set(orch._stream_first_suspicion_turns.keys())
    board = set(orch.hypothesis_board.first_suspicion_turns().keys())
    assert legacy == board, (
        f"board and legacy first-suspicion records disagree: "
        f"legacy={legacy} board={board}"
    )
    # And on the actual turn number.
    for sid in legacy:
        assert (
            orch._stream_first_suspicion_turns[sid]
            == orch.hypothesis_board.get(sid).first_suspicion_turn
        )


def test_orchestrator_current_teacher_obs_clean_of_latent_state(
    short_class_result,
):
    orch, _ = short_class_result
    batch = orch._current_teacher_obs
    assert batch is not None
    for obs in batch:
        leaked = set(obs.as_dict().keys()) & LATENT_FIELD_BLACKLIST
        assert not leaked, f"per-turn teacher obs leaked {leaked}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_orchestrator_board_is_deterministic_under_fixed_seed():
    def run_one() -> dict[str, int]:
        orch = OrchestratorV2(n_students=5, max_classes=1, seed=13)
        orch.classroom.MAX_TURNS = 120
        orch.run_class()
        return {
            sid: h.n_observations
            for sid, h in sorted(orch.hypothesis_board.items())
        }

    a = run_one()
    b = run_one()
    assert a == b
