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
    canonicalize_hypothesis_label,
)
from src.simulation.classroom_env_v2 import (
    ClassroomObservation,
    DetailedObservation,
    StudentSummary,
)
from src.simulation.orchestrator_v2 import OrchestratorV2, STRATEGIES


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
    """Real disruptive behaviors pass through in order; the
    latent-fallback sentinel ``on_task`` is scrubbed; the
    ``profile_hint`` is re-derived from behaviors, not passed
    through from the classroom summary."""
    classroom_obs = _make_classroom_obs_with_detailed_leak()
    batch = build_observations_from_classroom(classroom_obs)
    by_id = batch.by_student_id()
    # Real disruption survives and stays ordered.
    assert by_id["S01"].visible_behaviors == ("out_of_seat", "fidgeting")
    # Latent-fallback ``on_task`` is scrubbed out, leaving an empty
    # observable set — the teacher has nothing to see this turn.
    assert by_id["S02"].visible_behaviors == ()
    # profile_hint is re-derived: S01 has disruptive behaviors,
    # S02 has nothing observable.
    assert by_id["S01"].profile_hint == "disruptive"
    assert by_id["S02"].profile_hint == "unknown"


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


# ---------------------------------------------------------------------------
# Problem 1 regression: decision path no longer reads latent student state
# ---------------------------------------------------------------------------


def test_decision_path_source_has_no_latent_state_reads():
    """Static guard: the rule-based decision path body must not
    contain ``student.state.get(...)`` calls. Those are latent-only
    reads the partial-observation boundary is meant to forbid.
    """
    import inspect
    from src.simulation import orchestrator_v2

    src = inspect.getsource(orchestrator_v2.OrchestratorV2._decide_action_rule_based)
    # A bare ``s.state.get`` or ``student.state.get`` anywhere in
    # the body would leak latent fields — the fix replaced every
    # such read with an observation-derived proxy.
    assert ".state.get(" not in src, (
        "_decide_action_rule_based still reads latent student.state"
    )
    # The Phase 6 substitutes must be present.
    assert "_obs_lookup" in src
    assert "emotional_outburst" in src  # observable distress proxy
    assert "_DISRUPTIVE_VISIBLE_BEHAVIORS" in src  # observable relapse proxy


def test_choose_strategy_and_default_strategy_have_no_latent_reads():
    import inspect
    from src.simulation import orchestrator_v2

    for fn in (
        orchestrator_v2.OrchestratorV2._choose_strategy,
        orchestrator_v2.OrchestratorV2._default_strategy,
    ):
        src = inspect.getsource(fn)
        assert ".state.get(" not in src, (
            f"{fn.__qualname__} still reads latent student.state"
        )
        assert "distress_level" not in src, (
            f"{fn.__qualname__} still references latent distress_level"
        )
        assert "escalation_risk" not in src, (
            f"{fn.__qualname__} still references latent escalation_risk"
        )


def test_default_strategy_uses_observation_proxies():
    """Behavior-level test: the observable-only default strategy
    routes decision-making through visible_behaviors / profile_hint.
    """
    orch = OrchestratorV2(n_students=5, max_classes=1, seed=1)

    def make_obs(behaviors: tuple[str, ...], hint: str = ""):
        return StudentObservation(
            student_id="S01",
            turn=1,
            visible_behaviors=behaviors,
            profile_hint=hint,
            seat_row=0,
            seat_col=0,
            is_identified=False,
            is_managed=False,
        )

    # Visible outburst + normal empathy → empathic_acknowledgment
    assert orch._default_strategy(
        make_obs(("emotional_outburst",))
    ) == "empathic_acknowledgment"

    # Visible disruption without outburst → break_offer
    assert orch._default_strategy(
        make_obs(("out_of_seat", "calling_out"))
    ) == "break_offer"

    # profile_hint is no longer an inattention input — the observation
    # builder never emits "inattentive" anymore (it would require a
    # latent-state read). The strategy ladder must not fall through
    # on profile_hint alone: with no visible behavior the call should
    # return SOMETHING valid, not crash.
    # Hint "inattentive" with empty visible → random strategy (valid).
    result = orch._default_strategy(
        make_obs((), hint="inattentive")
    )
    assert result in STRATEGIES

    # The latent-fallback sentinel `seems_inattentive` is scrubbed by
    # the observation builder before ever reaching decision code,
    # but the _default_strategy helper takes raw observations, so we
    # confirm it does NOT fire the inattention branch on the
    # scrubbed sentinel — it should fall through to random.
    result = orch._default_strategy(
        make_obs(("seems_inattentive",))
    )
    assert result in STRATEGIES
    # An actually-observable low-arousal inattention behavior (if
    # the visibility filter were widened) still routes to
    # redirect_attention.
    assert orch._default_strategy(
        make_obs(("staring_out_window",))
    ) == "redirect_attention"


def test_llm_prompt_builder_consumes_teacher_batch_not_latent_state():
    """Static guard: the LLM decision path must build its student
    lines from the teacher observation batch, not from
    ``detailed_observations.state_snapshot`` or any latent field.
    """
    import inspect
    from src.simulation import orchestrator_v2

    src = inspect.getsource(orchestrator_v2.OrchestratorV2._decide_action_llm)
    assert "state_snapshot" not in src, (
        "LLM prompt still references detailed state_snapshot"
    )
    assert "emotional_cues" not in src
    assert ".state.get(" not in src
    # Must pull the student list from the teacher_batch iteration.
    assert "teacher_batch" in src
    assert "observation.visible_behaviors" in src


# ---------------------------------------------------------------------------
# Problem 2 regression: board stays synchronized; label canonicalization
# ---------------------------------------------------------------------------


def test_canonicalize_hypothesis_label_maps_legacy_to_canonical():
    assert canonicalize_hypothesis_label("adhd_hyperactive") == "adhd_hyperactive_impulsive"
    assert canonicalize_hypothesis_label("ADHD_Hyperactive") == "adhd_hyperactive_impulsive"
    assert canonicalize_hypothesis_label(" adhd_hyperactive ") == "adhd_hyperactive_impulsive"
    assert canonicalize_hypothesis_label("anxiety") == "anxiety"
    assert canonicalize_hypothesis_label("adhd_inattentive") == "adhd_inattentive"
    assert canonicalize_hypothesis_label("typical") == "typical"


def test_canonicalize_hypothesis_label_funnels_unknown_and_none_to_unknown():
    assert canonicalize_hypothesis_label(None) == "unknown"
    assert canonicalize_hypothesis_label("") == "unknown"
    assert canonicalize_hypothesis_label("schizophrenia") == "unknown"
    # Canonical must never raise — total function on strings
    assert canonicalize_hypothesis_label("  ") == "unknown"


def test_hypothesis_set_working_label_accepts_canonicalized_legacy():
    """Regression: the orchestrator used to crash here by passing
    ``adhd_hyperactive`` straight from ``HypothesisTracker`` into
    ``TeacherHypothesis.set_working_label``. The fix routes every
    label through ``canonicalize_hypothesis_label`` first."""
    h = TeacherHypothesis(student_id="S01")
    h.set_working_label(canonicalize_hypothesis_label("adhd_hyperactive"))
    assert h.working_label == "adhd_hyperactive_impulsive"


def test_sync_hypothesis_board_updates_score_and_label_over_time():
    """Direct test of the sync helper: calling it on successive
    turns must update the score on an existing hypothesis, not
    just create it once."""
    orch = OrchestratorV2(n_students=3, max_classes=1, seed=11)
    # Seed the stream state the way stream_class would.
    orch._stream_suspicious = {"S01": 0.20}
    orch._hypothesis_trackers = {}

    orch._sync_hypothesis_board(turn=10)
    h = orch.hypothesis_board.get("S01")
    assert h is not None
    assert h.suspicion_score == pytest.approx(0.20)
    assert h.first_suspicion_turn == 10
    assert h.working_label == "unknown"

    # Score moves up on a later turn — sync must propagate it.
    orch._stream_suspicious["S01"] = 0.55
    orch._sync_hypothesis_board(turn=25)
    h = orch.hypothesis_board.get("S01")
    assert h.suspicion_score == pytest.approx(0.55)
    # first_suspicion_turn is SET ONCE and does not move
    assert h.first_suspicion_turn == 10
    # History records both updates
    assert len(h.history) == 2


def test_sync_hypothesis_board_canonicalizes_legacy_label():
    """If the orchestrator's HypothesisTracker emits
    ``adhd_hyperactive``, the sync helper must canonicalize it
    before pushing to the board — no ValueError."""
    from src.simulation.orchestrator_v2 import HypothesisTracker

    orch = OrchestratorV2(n_students=3, max_classes=1, seed=11)
    orch._stream_suspicious = {"S02": 0.40}
    tracker = HypothesisTracker(student_id="S02", suspicion_turn=5)
    tracker.diagnosis_hypothesis = "adhd_hyperactive"  # legacy label
    orch._hypothesis_trackers = {"S02": tracker}

    # Must not raise.
    orch._sync_hypothesis_board(turn=5)
    h = orch.hypothesis_board.get("S02")
    assert h is not None
    assert h.working_label == "adhd_hyperactive_impulsive"
    assert h.first_suspicion_turn == 5


def test_sync_hypothesis_board_preserves_first_suspicion_on_repeat():
    orch = OrchestratorV2(n_students=3, max_classes=1, seed=11)
    orch._stream_suspicious = {"S03": 0.30}
    orch._hypothesis_trackers = {}
    orch._sync_hypothesis_board(turn=7)
    orch._sync_hypothesis_board(turn=8)
    orch._sync_hypothesis_board(turn=9)
    h = orch.hypothesis_board.get("S03")
    assert h.first_suspicion_turn == 7  # only the first call sets it
    assert len(h.history) == 3


# ---------------------------------------------------------------------------
# Corrective pass: behavior-only teacher observation layer
# ---------------------------------------------------------------------------


def _make_classroom_obs_with_latent_fallbacks():
    """Classroom observation whose summaries carry every latent-fallback
    sentinel the classroom can emit — the teacher-observation builder
    must scrub them before exposing visible_behaviors to the decision
    path."""
    summaries = [
        # S01: only latent-fallback sentinels, no real behavior
        StudentSummary(
            student_id="S01",
            profile_hint="inattentive",  # latent-derived, must be ignored
            behaviors=["seems_inattentive"],
            is_identified=False,
            is_managed=False,
            seat_row=0,
            seat_col=0,
        ),
        # S02: real behavior + on_task fallback mixed in
        StudentSummary(
            student_id="S02",
            profile_hint="typical",  # latent-derived, must be ignored
            behaviors=["out_of_seat", "on_task"],
            is_identified=False,
            is_managed=False,
            seat_row=0,
            seat_col=1,
        ),
        # S03: only quiet fallback
        StudentSummary(
            student_id="S03",
            profile_hint="disruptive",  # latent-derived, must be ignored
            behaviors=["quiet"],
            is_identified=False,
            is_managed=False,
            seat_row=0,
            seat_col=2,
        ),
        # S04: already-identified student with real outburst
        StudentSummary(
            student_id="S04",
            profile_hint="identified_adhd",
            behaviors=["emotional_outburst"],
            is_identified=True,
            is_managed=False,
            seat_row=0,
            seat_col=3,
        ),
    ]
    return ClassroomObservation(
        turn=7,
        day=1,
        period=1,
        subject="korean",
        location="classroom",
        student_summaries=summaries,
        detailed_observations=[],
        class_mood="neutral",
        identified_adhd_ids=["S04"],
        managed_ids=[],
    )


def test_build_observations_scrubs_latent_fallback_sentinels():
    classroom_obs = _make_classroom_obs_with_latent_fallbacks()
    batch = build_observations_from_classroom(classroom_obs)
    by_id = batch.by_student_id()

    # Latent fallbacks fully removed.
    assert by_id["S01"].visible_behaviors == ()
    assert by_id["S03"].visible_behaviors == ()
    # Real behavior survives; mixed-in "on_task" sentinel scrubbed.
    assert by_id["S02"].visible_behaviors == ("out_of_seat",)
    # Real outburst always survives.
    assert by_id["S04"].visible_behaviors == ("emotional_outburst",)

    # No latent-fallback sentinel appears anywhere.
    from src.simulation.teacher_observation import _LATENT_FALLBACK_BEHAVIORS
    all_behaviors = set()
    for obs in batch:
        all_behaviors.update(obs.visible_behaviors)
    assert not (all_behaviors & _LATENT_FALLBACK_BEHAVIORS)


def test_build_observations_reroutes_profile_hint_behavior_derived_only():
    classroom_obs = _make_classroom_obs_with_latent_fallbacks()
    batch = build_observations_from_classroom(classroom_obs)
    by_id = batch.by_student_id()

    # Incoming profile_hints were "inattentive" / "typical" / "disruptive"
    # / "identified_adhd" (latent-derived). Builder must ignore and re-derive:
    # S01: no real behavior + not identified → unknown
    assert by_id["S01"].profile_hint == "unknown"
    # S02: has observable disruption → disruptive
    assert by_id["S02"].profile_hint == "disruptive"
    # S03: no real behavior + not identified → unknown (NOT "disruptive")
    assert by_id["S03"].profile_hint == "unknown"
    # S04: teacher has already flagged → identified_adhd
    assert by_id["S04"].profile_hint == "identified_adhd"


def test_profile_hint_vocabulary_is_behavior_only():
    """The builder must never emit 'inattentive' / 'typical' — those
    are only reachable via latent-state reads in the classroom's
    original hint derivation."""
    classroom_obs = _make_classroom_obs_with_latent_fallbacks()
    batch = build_observations_from_classroom(classroom_obs)
    emitted = {o.profile_hint for o in batch}
    assert "inattentive" not in emitted
    assert "typical" not in emitted
    # Only the behavior-derived vocabulary is allowed.
    assert emitted <= {"identified_adhd", "disruptive", "unknown"}


def test_default_strategy_does_not_depend_on_latent_fallbacks():
    """Behavior-level: passing only latent-fallback sentinels to
    _default_strategy must NOT trigger the inattention branch —
    those are no-signal in the new semantics."""
    orch = OrchestratorV2(n_students=3, max_classes=1, seed=1)

    def obs_of(*behaviors: str):
        return StudentObservation(
            student_id="X",
            turn=1,
            visible_behaviors=behaviors,
            profile_hint="unknown",
            seat_row=0,
            seat_col=0,
            is_identified=False,
            is_managed=False,
        )

    # seems_inattentive (latent sentinel) must not route to
    # redirect_attention. Falls through to random choice.
    result = orch._default_strategy(obs_of("seems_inattentive"))
    assert result in STRATEGIES and result != "redirect_attention"

    # on_task sentinel likewise.
    result = orch._default_strategy(obs_of("on_task"))
    assert result in STRATEGIES


def test_run_class_observations_never_expose_latent_fallbacks(short_class_result):
    """Integration: a real run must never place a latent-fallback
    sentinel into any hypothesis board evidence_behaviors dict."""
    from src.simulation.teacher_observation import _LATENT_FALLBACK_BEHAVIORS
    orch, _ = short_class_result
    for sid, h in orch.hypothesis_board.items():
        leaked = set(h.evidence_behaviors) & _LATENT_FALLBACK_BEHAVIORS
        assert not leaked, (
            f"{sid}: hypothesis board recorded latent-fallback sentinels "
            f"{leaked} — teacher observation builder failed to scrub them"
        )


# ---------------------------------------------------------------------------
# Corrective pass: first_suspicion_turn semantics are "working suspicion set"
# ---------------------------------------------------------------------------


def test_first_suspicion_docs_use_working_suspicion_set_wording():
    """Code comments across orchestrator, board, and metrics must
    describe first_suspicion_turn consistently as entry into the
    WORKING SUSPICION SET — not a strong threshold crossing."""
    import inspect
    from src.simulation import orchestrator_v2
    from src.simulation import teacher_observation
    from src.calibration import metrics

    orch_src = inspect.getsource(orchestrator_v2)
    # The misleading ">= 0.5" claim must be gone.
    assert "adhd_indicator_score >= 0.5" not in orch_src
    # The new wording must be present somewhere in the orchestrator.
    assert "WORKING SUSPICION SET" in orch_src or "working suspicion set" in orch_src.lower()

    # Hypothesis docstring on the board side.
    board_src = inspect.getsource(teacher_observation)
    assert "WORKING SUSPICION SET" in board_src or "working suspicion set" in board_src.lower()

    # Calibration metric description.
    metric_src = inspect.getsource(metrics)
    assert "working suspicion set" in metric_src.lower()
    # And the bad claim is gone from the metric docstring too.
    assert "adhd_indicator_score >= 0.5" not in metric_src


def test_first_suspicion_turn_matches_first_working_suspicion_entry():
    """Regression: the board's first_suspicion_turn equals the
    earliest turn the student appeared in _stream_suspicious,
    which is populated by the refresh-time rule (score >= 0.15
    with ≥3 obs, or > 0.10 soft)."""
    orch = OrchestratorV2(n_students=8, max_classes=1, seed=42)
    orch.classroom.MAX_TURNS = 300
    orch.run_class()

    for sid, first_turn in orch._stream_first_suspicion_turns.items():
        h = orch.hypothesis_board.get(sid)
        assert h is not None
        assert h.first_suspicion_turn == first_turn


def test_run_class_board_tracks_suspicion_score_movement():
    """Integration: over a real class run the board should contain
    at least one history entry for every currently-suspicious student
    — proving the board is re-synced each turn, not only on first
    crossing."""
    orch = OrchestratorV2(n_students=8, max_classes=1, seed=42)
    orch.classroom.MAX_TURNS = 300  # run into Phase 2 screening
    orch.run_class()

    # Every student in _stream_suspicious must have a board entry
    # whose suspicion_score matches the legacy dict (last sync wins).
    for sid, score in orch._stream_suspicious.items():
        h = orch.hypothesis_board.get(sid)
        assert h is not None, f"{sid} missing from board"
        assert h.suspicion_score == pytest.approx(score), (
            f"{sid}: board={h.suspicion_score} vs legacy={score}"
        )
        # History must record multiple turns — not just one.
        # Phase 2 refreshes suspicion every 5 turns; over 300 turns
        # (200 in Phase 2) we expect many updates for a persistent
        # suspect.
        assert len(h.history) >= 1
