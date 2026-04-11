"""Teacher-facing partial observation layer (Phase 6 slice 1).

This module introduces an explicit boundary between:

  1. Latent student state (cognitive parameters, emotions, distress,
     attention scalars, etc.) -- owned by ``CognitiveStudent`` in
     ``classroom_env_v2``.
  2. Observable cues the teacher could plausibly witness from the
     front of the classroom -- expressed as ``StudentObservation``
     here. These are derived ONLY from ``StudentSummary`` (the
     already visibility-filtered view) and NEVER from
     ``DetailedObservation.state_snapshot`` / ``.emotional_cues``
     which leak latent fields.
  3. Teacher-side interpreted evidence and working hypothesis --
     expressed as ``TeacherHypothesis`` / ``TeacherHypothesisBoard``
     here. This is the substrate later passes will build on for
     hypothesis testing, delayed feedback, and memory noise.

Design notes for later phases:

  * ``StudentObservation`` stores tuples of observed behavior strings,
    not structured scores. Later passes can layer noise on top of this
    (drop/keep probability, mislabel probability) without changing
    the boundary.
  * ``TeacherHypothesis`` is deliberately serializable
    (``as_dict``/``from_dict``) so future delayed-feedback logic can
    snapshot it to a queue and re-apply it later without pickling
    orchestrator internals.
  * ``TeacherHypothesisBoard`` tracks ``first_suspicion_turn`` through
    an explicit record method. The orchestrator used to maintain this
    in a parallel dict; the board is now the authoritative record.

This module is pure data + small pure functions. It owns NO random
state and makes NO decisions; it is consumed by the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Latent field blacklist -- used by tests to prove observations stay clean
# ---------------------------------------------------------------------------


#: Fields that describe latent student state and MUST NEVER appear in the
#: keys of a serialized ``StudentObservation``. Enforced by
#: ``test_teacher_observation`` via structural assertion.
LATENT_FIELD_BLACKLIST: frozenset[str] = frozenset({
    # Cognitive parameters (cognitive_agent.CognitiveParams)
    "att_bandwidth",
    "vision_r",
    "retention",
    "recency_decay",
    "importance_trigger",
    "plan_consistency",
    "impulse_override",
    "task_initiation_delay",
    "social_sensitivity",
    "conflict_tendency",
    # Emotional state (cognitive_agent.EmotionalState)
    "frustration",
    "shame",
    "anxiety",
    "anger",
    "loneliness",
    "excitement",
    "self_esteem",
    "trust_in_teacher",
    # Classroom-visible-but-internal scalars
    "distress_level",
    "escalation_risk",
    "attention",
    "compliance",
    # Ground truth label -- the teacher may SUSPECT this but must
    # never read it directly from the observation layer.
    "true_profile",
    "true_adhd_label",
    "state_snapshot",
    "emotional_cues",
})


#: Canonical set of working hypothesis labels a teacher may hold
#: against a student. ``unknown`` is the default when no suspicion
#: has been raised yet; ``typical`` means actively ruled out.
HYPOTHESIS_LABELS: frozenset[str] = frozenset({
    "unknown",
    "typical",
    "adhd_inattentive",
    "adhd_hyperactive_impulsive",
    "adhd_combined",
    "anxiety",
    "odd",
    "other",
})


#: Mapping from legacy / short hypothesis labels emitted by other
#: parts of the simulator (notably ``HypothesisTracker.likely_profile``
#: which uses ``"adhd_hyperactive"``) to the canonical set above.
#: This table is the single normalization point — callers should
#: route through ``canonicalize_hypothesis_label`` rather than
#: inlining string substitutions.
_LEGACY_HYPOTHESIS_ALIASES: dict[str, str] = {
    "adhd_hyperactive": "adhd_hyperactive_impulsive",
    "adhd-hyperactive": "adhd_hyperactive_impulsive",
    "adhd_h": "adhd_hyperactive_impulsive",
    "adhd_i": "adhd_inattentive",
    "adhd_c": "adhd_combined",
    "normal": "typical",
    "non_adhd": "typical",
    "": "unknown",
}


def canonicalize_hypothesis_label(raw: str | None) -> str:
    """Map any caller-supplied label into ``HYPOTHESIS_LABELS``.

    * ``None``/empty → ``"unknown"``.
    * Whitespace is stripped and lowercased before lookup.
    * Legacy aliases (e.g. ``"adhd_hyperactive"``) are rewritten via
      ``_LEGACY_HYPOTHESIS_ALIASES``.
    * Anything already in ``HYPOTHESIS_LABELS`` passes through.
    * Anything else → ``"unknown"`` — never raises. Keeping this
      total avoids blowing up the simulator when future components
      invent new labels; the board's canonical set stays the
      source of truth and noise gets funneled into ``"unknown"``.
    """
    if raw is None:
        return "unknown"
    key = raw.strip().lower()
    if not key:
        return "unknown"
    if key in HYPOTHESIS_LABELS:
        return key
    if key in _LEGACY_HYPOTHESIS_ALIASES:
        return _LEGACY_HYPOTHESIS_ALIASES[key]
    return "unknown"


# ---------------------------------------------------------------------------
# Observation objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StudentObservation:
    """Per-turn observable cues for a single student.

    Only contains data the teacher could plausibly witness from the
    front of the room:

      * ``visible_behaviors``: pre-filtered high-visibility behavior
        strings (e.g. out_of_seat, calling_out, fidgeting). Already
        restricted by ``ClassroomV2._visible_behaviors``.
      * ``profile_hint``: a coarse categorical label the teacher forms
        from visible behavior, NOT the ground-truth profile. Derived
        by ``ClassroomV2._build_observation``.
      * ``seat_row`` / ``seat_col``: physical location.
      * ``is_identified`` / ``is_managed``: teacher-side flags the
        teacher has already set themselves. Not latent truth.

    Fields explicitly excluded (would leak latent state):
      * cognitive parameters
      * raw emotional scalars
      * attention / compliance / distress_level / escalation_risk
      * true profile label
    """

    student_id: str
    turn: int
    visible_behaviors: tuple[str, ...]
    profile_hint: str
    seat_row: int
    seat_col: int
    is_identified: bool
    is_managed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "student_id": self.student_id,
            "turn": self.turn,
            "visible_behaviors": list(self.visible_behaviors),
            "profile_hint": self.profile_hint,
            "seat_row": self.seat_row,
            "seat_col": self.seat_col,
            "is_identified": self.is_identified,
            "is_managed": self.is_managed,
        }


@dataclass(frozen=True)
class TeacherObservationBatch:
    """All observations the teacher has available at one turn.

    Holds the per-student observations plus a small set of class-level
    cues that are already derived in
    ``ClassroomObservation`` (turn, class_mood) and safe to surface.
    """

    turn: int
    class_mood: str
    observations: tuple[StudentObservation, ...]

    def by_student_id(self) -> dict[str, StudentObservation]:
        return {o.student_id: o for o in self.observations}

    def __iter__(self) -> Iterable[StudentObservation]:
        return iter(self.observations)

    def __len__(self) -> int:
        return len(self.observations)


# ---------------------------------------------------------------------------
# Hypothesis / suspicion state
# ---------------------------------------------------------------------------


@dataclass
class TeacherHypothesis:
    """Per-student teacher working hypothesis + accumulated evidence.

    This is the teacher's *interpretation* of observations — explicitly
    separate from the student's latent ground truth. Serializable so
    future delayed-feedback logic can snapshot it.

    Fields:
      student_id:            which student this hypothesis is about
      suspicion_score:       last-updated suspicion score in [0, 1]
      working_label:         current working hypothesis category,
                             drawn from ``HYPOTHESIS_LABELS``
      first_suspicion_turn:  turn when suspicion_score first crossed
                             the orchestrator's suspicion threshold;
                             None until then. This is the authoritative
                             first-suspicion record (the orchestrator
                             previously kept a parallel dict).
      last_observation_turn: most recent turn this student produced
                             an observation entry
      n_observations:        cumulative count of observations that
                             have fed this hypothesis
      evidence_behaviors:    cumulative dict of behavior_string -> count
                             derived entirely from ``StudentObservation``
      history:               list of (turn, suspicion_score, working_label)
                             triples, appended on each update
    """

    student_id: str
    suspicion_score: float = 0.0
    working_label: str = "unknown"
    first_suspicion_turn: int | None = None
    last_observation_turn: int | None = None
    n_observations: int = 0
    evidence_behaviors: dict[str, int] = field(default_factory=dict)
    history: list[tuple[int, float, str]] = field(default_factory=list)

    def record_observation(self, observation: StudentObservation) -> None:
        """Update counts + last-seen turn from a single observation.

        Intentionally does NOT move suspicion_score or working_label —
        those are updated by the orchestrator through
        ``record_suspicion`` / ``set_working_label`` once the
        decision path has computed them. This separation keeps the
        data-recording path pure and the interpretation path explicit.
        """
        self.n_observations += 1
        self.last_observation_turn = observation.turn
        for behavior in observation.visible_behaviors:
            self.evidence_behaviors[behavior] = (
                self.evidence_behaviors.get(behavior, 0) + 1
            )

    def record_suspicion(
        self,
        turn: int,
        score: float,
        threshold: float,
        working_label: str | None = None,
    ) -> bool:
        """Update suspicion score and return True if first crossing.

        If ``score`` is at or above ``threshold`` and
        ``first_suspicion_turn`` is still None, records ``turn`` as
        the first suspicion turn and returns True. Otherwise returns
        False.

        If ``working_label`` is provided, it must be in
        ``HYPOTHESIS_LABELS``.
        """
        self.suspicion_score = float(score)
        if working_label is not None:
            self.set_working_label(working_label)
        first = False
        if score >= threshold and self.first_suspicion_turn is None:
            self.first_suspicion_turn = turn
            first = True
        self.history.append((turn, float(score), self.working_label))
        return first

    def set_working_label(self, label: str) -> None:
        if label not in HYPOTHESIS_LABELS:
            raise ValueError(
                f"unknown hypothesis label {label!r}; "
                f"must be one of {sorted(HYPOTHESIS_LABELS)}"
            )
        self.working_label = label

    def as_dict(self) -> dict[str, Any]:
        return {
            "student_id": self.student_id,
            "suspicion_score": self.suspicion_score,
            "working_label": self.working_label,
            "first_suspicion_turn": self.first_suspicion_turn,
            "last_observation_turn": self.last_observation_turn,
            "n_observations": self.n_observations,
            "evidence_behaviors": dict(self.evidence_behaviors),
            "history": list(self.history),
        }


class TeacherHypothesisBoard:
    """Per-student hypothesis registry held by the teacher.

    Thin wrapper over a dict so we can attach helpers without
    exposing internal state. The orchestrator owns one of these per
    class (reset between classes) and feeds it observations per turn.
    """

    def __init__(self) -> None:
        self._hypotheses: dict[str, TeacherHypothesis] = {}

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, student_id: str) -> TeacherHypothesis | None:
        return self._hypotheses.get(student_id)

    def get_or_create(self, student_id: str) -> TeacherHypothesis:
        h = self._hypotheses.get(student_id)
        if h is None:
            h = TeacherHypothesis(student_id=student_id)
            self._hypotheses[student_id] = h
        return h

    def __contains__(self, student_id: str) -> bool:
        return student_id in self._hypotheses

    def __len__(self) -> int:
        return len(self._hypotheses)

    def items(self) -> Iterable[tuple[str, TeacherHypothesis]]:
        return self._hypotheses.items()

    # ------------------------------------------------------------------
    # Bulk updates
    # ------------------------------------------------------------------

    def record_batch(self, batch: TeacherObservationBatch) -> None:
        """Fold one turn's observation batch into the board.

        Creates hypotheses lazily for previously-unseen students.
        """
        for obs in batch.observations:
            h = self.get_or_create(obs.student_id)
            h.record_observation(obs)

    def record_suspicion(
        self,
        student_id: str,
        turn: int,
        score: float,
        threshold: float,
        working_label: str | None = None,
    ) -> bool:
        """Route suspicion updates through the board.

        Returns True if this call represents the first-ever crossing
        of the suspicion threshold for this student.
        """
        h = self.get_or_create(student_id)
        return h.record_suspicion(turn, score, threshold, working_label)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def first_suspicion_turns(self) -> dict[str, int]:
        """Return ``{student_id: first_suspicion_turn}`` for all
        hypotheses where suspicion has been raised at least once."""
        return {
            sid: h.first_suspicion_turn
            for sid, h in self._hypotheses.items()
            if h.first_suspicion_turn is not None
        }

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {sid: h.as_dict() for sid, h in self._hypotheses.items()}


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------


def build_observations_from_classroom(classroom_obs: Any) -> TeacherObservationBatch:
    """Project a ``ClassroomObservation`` into a partial-observation batch.

    Reads ONLY from ``student_summaries`` — the already
    visibility-filtered view. Deliberately IGNORES
    ``detailed_observations``, because ``DetailedObservation``
    carries ``state_snapshot`` (latent cognitive/emotional state)
    and ``emotional_cues`` which would leak latent truth into the
    teacher's view.

    This function is the single enforcement point for partial
    observability: if a future caller tries to read latent state
    for the teacher, they will have to go around this boundary and
    it will be obvious in review.
    """
    turn = int(getattr(classroom_obs, "turn", 0))
    class_mood = str(getattr(classroom_obs, "class_mood", "neutral"))
    summaries = getattr(classroom_obs, "student_summaries", []) or []

    observations: list[StudentObservation] = []
    for s in summaries:
        observations.append(
            StudentObservation(
                student_id=str(s.student_id),
                turn=turn,
                visible_behaviors=tuple(s.behaviors or ()),
                profile_hint=str(s.profile_hint),
                seat_row=int(s.seat_row),
                seat_col=int(s.seat_col),
                is_identified=bool(s.is_identified),
                is_managed=bool(s.is_managed),
            )
        )
    return TeacherObservationBatch(
        turn=turn,
        class_mood=class_mood,
        observations=tuple(observations),
    )


__all__ = [
    "LATENT_FIELD_BLACKLIST",
    "HYPOTHESIS_LABELS",
    "canonicalize_hypothesis_label",
    "StudentObservation",
    "TeacherObservationBatch",
    "TeacherHypothesis",
    "TeacherHypothesisBoard",
    "build_observations_from_classroom",
]
