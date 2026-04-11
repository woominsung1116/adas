"""
Teacher agent memory system for ADHD classroom simulation.

Inspired by MedAgent-Zero (arXiv:2405.02957):
  - Case Base   (관찰 기록)    : per-student observation records with RAG-style retrieval
  - Experience Base (경험 원칙) : accumulated principles from successes and failures

Behavioral categories from Korean research:
  KCI ART002794420 - 과잉행동/충동성/부주의 행동 관찰 척도
  KCI ART002478306 - ADHD 아동 교실 행동 특성 연구

Hyperactivity  : seat-leaving, leg-swinging, paper-folding, running/climbing, excessive-talking
Impulsivity    : blurting-answers, interrupting, off-topic-comments, grabbing-objects
Inattention    : careless-mistakes, not-following-instructions, incomplete-tasks,
                 poor-organization, easily-distracted
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Behavior vocabulary (Korean research-derived)
# ---------------------------------------------------------------------------

HYPERACTIVITY_BEHAVIORS: tuple[str, ...] = (
    "seat-leaving",
    "leg-swinging",
    "paper-folding",
    "running/climbing",
    "excessive-talking",
)

IMPULSIVITY_BEHAVIORS: tuple[str, ...] = (
    "blurting-answers",
    "interrupting",
    "off-topic-comments",
    "grabbing-objects",
)

INATTENTION_BEHAVIORS: tuple[str, ...] = (
    "careless-mistakes",
    "not-following-instructions",
    "incomplete-tasks",
    "poor-organization",
    "easily-distracted",
)

ALL_BEHAVIORS: tuple[str, ...] = (
    HYPERACTIVITY_BEHAVIORS + IMPULSIVITY_BEHAVIORS + INATTENTION_BEHAVIORS
)

_BEHAVIOR_INDEX: dict[str, int] = {b: i for i, b in enumerate(ALL_BEHAVIORS)}


def _behavior_vector(behaviors: list[str]) -> np.ndarray:
    """Convert a list of behavior strings to a fixed-length indicator vector."""
    vec = np.zeros(len(ALL_BEHAVIORS), dtype=float)
    for b in behaviors:
        idx = _BEHAVIOR_INDEX.get(b)
        if idx is not None:
            vec[idx] = 1.0
    return vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ObservationOutcome:
    """Explicit teacher-visible outcome payload for a memory commit.

    Phase 6 slice 2 substrate: before this existed, the memory
    commit path blended together the observed behaviors, the
    action taken, a hand-coded ``outcome`` string, and a latent
    ``state_snapshot`` dump from the student object. That blended
    shape made it impossible to enforce the partial-observability
    boundary at the memory layer.

    This dataclass makes the feedback channel explicit and closes
    the boundary: every field is something the teacher could
    plausibly observe after their action, never a raw latent
    scalar. Future passes (delayed feedback, memory noise) can
    queue / corrupt / redate these payloads without touching the
    memory storage layer.

    Fields:
      outcome:         coarse teacher-assigned label — one of
                       ``"positive"``, ``"neutral"``, ``"negative"``.
                       The orchestrator derives this from the
                       student's post-action response in an
                       observable way (legacy code mapped compliance
                       thresholds onto these labels; the label
                       survives, the latent scalar does not).
      teacher_action:  which action the teacher took this turn
                       (e.g. ``"individual_intervention"``,
                       ``"observe"``). Teacher-side fact.
      post_behaviors:  optional tuple of teacher-visible behaviors
                       the student exhibited after the action. Empty
                       tuple when the classroom produced no
                       high-visibility follow-up. Future delayed-
                       feedback passes can populate this with
                       later-turn observations.
    """

    outcome: str = "neutral"
    teacher_action: str = "none"
    post_behaviors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.outcome not in ("positive", "neutral", "negative"):
            raise ValueError(
                f"ObservationOutcome.outcome must be positive / neutral / "
                f"negative, got {self.outcome!r}"
            )
        # Normalize to a tuple so the record is hashable-ish and
        # callers cannot mutate it by reference.
        if not isinstance(self.post_behaviors, tuple):
            self.post_behaviors = tuple(self.post_behaviors)

    def as_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "teacher_action": self.teacher_action,
            "post_behaviors": list(self.post_behaviors),
        }


@dataclass
class ObservationRecord:
    """A single teacher-memory observation.

    Phase 6 slice 2: the former ``state_snapshot`` field (a dump
    of latent ``distress_level`` / ``compliance`` / ``attention``
    / ``escalation_risk``) has been removed. Teacher memory now
    stores only observable cues plus an explicit
    ``ObservationOutcome`` feedback payload.

    Retrieval (case-base cosine similarity) still works: the
    similarity function reads ``behavior_vector``, which is
    derived from ``observed_behaviors`` — no latent fields were
    ever involved in retrieval.
    """

    student_id: str
    turn: int
    observed_behaviors: list[str]
    action_taken: str
    outcome: str  # 'positive' | 'negative' | 'neutral' — kept flat for
                  # backwards compatibility with existing retrieval code
                  # that inspects record.outcome directly.
    was_adhd: Optional[bool] = None  # set after identification outcome is known
    # Explicit teacher-visible feedback payload. The flat ``outcome``
    # and ``action_taken`` fields above are a convenience mirror of
    # this object; ``feedback`` is the authoritative source.
    feedback: Optional["ObservationOutcome"] = None
    behavior_vector: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.behavior_vector = _behavior_vector(self.observed_behaviors)
        # Keep flat fields and structured feedback in sync.
        if self.feedback is None:
            self.feedback = ObservationOutcome(
                outcome=self.outcome,
                teacher_action=self.action_taken,
            )
        else:
            # Authoritative source is the feedback object; mirror
            # its fields onto the flat ones for backward compat.
            self.outcome = self.feedback.outcome
            self.action_taken = self.feedback.teacher_action


@dataclass(frozen=True)
class RetrievalNoiseConfig:
    """Explicit, auditable teacher-memory retrieval noise knobs.

    Phase 6 slice 9: the legacy ``retrieval_noise`` scalar (just
    a per-result dropout probability) is too blunt to model
    imperfect teacher recall — it can only remove candidates,
    never reshuffle them. This config adds two orthogonal
    levers that together produce a richer "I sort of remember
    a similar case, maybe this one?" effect:

      dropout_prob:
        Per-candidate probability of forgetting the case
        entirely at recall time. Clamped to ``[0.0, 1.0]``.
      similarity_jitter:
        Magnitude (``>= 0``) of additive uniform noise applied
        to each candidate's similarity score BEFORE re-sorting.
        Range: a single draw from
        ``rng.uniform(-similarity_jitter, +similarity_jitter)``
        is added to each similarity. Zero is a no-op.

    Both levers default to zero so ``RetrievalNoiseConfig()`` is
    a pure pass-through and existing TeacherMemory callers see
    no behavior change.

    Stored records are NEVER mutated by this config — the noise
    layer operates on transient ``(similarity, record)`` tuples
    copied out of the case base.
    """

    dropout_prob: float = 0.0
    similarity_jitter: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "dropout_prob", _clamp01_local(self.dropout_prob))
        if self.similarity_jitter < 0.0:
            object.__setattr__(self, "similarity_jitter", 0.0)

    @property
    def is_disabled(self) -> bool:
        return self.dropout_prob == 0.0 and self.similarity_jitter == 0.0

    def as_dict(self) -> dict:
        return {
            "dropout_prob": self.dropout_prob,
            "similarity_jitter": self.similarity_jitter,
        }


def _clamp01_local(value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def apply_retrieval_noise(
    scored_results: list[tuple[float, "ObservationRecord"]],
    rng: random.Random,
    config: RetrievalNoiseConfig,
) -> list[tuple[float, "ObservationRecord"]]:
    """Apply explicit retrieval noise to a scored candidate list.

    Phase 6 slice 9 substrate. Pure function:
      * reads nothing but its arguments
      * does NOT mutate ``scored_results`` or any record
      * returns a NEW list of new tuples so callers can pass
        the return value into any downstream consumer safely

    Operation (in order):
      1. For each ``(sim, record)`` tuple, add a uniform jitter
         in ``[-similarity_jitter, +similarity_jitter]`` to
         ``sim`` (unless jitter is zero — early-return path
         consumes no RNG).
      2. For each candidate, roll ``rng.random()`` against
         ``dropout_prob`` and drop the candidate if it fires.
      3. Re-sort the surviving candidates by perturbed sim,
         descending.

    When ``config.is_disabled`` the helper returns the input
    list unchanged WITHOUT consuming any RNG state — this is
    the invariant that keeps legacy callers bit-identical
    to the pre-slice-9 baseline.
    """
    if config.is_disabled or not scored_results:
        return list(scored_results)

    jitter = config.similarity_jitter
    dropout = config.dropout_prob

    perturbed: list[tuple[float, "ObservationRecord"]] = []
    for sim, record in scored_results:
        if jitter > 0.0:
            noisy_sim = sim + rng.uniform(-jitter, jitter)
        else:
            noisy_sim = sim
        if dropout > 0.0 and rng.random() < dropout:
            # Dropped: teacher forgot this case entirely.
            continue
        perturbed.append((noisy_sim, record))

    perturbed.sort(key=lambda x: x[0], reverse=True)
    return perturbed


@dataclass
class PendingObservationFeedback:
    """A memory commit queued for a later turn (Phase 6 slice 3).

    Before this slice, the orchestrator derived every observation's
    outcome immediately on the same turn and committed the memory
    record right away — the teacher was getting unrealistically
    instantaneous feedback from their interventions. This dataclass
    represents a commit that has been STAGED but not yet applied.

    Fields intentionally avoid any latent student scalar: the only
    cross-turn state we need later is the student id (to re-query
    visible post-action signals), the observed behaviors that were
    seen at action time, the action the teacher took, and the
    two turn markers. Outcome derivation happens at dequeue time
    through the orchestrator's existing observable-only
    ``_derive_feedback_outcome`` helper.

    Fields:
      student_id:         who the observation is about
      observed_behaviors: teacher-visible behavior strings at
                          action time (already translated to the
                          ALL_BEHAVIORS vocabulary)
      teacher_action:     action_type the teacher took
      observed_turn:      turn at which the observation was made
                          (record.turn will be set from this)
      due_turn:           turn at which the commit should actually
                          happen (observed_turn + delay)
    """

    student_id: str
    observed_behaviors: list[str]
    teacher_action: str
    observed_turn: int
    due_turn: int
    # Phase 6 slice 4: teacher-visible disruptive behaviors the
    # teacher saw BEFORE the action executed. Used at drain time
    # by the observable-response feedback heuristic to label the
    # commit ``positive`` / ``negative`` / ``neutral`` without
    # reading latent compliance. Empty tuple means the teacher
    # had nothing disruptive to see before acting.
    pre_visible_disruptive: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "observed_behaviors": list(self.observed_behaviors),
            "teacher_action": self.teacher_action,
            "observed_turn": self.observed_turn,
            "due_turn": self.due_turn,
            "pre_visible_disruptive": list(self.pre_visible_disruptive),
        }


@dataclass
class PendingHypothesisFeedback:
    """A hypothesis-test effect queued for a later turn (Phase 6 slice 6).

    Before this slice, `_update_memory` called
    `HypothesisTracker.record_test(strategy, effect)` immediately
    on the same turn the intervention ran, so the teacher's
    hypothesis learning had unrealistic instant feedback. This
    dataclass represents a hypothesis-test outcome that has been
    STAGED but not yet applied.

    Fields intentionally avoid any latent scalar — exactly the
    same observable-only policy as
    ``PendingObservationFeedback``. Outcome derivation happens at
    drain time through ``observable_response_effect``.

    Fields:
      student_id:             the student whose hypothesis tracker
                              should receive the update
      strategy:               the hypothesis-test strategy applied
                              (one of ``_HYPOTHESIS_TEST_STRATEGIES``)
      observed_turn:          turn on which the intervention ran
      due_turn:               turn at which the tracker update
                              should actually happen
      pre_visible_disruptive: teacher-visible disruptive behaviors
                              snapshotted BEFORE the intervention
                              ran — same semantics as the
                              ``PendingObservationFeedback``
                              field of the same name
    """

    student_id: str
    strategy: str
    observed_turn: int
    due_turn: int
    pre_visible_disruptive: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "strategy": self.strategy,
            "observed_turn": self.observed_turn,
            "due_turn": self.due_turn,
            "pre_visible_disruptive": list(self.pre_visible_disruptive),
        }


class FeedbackDelayQueue:
    """Minimal FIFO queue for delayed-feedback items.

    Generic over any payload type that exposes a ``due_turn: int``
    attribute. The orchestrator owns two instances:

      * ``_feedback_queue`` — holds ``PendingObservationFeedback``
        for delayed teacher-memory commits
      * ``_hypothesis_feedback_queue`` — holds
        ``PendingHypothesisFeedback`` for delayed
        hypothesis-tracker updates

    Determinism:
      * order-preserving: items are popped in the order they were
        enqueued (stable sort by due_turn, then by enqueue index)
      * ``pop_due(current_turn)`` returns all items whose
        ``due_turn <= current_turn`` without touching the rest
      * ``flush_all()`` returns everything still pending, for use
        at class end so no observations silently disappear

    No RNG, no hashing, no wall-clock reads — a later memory-noise
    pass can wrap this queue but the queue itself is pure data.
    """

    def __init__(self) -> None:
        self._items: list = []

    def __len__(self) -> int:
        return len(self._items)

    def enqueue(self, item) -> None:
        self._items.append(item)

    def peek_all(self) -> list:
        return list(self._items)

    def pop_due(self, current_turn: int) -> list:
        """Return and remove every item whose ``due_turn <= current_turn``.

        The remaining items keep their original relative order.
        """
        due: list = []
        rest: list = []
        for item in self._items:
            if item.due_turn <= current_turn:
                due.append(item)
            else:
                rest.append(item)
        self._items = rest
        return due

    def flush_all(self) -> list:
        """Return and remove every pending item. Used at class end."""
        out = self._items
        self._items = []
        return out


@dataclass
class Principle:
    """A single entry in the Experience Base."""

    text: str
    evidence_case_ids: list[int]  # indices into CaseBase._records
    support_count: int = 1
    is_corrective: bool = False  # True when derived from a misidentification


@dataclass
class StudentProfile:
    """Per-student accumulated profile. Resets when a new class begins."""

    student_id: str
    behavior_frequency_counts: dict[str, int] = field(default_factory=dict)
    response_to_interventions: dict[str, float] = field(default_factory=dict)
    trust_level: float = 0.5
    identified_as_adhd: bool = False
    identification_confidence: float = 0.0
    identification_reasoning: str = ""

    def record_behavior(self, behavior: str) -> None:
        self.behavior_frequency_counts[behavior] = (
            self.behavior_frequency_counts.get(behavior, 0) + 1
        )

    def record_intervention_response(self, action: str, delta_compliance: float) -> None:
        old = self.response_to_interventions.get(action, 0.0)
        self.response_to_interventions[action] = 0.7 * old + 0.3 * delta_compliance

    def dominant_behaviors(self, top_k: int = 5) -> list[str]:
        return sorted(
            self.behavior_frequency_counts,
            key=lambda b: self.behavior_frequency_counts[b],
            reverse=True,
        )[:top_k]

    def adhd_indicator_score(self) -> float:
        """
        Heuristic score (0-1) based on PROPORTION of ADHD-linked behaviors
        relative to total observed behaviors. Uses proportion instead of
        absolute frequency so that a heavily-observed normal_active student
        (who shows some ADHD-like behaviors mixed with normal ones) does not
        score higher than a less-observed but genuinely ADHD student.

        Weights hyperactivity/impulsivity slightly higher than inattention
        because they are more observable in classroom settings (KCI ART002794420).
        """
        total = sum(self.behavior_frequency_counts.values())
        if total == 0:
            return 0.0

        hyper = sum(
            self.behavior_frequency_counts.get(b, 0) for b in HYPERACTIVITY_BEHAVIORS
        )
        impuls = sum(
            self.behavior_frequency_counts.get(b, 0) for b in IMPULSIVITY_BEHAVIORS
        )
        inatten = sum(
            self.behavior_frequency_counts.get(b, 0) for b in INATTENTION_BEHAVIORS
        )
        adhd_count = hyper + impuls + inatten
        adhd_ratio = adhd_count / total  # 0-1: what fraction of behaviors are ADHD-linked

        # Weight by category importance
        if adhd_count > 0:
            weighted = (0.4 * hyper + 0.35 * impuls + 0.25 * inatten) / adhd_count
        else:
            weighted = 0.0

        # Final score = ratio of ADHD behaviors × category weight
        # Require minimum 5 observations to avoid noisy early scores
        if total < 5:
            return adhd_ratio * weighted * 0.5  # dampen early
        return min(1.0, adhd_ratio * weighted)


# ---------------------------------------------------------------------------
# Case Base
# ---------------------------------------------------------------------------


class CaseBase:
    """
    Stores ObservationRecords and supports RAG-style retrieval
    by cosine similarity on behavior vectors.

    max_records caps storage to prevent O(n) retrieval from growing
    unboundedly across classes. When exceeded, oldest records are dropped.
    Default 10,000 keeps ~2 classes of history while staying fast (<2ms).
    """

    def __init__(self, max_records: int = 3000) -> None:
        self._records: list[ObservationRecord] = []
        self._max_records = max_records

    def add(self, record: ObservationRecord) -> int:
        idx = len(self._records)
        self._records.append(record)
        if len(self._records) > self._max_records:
            # Protect labeled records (was_adhd is not None) from eviction.
            # Partition into labeled (keep all) and unlabeled (keep recent).
            labeled = [r for r in self._records if r.was_adhd is not None]
            unlabeled = [r for r in self._records if r.was_adhd is None]
            keep_unlabeled = max(0, self._max_records - len(labeled))
            self._records = labeled + unlabeled[-keep_unlabeled:]
        return idx

    def retrieve_similar(
        self,
        query_behaviors: list[str],
        top_k: int = 5,
        exclude_student_id: Optional[str] = None,
    ) -> list[tuple[float, ObservationRecord]]:
        """
        Return the top_k most similar past cases by cosine similarity.
        Optionally exclude records from the same student to avoid trivial retrieval.
        """
        query_vec = _behavior_vector(query_behaviors)
        scored: list[tuple[float, ObservationRecord]] = []
        for record in self._records:
            if exclude_student_id and record.student_id == exclude_student_id:
                continue
            sim = _cosine_similarity(query_vec, record.behavior_vector)
            scored.append((sim, record))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._records)


# ---------------------------------------------------------------------------
# Experience Base
# ---------------------------------------------------------------------------


class ExperienceBase:
    """
    Accumulates principles derived from correct and incorrect identifications.
    Principles are validated against past cases before being retained.
    """

    def __init__(self) -> None:
        self._principles: list[Principle] = []

    def add_principle(
        self,
        text: str,
        evidence_case_ids: list[int],
        is_corrective: bool = False,
        case_base: Optional[CaseBase] = None,
    ) -> bool:
        """
        Add a principle if it does not duplicate an existing one.
        When a CaseBase is provided, the principle is validated: it must
        be consistent with at least one evidence case (returns False otherwise).
        """
        if case_base is not None and evidence_case_ids:
            valid = self._validate_against_cases(text, evidence_case_ids, case_base)
            if not valid:
                return False

        for existing in self._principles:
            if self._similar_text(existing.text, text):
                existing.support_count += 1
                for idx in evidence_case_ids:
                    if idx not in existing.evidence_case_ids:
                        existing.evidence_case_ids.append(idx)
                return True

        self._principles.append(
            Principle(
                text=text,
                evidence_case_ids=list(evidence_case_ids),
                is_corrective=is_corrective,
            )
        )
        return True

    def _validate_against_cases(
        self, principle_text: str, evidence_ids: list[int], case_base: CaseBase
    ) -> bool:
        """
        Minimal validation: at least one referenced case must exist in the case base.
        """
        return any(
            0 <= idx < len(case_base._records) for idx in evidence_ids
        )

    @staticmethod
    def _similar_text(a: str, b: str) -> bool:
        """Simple word-overlap similarity (Jaccard >= 0.5 → treat as duplicate)."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return False
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union) >= 0.5

    def top_principles(self, top_k: int = 5) -> list[Principle]:
        return sorted(self._principles, key=lambda p: p.support_count, reverse=True)[:top_k]

    def corrective_principles(self) -> list[Principle]:
        return [p for p in self._principles if p.is_corrective]

    def __len__(self) -> int:
        return len(self._principles)


# ---------------------------------------------------------------------------
# Cross-class performance tracker
# ---------------------------------------------------------------------------


@dataclass
class CrossClassMetrics:
    classes_seen: int = 0
    total_identifications: int = 0
    correct_identifications: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        tp = self.correct_identifications
        fp = self.false_positives
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        tp = self.correct_identifications
        fn = self.false_negatives
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------------------
# Main teacher memory system
# ---------------------------------------------------------------------------


class TeacherMemory:
    """
    Two-tier memory system for the teacher agent in the ADHD classroom simulation.

    Tier 1 (within-class): StudentProfile per student, reset each new class.
    Tier 2 (cross-class):  CaseBase + ExperienceBase persist across classes.

    Usage pattern:
        memory = TeacherMemory()
        memory.new_class()                          # reset student profiles
        memory.observe("s1", ["seat-leaving"], state)
        is_adhd, conf, reason = memory.identify_adhd("s1")
        memory.record_outcome("s1", was_correct=True)
    """

    def __init__(
        self,
        retrieval_noise: float = 0.0,
        principle_promotion_threshold: int = 7,
        principle_min_classes: int = 3,
        memory_decay_rate: float = 0.99,
        seed: int | None = None,
        retrieval_noise_config: RetrievalNoiseConfig | None = None,
    ) -> None:
        self.case_base = CaseBase()
        self.experience_base = ExperienceBase()
        self._metrics = CrossClassMetrics()

        self._profiles: dict[str, StudentProfile] = {}
        self._turn: int = 0
        self._pending_action: dict[str, str] = {}
        # Phase 6 slice 2: _pending_state removed. Teacher memory
        # no longer stores latent student state scalars; the feedback
        # path is the explicit ObservationOutcome payload passed into
        # commit_observation.
        self._pending_behaviors: dict[str, list[str]] = {}

        # Phase 2 enhancements: noise, promotion gating, decay
        self.retrieval_noise = retrieval_noise
        # Phase 6 slice 9: explicit retrieval noise config. Default
        # is a no-op; ``retrieve_similar_cases`` falls back to the
        # legacy ``retrieval_noise`` scalar when this config is
        # disabled, so existing callers see no behavior change.
        self.retrieval_noise_config: RetrievalNoiseConfig = (
            retrieval_noise_config
            if retrieval_noise_config is not None
            else RetrievalNoiseConfig()
        )
        self.principle_promotion_threshold = principle_promotion_threshold
        self.principle_min_classes = principle_min_classes
        self.memory_decay_rate = memory_decay_rate
        self._rng = random.Random(seed)

        # Pending principles that haven't yet met promotion threshold
        self._pending_principles: dict[str, list[int]] = defaultdict(list)
        # Track which class_id each case record belongs to
        self._record_class_ids: list[int] = []

    # ------------------------------------------------------------------
    # Class lifecycle
    # ------------------------------------------------------------------

    def new_class(self) -> None:
        """Reset per-student profiles and turn counter for a new class session."""
        self._profiles = {}
        self._turn = 0
        self._pending_action = {}
        self._pending_behaviors = {}
        self._metrics.classes_seen += 1
        self._current_class_id = self._metrics.classes_seen

    def advance_turn(self) -> None:
        self._turn += 1

    # ------------------------------------------------------------------
    # Core observation pipeline
    # ------------------------------------------------------------------

    def observe(
        self,
        student_id: str,
        behaviors: list[str],
        state: dict[str, float] | None = None,
        action_taken: str = "none",
    ) -> None:
        """
        Record a new observation for a student.

        Phase 6 slice 2: the ``state`` parameter is accepted for
        backward compatibility but its contents are NO LONGER
        stored anywhere inside teacher memory. It was previously
        copied into ``_pending_state`` and then dumped into
        ``ObservationRecord.state_snapshot``, which leaked latent
        ``distress_level`` / ``compliance`` / ``attention`` /
        ``escalation_risk`` scalars into the long-lived case base.
        The observable-only memory pipeline now only stores
        behaviors + teacher actions + a structured
        ``ObservationOutcome`` payload at commit time.

        Args:
            student_id:    Unique student identifier.
            behaviors:     Observed behavior strings (from
                           ALL_BEHAVIORS vocabulary).
            state:         IGNORED (accepted for legacy callers).
            action_taken:  Teacher action applied this turn.
        """
        del state  # explicitly dropped — not stored anywhere
        profile = self._get_or_create_profile(student_id)
        for b in behaviors:
            if b in _BEHAVIOR_INDEX:
                profile.record_behavior(b)

        self._pending_action[student_id] = action_taken
        self._pending_behaviors[student_id] = list(behaviors)

    def commit_observation(
        self,
        student_id: str,
        outcome: "str | ObservationOutcome" = "neutral",
    ) -> int:
        """Commit the pending observation to the Case Base.

        Accepts either a plain outcome label (legacy string form)
        or a structured ``ObservationOutcome`` payload (Phase 6
        slice 2 explicit feedback channel). In either case the
        stored record contains NO latent student state.

        Args:
            student_id: Student to commit for.
            outcome:    ``"positive"`` / ``"neutral"`` / ``"negative"``
                        string OR an ``ObservationOutcome`` instance
                        carrying outcome + teacher_action +
                        optional post_behaviors.

        Returns:
            Index of the new record in the Case Base.
        """
        if isinstance(outcome, ObservationOutcome):
            feedback = outcome
        else:
            feedback = ObservationOutcome(
                outcome=outcome,
                teacher_action=self._pending_action.get(student_id, "none"),
            )

        record = ObservationRecord(
            student_id=student_id,
            turn=self._turn,
            observed_behaviors=self._pending_behaviors.get(student_id, []),
            action_taken=feedback.teacher_action,
            outcome=feedback.outcome,
            feedback=feedback,
        )
        idx = self.case_base.add(record)
        # Track which class this record belongs to (for principle promotion)
        class_id = getattr(self, "_current_class_id", self._metrics.classes_seen)
        while len(self._record_class_ids) <= idx:
            self._record_class_ids.append(class_id)
        return idx

    def append_record(
        self,
        student_id: str,
        turn: int,
        observed_behaviors: list[str],
        feedback: "ObservationOutcome",
    ) -> int:
        """Commit an observation directly, bypassing the pending buffer.

        Phase 6 slice 3 hook for the delayed-feedback queue. Unlike
        ``commit_observation``, this helper does NOT read from
        ``_pending_behaviors`` / ``_pending_action`` — it takes all
        record fields as explicit arguments. That matters because a
        delayed commit may happen several turns after the original
        observation, by which time subsequent same-turn observations
        have overwritten the pending buffer for this student.

        No latent state is stored: the record carries only the
        explicit teacher-visible arguments and the feedback payload.
        """
        record = ObservationRecord(
            student_id=student_id,
            turn=turn,
            observed_behaviors=list(observed_behaviors),
            action_taken=feedback.teacher_action,
            outcome=feedback.outcome,
            feedback=feedback,
        )
        idx = self.case_base.add(record)
        class_id = getattr(self, "_current_class_id", self._metrics.classes_seen)
        while len(self._record_class_ids) <= idx:
            self._record_class_ids.append(class_id)
        return idx

    # ------------------------------------------------------------------
    # RAG-style retrieval
    # ------------------------------------------------------------------

    def retrieve_similar_cases(
        self,
        observation: list[str],
        top_k: int = 5,
        exclude_student_id: Optional[str] = None,
        noise_rate: float | None = None,
    ) -> list[tuple[float, ObservationRecord]]:
        """
        Retrieve similar past cases with retrieval noise and memory decay.

        Retrieval noise: ~20% chance of dropping each result (simulates
        imperfect teacher recall). Memory decay: older observations get
        reduced similarity scores.

        Args:
            observation:        List of observed behavior strings.
            top_k:              Number of cases to return.
            exclude_student_id: Exclude this student's own past records.
            noise_rate:         Override default retrieval_noise rate.

        Returns:
            List of (similarity_score, ObservationRecord) sorted descending.
        """
        rate = noise_rate if noise_rate is not None else self.retrieval_noise

        # Early exit: noise=1.0 drops everything, skip expensive O(N) scan
        if rate >= 1.0:
            return []

        # Early exit: empty case base
        if len(self.case_base) == 0:
            return []

        # Over-fetch to compensate for noise drops
        raw_results = self.case_base.retrieve_similar(
            observation, top_k=top_k + 3, exclude_student_id=exclude_student_id
        )

        # Apply memory decay: reduce similarity for old memories.
        # Constructs NEW (similarity, record) tuples — records
        # themselves are never mutated.
        decayed_results: list[tuple[float, ObservationRecord]] = []
        for sim, record in raw_results:
            age = max(0, self._turn - record.turn)
            decay_factor = self.memory_decay_rate ** age
            decayed_results.append((sim * decay_factor, record))

        # Re-sort after decay
        decayed_results.sort(key=lambda x: x[0], reverse=True)

        # Phase 6 slice 9: explicit retrieval noise layer. When
        # ``retrieval_noise_config`` is active, run similarity
        # jitter + per-candidate dropout through the auditable
        # ``apply_retrieval_noise`` helper. Otherwise fall back
        # to the legacy blunt per-result dropout.
        if not self.retrieval_noise_config.is_disabled:
            noisy_results = apply_retrieval_noise(
                decayed_results, self._rng, self.retrieval_noise_config
            )
        else:
            # Legacy path: simple per-result dropout.
            noisy_results = [
                r for r in decayed_results if self._rng.random() > rate
            ]
        return noisy_results[:top_k]

    # ------------------------------------------------------------------
    # ADHD identification
    # ------------------------------------------------------------------

    def identify_adhd(
        self, student_id: str
    ) -> tuple[bool, float, str]:
        """
        Identify whether a student is likely ADHD.

        Uses:
          1. StudentProfile.adhd_indicator_score() as a base signal.
          2. Similar cases from the Case Base to refine the estimate.
          3. Applicable principles from the Experience Base.

        Returns:
            (is_adhd, confidence, reasoning)
        """
        profile = self._get_or_create_profile(student_id)
        base_score = profile.adhd_indicator_score()
        dominant = profile.dominant_behaviors(top_k=5)

        # RAG: retrieve similar cases from other students.
        # Only use records with known ADHD labels (from confirmed identifications).
        similar = self.retrieve_similar_cases(dominant, top_k=5, exclude_student_id=student_id)
        case_votes: list[float] = []
        case_context_parts: list[str] = []
        for sim, rec in similar:
            if sim < 0.1:
                continue
            if rec.was_adhd is None:
                continue  # skip unlabeled records
            vote = 1.0 if rec.was_adhd else 0.0
            case_votes.append(sim * vote)
            case_context_parts.append(
                f"similar case ({rec.student_id}, turn {rec.turn}): "
                f"{rec.observed_behaviors} -> adhd={rec.was_adhd} (sim={sim:.2f})"
            )

        # Case-base signal: how many similar past students were ADHD?
        if case_votes:
            case_signal = sum(case_votes) / len(case_votes)
        else:
            case_signal = 0.0

        # Principle signal: corrective principles lower, positive raise
        principle_signal = 0.0
        applied_principles: list[str] = []
        for p in self.experience_base.top_principles(top_k=5):
            if self._principle_applies(p.text, dominant):
                delta = -0.08 if p.is_corrective else 0.08
                principle_signal += delta
                applied_principles.append(p.text)
        principle_signal = min(1.0, max(-1.0, principle_signal))

        # Blend: with populated case base, case evidence dominates and can
        # push confidence higher. Without case evidence, behavioral score
        # alone yields a lower ceiling, making early-class identification
        # harder (the growth curve we want).
        if case_votes:
            raw_confidence = (
                base_score * 0.3
                + case_signal * 0.5
                + min(1.0, max(0.0, base_score + principle_signal)) * 0.2
            )
        else:
            # No cross-student evidence: rely on behavioral score with a cap.
            # observation count gives slight boost with more data.
            n_obs = sum(profile.behavior_frequency_counts.values())
            obs_factor = min(1.0, n_obs / 20.0)  # ramps 0->1 over 20 obs
            raw_confidence = (
                base_score * (0.55 + 0.10 * obs_factor)
                + min(1.0, max(0.0, base_score + principle_signal)) * 0.15
            )
        confidence = min(1.0, max(0.0, raw_confidence))
        is_adhd = confidence >= 0.5

        reasoning_parts = [
            f"behavioral score={base_score:.2f}",
            f"case-base signal={case_signal:.2f} from {len(case_votes)} similar cases",
        ]
        if case_context_parts:
            reasoning_parts.append("evidence: " + "; ".join(case_context_parts[:2]))
        if applied_principles:
            reasoning_parts.append("principles: " + "; ".join(applied_principles[:2]))

        reasoning = " | ".join(reasoning_parts)

        profile.identified_as_adhd = is_adhd
        profile.identification_confidence = confidence
        profile.identification_reasoning = reasoning

        return is_adhd, confidence, reasoning

    @staticmethod
    def _principle_applies(principle_text: str, behaviors: list[str]) -> bool:
        """Check whether any behavior keyword appears in the principle text."""
        text_lower = principle_text.lower()
        return any(b.replace("-", " ") in text_lower or b in text_lower for b in behaviors)

    # ------------------------------------------------------------------
    # Outcome recording and principle extraction
    # ------------------------------------------------------------------

    def record_outcome(self, student_id: str, was_correct: bool) -> None:
        """
        Update cross-class metrics and extract a principle from this outcome.
        Also retroactively tags all case-base records for this student with
        the ADHD determination so future retrieval can use it.

        Args:
            student_id:  Student whose identification is being evaluated.
            was_correct: Whether the identification was correct.
        """
        profile = self._get_or_create_profile(student_id)
        self._metrics.total_identifications += 1

        # Determine actual ADHD status from identification + correctness
        if profile.identified_as_adhd:
            actual_adhd = was_correct  # identified as ADHD, correct -> is ADHD
        else:
            actual_adhd = not was_correct  # identified as non-ADHD, correct -> not ADHD

        # Retroactively tag all case-base records for this student
        for rec in self.case_base._records:
            if rec.student_id == student_id:
                rec.was_adhd = actual_adhd

        if was_correct:
            self._metrics.correct_identifications += 1
            self._extract_positive_principle(student_id, profile)
        else:
            if profile.identified_as_adhd:
                self._metrics.false_positives += 1
            else:
                self._metrics.false_negatives += 1
            self._extract_corrective_principle(student_id, profile)

    def _extract_positive_principle(
        self, student_id: str, profile: StudentProfile
    ) -> None:
        dominant = profile.dominant_behaviors(top_k=3)
        if not dominant:
            return
        behavior_str = " + ".join(dominant)
        principle = (
            f"Students who show {behavior_str} "
            f"in the first {min(self._turn, 10)} turns are likely ADHD "
            f"(confidence={profile.identification_confidence:.2f})."
        )
        recent_ids = [
            i
            for i, r in enumerate(self.case_base._records)
            if r.student_id == student_id
        ][-3:]
        self.experience_base.add_principle(
            text=principle,
            evidence_case_ids=recent_ids,
            is_corrective=False,
            case_base=self.case_base,
        )

    def _extract_corrective_principle(
        self, student_id: str, profile: StudentProfile
    ) -> None:
        dominant = profile.dominant_behaviors(top_k=3)
        if not dominant:
            return
        behavior_str = " + ".join(dominant)
        label = "ADHD" if profile.identified_as_adhd else "non-ADHD"
        principle = (
            f"Caution: {behavior_str} alone does not confirm {label}; "
            f"prior identification at confidence={profile.identification_confidence:.2f} was wrong."
        )
        recent_ids = [
            i
            for i, r in enumerate(self.case_base._records)
            if r.student_id == student_id
        ][-3:]
        self.experience_base.add_principle(
            text=principle,
            evidence_case_ids=recent_ids,
            is_corrective=True,
            case_base=self.case_base,
        )

    def add_principle(self, principle: str, evidence: list[int]) -> bool:
        """
        Add a principle with promotion gating.

        Principles require `principle_promotion_threshold` supporting cases
        from at least `principle_min_classes` different classes before being
        promoted to the Experience Base. Until then they accumulate in
        `_pending_principles`.

        Args:
            principle: Natural-language principle string.
            evidence:  List of Case Base record indices supporting this principle.

        Returns:
            True if the principle was promoted, False if still pending.
        """
        # Accumulate evidence in pending store
        pending = self._pending_principles[principle]
        for idx in evidence:
            if idx not in pending:
                pending.append(idx)

        # Check promotion criteria
        n_evidence = len(pending)
        distinct_classes = set()
        for idx in pending:
            if idx < len(self._record_class_ids):
                distinct_classes.add(self._record_class_ids[idx])

        if (
            n_evidence >= self.principle_promotion_threshold
            and len(distinct_classes) >= self.principle_min_classes
        ):
            promoted = self.experience_base.add_principle(
                text=principle,
                evidence_case_ids=list(pending),
                is_corrective=False,
                case_base=self.case_base,
            )
            if promoted:
                del self._pending_principles[principle]
            return promoted

        return False

    # ------------------------------------------------------------------
    # Cross-class growth report
    # ------------------------------------------------------------------

    def growth_report(self) -> dict[str, object]:
        """
        Return performance metrics showing improvement over classes.

        Returns a dict with keys:
            classes_seen, total_identifications, correct_identifications,
            false_positives, false_negatives, precision, recall, f1,
            case_base_size, experience_base_size, top_principles
        """
        m = self._metrics
        return {
            "classes_seen": m.classes_seen,
            "total_identifications": m.total_identifications,
            "correct_identifications": m.correct_identifications,
            "false_positives": m.false_positives,
            "false_negatives": m.false_negatives,
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "f1": round(m.f1, 4),
            "case_base_size": len(self.case_base),
            "experience_base_size": len(self.experience_base),
            "top_principles": [
                {"text": p.text, "support": p.support_count, "corrective": p.is_corrective}
                for p in self.experience_base.top_principles(top_k=5)
            ],
        }

    # ------------------------------------------------------------------
    # Profile access
    # ------------------------------------------------------------------

    def get_profile(self, student_id: str) -> StudentProfile:
        return self._get_or_create_profile(student_id)

    def _get_or_create_profile(self, student_id: str) -> StudentProfile:
        if student_id not in self._profiles:
            self._profiles[student_id] = StudentProfile(student_id=student_id)
        return self._profiles[student_id]

    def all_profiles(self) -> dict[str, StudentProfile]:
        return dict(self._profiles)
