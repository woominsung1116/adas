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
class ObservationRecord:
    """A single observation stored in the Case Base."""

    student_id: str
    turn: int
    observed_behaviors: list[str]
    state_snapshot: dict[str, float]  # distress/compliance/attention/escalation
    action_taken: str
    outcome: str  # 'positive', 'negative', 'neutral'
    behavior_vector: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.behavior_vector = _behavior_vector(self.observed_behaviors)


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
        Heuristic score (0-1) based on frequency of ADHD-linked behaviors.
        Weights hyperactivity/impulsivity slightly higher than inattention
        because they are more observable in classroom settings (KCI ART002794420).
        """
        hyper = sum(
            self.behavior_frequency_counts.get(b, 0) for b in HYPERACTIVITY_BEHAVIORS
        )
        impuls = sum(
            self.behavior_frequency_counts.get(b, 0) for b in IMPULSIVITY_BEHAVIORS
        )
        inatten = sum(
            self.behavior_frequency_counts.get(b, 0) for b in INATTENTION_BEHAVIORS
        )
        raw = 0.4 * hyper + 0.35 * impuls + 0.25 * inatten
        return min(1.0, raw / 10.0)


# ---------------------------------------------------------------------------
# Case Base
# ---------------------------------------------------------------------------


class CaseBase:
    """
    Stores ObservationRecords and supports RAG-style retrieval
    by cosine similarity on behavior vectors.
    """

    def __init__(self) -> None:
        self._records: list[ObservationRecord] = []

    def add(self, record: ObservationRecord) -> int:
        idx = len(self._records)
        self._records.append(record)
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

    def __init__(self) -> None:
        self.case_base = CaseBase()
        self.experience_base = ExperienceBase()
        self._metrics = CrossClassMetrics()

        self._profiles: dict[str, StudentProfile] = {}
        self._turn: int = 0
        self._pending_action: dict[str, str] = {}
        self._pending_state: dict[str, dict[str, float]] = {}
        self._pending_behaviors: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Class lifecycle
    # ------------------------------------------------------------------

    def new_class(self) -> None:
        """Reset per-student profiles and turn counter for a new class session."""
        self._profiles = {}
        self._turn = 0
        self._pending_action = {}
        self._pending_state = {}
        self._pending_behaviors = {}
        self._metrics.classes_seen += 1

    def advance_turn(self) -> None:
        self._turn += 1

    # ------------------------------------------------------------------
    # Core observation pipeline
    # ------------------------------------------------------------------

    def observe(
        self,
        student_id: str,
        behaviors: list[str],
        state: dict[str, float],
        action_taken: str = "none",
    ) -> None:
        """
        Record a new observation for a student.

        Args:
            student_id:    Unique student identifier.
            behaviors:     Observed behavior strings (from ALL_BEHAVIORS vocabulary).
            state:         State snapshot with keys distress_level, compliance,
                           attention, escalation_risk.
            action_taken:  Teacher action applied this turn.
        """
        profile = self._get_or_create_profile(student_id)
        for b in behaviors:
            if b in _BEHAVIOR_INDEX:
                profile.record_behavior(b)

        self._pending_action[student_id] = action_taken
        self._pending_state[student_id] = dict(state)
        self._pending_behaviors[student_id] = list(behaviors)

    def commit_observation(
        self,
        student_id: str,
        outcome: str = "neutral",
    ) -> int:
        """
        Commit the pending observation to the Case Base with its outcome.
        Call this after the environment has stepped so outcome is known.

        Args:
            student_id: Student to commit for.
            outcome:    'positive', 'negative', or 'neutral'.

        Returns:
            Index of the new record in the Case Base.
        """
        record = ObservationRecord(
            student_id=student_id,
            turn=self._turn,
            observed_behaviors=self._pending_behaviors.get(student_id, []),
            state_snapshot=self._pending_state.get(student_id, {}),
            action_taken=self._pending_action.get(student_id, "none"),
            outcome=outcome,
        )
        return self.case_base.add(record)

    # ------------------------------------------------------------------
    # RAG-style retrieval
    # ------------------------------------------------------------------

    def retrieve_similar_cases(
        self,
        observation: list[str],
        top_k: int = 5,
        exclude_student_id: Optional[str] = None,
    ) -> list[tuple[float, ObservationRecord]]:
        """
        Retrieve similar past cases by cosine similarity on behavior vectors.

        Args:
            observation:        List of observed behavior strings.
            top_k:              Number of cases to return.
            exclude_student_id: Exclude this student's own past records.

        Returns:
            List of (similarity_score, ObservationRecord) sorted descending.
        """
        return self.case_base.retrieve_similar(
            observation, top_k=top_k, exclude_student_id=exclude_student_id
        )

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

        # RAG: retrieve similar cases from other students
        similar = self.retrieve_similar_cases(dominant, top_k=5, exclude_student_id=student_id)
        case_votes: list[float] = []
        case_context_parts: list[str] = []
        for sim, rec in similar:
            if sim < 0.1:
                continue
            vote = 1.0 if rec.outcome == "positive" else 0.0
            case_votes.append(sim * vote)
            case_context_parts.append(
                f"similar case ({rec.student_id}, turn {rec.turn}): "
                f"{rec.observed_behaviors} -> {rec.outcome} (sim={sim:.2f})"
            )

        case_adjustment = (
            sum(case_votes) / max(len(case_votes), 1) if case_votes else 0.0
        )

        # Principle boost: corrective principles lower confidence
        principle_adjustment = 0.0
        applied_principles: list[str] = []
        for p in self.experience_base.top_principles(top_k=5):
            if self._principle_applies(p.text, dominant):
                delta = -0.05 if p.is_corrective else 0.05
                principle_adjustment += delta
                applied_principles.append(p.text)

        raw_confidence = (
            0.5 * base_score
            + 0.3 * case_adjustment
            + 0.2 * min(1.0, max(0.0, base_score + principle_adjustment))
        )
        confidence = min(1.0, max(0.0, raw_confidence))
        is_adhd = confidence >= 0.5

        reasoning_parts = [
            f"behavioral score={base_score:.2f}",
            f"case-base signal={case_adjustment:.2f} from {len(case_votes)} similar cases",
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

        Args:
            student_id:  Student whose identification is being evaluated.
            was_correct: Whether the identification was correct.
        """
        profile = self._get_or_create_profile(student_id)
        self._metrics.total_identifications += 1

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
        Manually add a principle to the Experience Base.

        Args:
            principle: Natural-language principle string.
            evidence:  List of Case Base record indices supporting this principle.

        Returns:
            True if the principle was accepted, False if rejected.
        """
        return self.experience_base.add_principle(
            text=principle,
            evidence_case_ids=evidence,
            is_corrective=False,
            case_base=self.case_base,
        )

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
