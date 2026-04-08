"""
classroom_env_v2.py

950-turn (1 academic year) classroom environment with CognitiveStudent agents,
peer relationships, and student-student interactions.

Time model: 190 school days x 5 periods/day = 950 turns.
Each turn = one class period (~40 min).

Korean epidemiological data sources:
  - ADHD prevalence 6-11%: KCI ART001701933
  - Male:Female ADHD ratio 3.19:1: KCI ART001701933
  - Subtypes inattentive/HI/combined: 5.0/2.3/2.3 proportional
  - Comorbidity rates: PMC5290097
  - ODD-anxiety bully dynamic: KCI ART003153213
  - Behavioral frequencies: KCI ART002478306
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.simulation.interaction_log import InteractionEvent, InteractionLog

# ---------------------------------------------------------------------------
# Import cognitive_agent classes (stub fallback if not yet created)
# ---------------------------------------------------------------------------
try:
    from src.simulation.cognitive_agent import (
        CognitiveStudent,
        CognitiveParameters,
        EmotionalState,
        ClassroomContext,
        COGNITIVE_PRESETS,
        EMOTIONAL_PRESETS,
        RelationshipGraph,
    )
except ImportError:
    # Stubs so the module loads even if cognitive_agent.py is not ready yet.
    class CognitiveParameters:  # type: ignore[no-redef]
        pass

    class EmotionalState:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any):
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.valence = getattr(self, "valence", 0.5)
            self.arousal = getattr(self, "arousal", 0.3)
            self.stress = getattr(self, "stress", 0.2)
            self.frustration = getattr(self, "frustration", 0.1)
            self.engagement = getattr(self, "engagement", 0.5)

        def to_dict(self) -> dict[str, float]:
            return {
                "valence": self.valence,
                "arousal": self.arousal,
                "stress": self.stress,
                "frustration": self.frustration,
                "engagement": self.engagement,
            }

    class ClassroomContext:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class CognitiveStudent:  # type: ignore[no-redef]
        """Minimal stub matching expected CognitiveStudent interface."""

        def __init__(
            self,
            student_id: str = "",
            profile_type: str = "normal_quiet",
            age: int = 9,
            gender: str = "male",
            severity: str | None = None,
        ):
            self.student_id = student_id
            self.profile_type = profile_type
            self.age = age
            self.gender = gender
            self.severity = severity
            self.is_adhd = profile_type.startswith("adhd_")
            self.emotions = EmotionalState()
            self.state: dict[str, float] = {
                "distress_level": 0.15,
                "compliance": 0.70 if self.is_adhd else 0.80,
                "attention": 0.40 if self.is_adhd else 0.72,
                "escalation_risk": 0.20 if self.is_adhd else 0.05,
            }
            self.exhibited_behaviors: list[str] = []
            self.managed: bool = False
            self.managed_turns: int = 0
            self.identified: bool = False
            self.intervention_history: list[str] = []

        def step(self, context: Any, rng: random.Random) -> None:
            """Advance one cognitive cycle (stub: random walk)."""
            amp = 0.08 if self.is_adhd else 0.03
            self.state["distress_level"] = _clamp(
                self.state["distress_level"] + rng.gauss(0, amp)
            )
            self.state["compliance"] = _clamp(
                self.state["compliance"] + rng.gauss(0, amp)
            )
            self.state["attention"] = _clamp(
                self.state["attention"] + rng.gauss(0, amp)
            )
            self.state["escalation_risk"] = _clamp(
                self.state["escalation_risk"] + rng.gauss(0, amp)
            )

    class RelationshipGraph:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self._edges: dict[tuple[str, str], tuple[str, float]] = {}

        def add(self, a: str, b: str, rel_type: str, weight: float) -> None:
            key = (min(a, b), max(a, b))
            self._edges[key] = (rel_type, weight)

        def get(self, a: str, b: str) -> tuple[str, float] | None:
            key = (min(a, b), max(a, b))
            return self._edges.get(key)

        def get_related(self, sid: str, rel_type: str | None = None) -> list[tuple[str, str, float]]:
            results = []
            for (a, b), (rt, w) in self._edges.items():
                if rel_type and rt != rel_type:
                    continue
                if a == sid:
                    results.append((b, rt, w))
                elif b == sid:
                    results.append((a, rt, w))
            return results

    COGNITIVE_PRESETS: dict[str, Any] = {}  # type: ignore[no-redef]
    EMOTIONAL_PRESETS: dict[str, Any] = {}  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBJECTS = ["korean", "math", "science", "social", "art", "music", "pe", "moral"]


# ---------------------------------------------------------------------------
# Classroom Archetypes (Phase 2 enhancement)
# ---------------------------------------------------------------------------


@dataclass
class ClassroomArchetype:
    """Archetype that modifies classroom characteristics and student behavior."""

    name: str
    description: str
    adhd_prevalence_modifier: float = 0.0  # added to base prevalence
    noise_level: float = 0.5  # 0=very quiet, 1=very chaotic
    structure_level: float = 0.5  # 0=unstructured, 1=highly structured
    peer_conflict_modifier: float = 0.0  # added to base conflict rate
    teacher_support_available: bool = True  # counselor/aide available


CLASSROOM_ARCHETYPES: dict[str, ClassroomArchetype] = {
    "quiet_structured": ClassroomArchetype(
        name="quiet_structured",
        description="조용하고 구조화된 교실. 학습 규칙이 명확함.",
        noise_level=0.2,
        structure_level=0.8,
        peer_conflict_modifier=-0.05,
    ),
    "chaotic": ClassroomArchetype(
        name="chaotic",
        description="소란스러운 교실. 학생들이 자주 떠들고 이동함.",
        noise_level=0.8,
        structure_level=0.3,
        peer_conflict_modifier=0.10,
    ),
    "exam_period": ClassroomArchetype(
        name="exam_period",
        description="시험 기간. 스트레스 높고 조용하지만 긴장감 있음.",
        noise_level=0.3,
        structure_level=0.9,
        peer_conflict_modifier=0.05,
    ),
    "high_ses": ClassroomArchetype(
        name="high_ses",
        description="교육열 높은 지역. 학부모 관심 많고 자원 풍부.",
        noise_level=0.3,
        structure_level=0.7,
        teacher_support_available=True,
    ),
    "low_ses": ClassroomArchetype(
        name="low_ses",
        description="교육 자원 부족 지역. 교사 1인 담당, 지원 제한적.",
        adhd_prevalence_modifier=0.02,
        noise_level=0.6,
        structure_level=0.4,
        teacher_support_available=False,
    ),
    "mixed_grade": ClassroomArchetype(
        name="mixed_grade",
        description="복식 학급. 다양한 학년이 섞여있음.",
        noise_level=0.5,
        structure_level=0.4,
        peer_conflict_modifier=0.05,
    ),
}

# Management thresholds
MANAGED_COMPLIANCE = 0.80
MANAGED_CONSECUTIVE = 25  # ~1 week of sustained improvement

# ADHD behavior pools (mirrored from multi_student_env for consistency)
_ADHD_BEHAVIORS: dict[str, list[str]] = {
    "adhd_inattentive": [
        "daydreaming", "losing_materials", "forgetting_instructions",
        "staring_out_window", "not_starting_task", "off_task",
    ],
    "adhd_hyperactive_impulsive": [
        "out_of_seat", "calling_out", "interrupting",
        "excessive_talking", "fidgeting", "running_in_classroom",
    ],
    "adhd_combined": [
        "daydreaming", "out_of_seat", "calling_out", "off_task",
        "forgetting_instructions", "interrupting", "fidgeting",
    ],
}

_NORMAL_BEHAVIORS: list[str] = [
    "on_task", "listening", "writing", "whispering_briefly",
    "looking_around", "fidgeting_slightly",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TeacherAction:
    """What the teacher does this turn."""
    action_type: str  # observe, class_instruction, individual_intervention,
                      # private_correction, public_correction, identify_adhd,
                      # generate_report, wait
    student_id: str | None = None
    strategy: str | None = None
    reasoning: str = ""


@dataclass
class StudentSummary:
    """Visible-to-teacher summary for one student (partial observability)."""
    student_id: str
    profile_hint: str        # what the teacher can see (not the true label)
    behaviors: list[str]
    is_identified: bool
    is_managed: bool
    seat_row: int
    seat_col: int


@dataclass
class DetailedObservation:
    """Full observation for a focused student (max 3 per turn)."""
    student_id: str
    behaviors: list[str]
    state_snapshot: dict[str, float]
    emotional_cues: dict[str, float]
    recent_interactions: list[str]


@dataclass
class ClassroomObservation:
    """What the teacher sees each turn (partial observability)."""
    turn: int
    day: int
    period: int
    subject: str
    location: str

    student_summaries: list[StudentSummary]
    detailed_observations: list[DetailedObservation]

    class_mood: str
    identified_adhd_ids: list[str]
    managed_ids: list[str]


# ---------------------------------------------------------------------------
# InteractionEngine
# ---------------------------------------------------------------------------


class InteractionEngine:
    """Processes student-student interactions each turn."""

    def process_turn(
        self,
        students: list[CognitiveStudent],
        relationships: RelationshipGraph,
        context: ClassroomContext,
        rng: random.Random,
        class_id: int = 0,
    ) -> list[InteractionEvent]:
        events: list[InteractionEvent] = []

        events += self._peer_contagion(students, relationships, context, rng, class_id)
        events += self._friend_chatter(students, relationships, context, rng, class_id)
        events += self._conflicts(students, relationships, context, rng, class_id)
        events += self._bullying(students, relationships, context, rng, class_id)
        events += self._helping(students, relationships, context, rng, class_id)

        for event in events:
            self._apply_effects(event, students)

        return events

    # -- interaction helpers ------------------------------------------------

    def _activity_multiplier(self, context: ClassroomContext) -> float:
        """Higher interaction rates during recess/PE, lower during structured class."""
        subject = getattr(context, "subject", "korean")
        if subject == "pe":
            return 2.0
        if subject in ("art", "music"):
            return 1.3
        if subject in ("math", "korean"):
            return 0.6
        return 1.0

    def _get_neighbors(
        self, student_id: str, relationships: RelationshipGraph,
    ) -> list[str]:
        """Get seat neighbors via directed 'neighbor' edges in the graph."""
        return [
            rel.target_id
            for (src, _), rel in relationships._edges.items()
            if src == student_id and rel.type == "neighbor"
        ]

    def _peer_contagion(
        self,
        students: list[CognitiveStudent],
        relationships: RelationshipGraph,
        context: ClassroomContext,
        rng: random.Random,
        class_id: int = 0,
    ) -> list[InteractionEvent]:
        """Seat neighbors copy disruptive behavior."""
        events: list[InteractionEvent] = []
        mult = self._activity_multiplier(context)
        turn = getattr(context, "turn", 0)

        for student in students:
            if student.state.get("escalation_risk", 0) < 0.4:
                continue
            neighbor_ids = self._get_neighbors(student.student_id, relationships)
            for nid in neighbor_ids:
                neighbor = _find_student(students, nid)
                if neighbor is None:
                    continue
                prob = 0.12 * mult
                if rng.random() < prob:
                    events.append(InteractionEvent(
                        class_id=class_id,
                        turn=turn,
                        actor=student.student_id,
                        target=nid,
                        participants=[student.student_id, nid],
                        event_type="peer_contagion",
                        action="behavior_spread",
                        content=f"{student.student_id} disruptive behavior spreads to neighbor {nid}",
                        location=getattr(context, "location", "classroom"),
                        trigger="seat_proximity",
                        outcome="negative",
                    ))
        return events

    def _friend_chatter(
        self,
        students: list[CognitiveStudent],
        relationships: RelationshipGraph,
        context: ClassroomContext,
        rng: random.Random,
        class_id: int = 0,
    ) -> list[InteractionEvent]:
        """Friends chat during class, reducing attention for both."""
        events: list[InteractionEvent] = []
        mult = self._activity_multiplier(context)
        turn = getattr(context, "turn", 0)
        seen: set[tuple[str, str]] = set()

        for student in students:
            friend_ids = relationships.get_friends(student.student_id)
            for fid in friend_ids:
                pair = (min(student.student_id, fid), max(student.student_id, fid))
                if pair in seen:
                    continue
                seen.add(pair)
                rel = relationships.get(student.student_id, fid)
                weight = rel.strength if rel else 0.5
                prob = 0.08 * mult * weight
                if rng.random() < prob:
                    events.append(InteractionEvent(
                        class_id=class_id,
                        turn=turn,
                        actor=student.student_id,
                        target=fid,
                        participants=[student.student_id, fid],
                        event_type="peer_chat",
                        action="friend_chatter",
                        content=f"{student.student_id} and {fid} chatting",
                        location=getattr(context, "location", "classroom"),
                        trigger="friendship",
                        outcome="neutral",
                    ))
        return events

    def _conflicts(
        self,
        students: list[CognitiveStudent],
        relationships: RelationshipGraph,
        context: ClassroomContext,
        rng: random.Random,
        class_id: int = 0,
    ) -> list[InteractionEvent]:
        """Students in conflict relationships may clash."""
        events: list[InteractionEvent] = []
        mult = self._activity_multiplier(context)
        turn = getattr(context, "turn", 0)
        seen: set[tuple[str, str]] = set()

        for student in students:
            conflict_ids = relationships.get_conflicts(student.student_id)
            for cid in conflict_ids:
                pair = (min(student.student_id, cid), max(student.student_id, cid))
                if pair in seen:
                    continue
                seen.add(pair)
                rel = relationships.get(student.student_id, cid)
                weight = rel.strength if rel else 0.4
                prob = 0.06 * mult * weight
                prob += student.state.get("escalation_risk", 0) * 0.05
                if rng.random() < prob:
                    events.append(InteractionEvent(
                        class_id=class_id,
                        turn=turn,
                        actor=student.student_id,
                        target=cid,
                        participants=[student.student_id, cid],
                        event_type="peer_conflict",
                        action="verbal_conflict",
                        content=f"Conflict between {student.student_id} and {cid}",
                        location=getattr(context, "location", "classroom"),
                        trigger="relationship_tension",
                        outcome="negative",
                    ))
        return events

    def _bullying(
        self,
        students: list[CognitiveStudent],
        relationships: RelationshipGraph,
        context: ClassroomContext,
        rng: random.Random,
        class_id: int = 0,
    ) -> list[InteractionEvent]:
        """ODD students may bully anxious/quiet students (KCI ART003153213)."""
        events: list[InteractionEvent] = []
        mult = self._activity_multiplier(context)
        turn = getattr(context, "turn", 0)

        aggressors = [s for s in students if s.profile_type == "odd"]
        targets = [s for s in students if s.profile_type in ("anxiety", "normal_quiet")]

        for agg in aggressors:
            for tgt in targets:
                if agg.student_id == tgt.student_id:
                    continue
                prob = 0.04 * mult
                prob += agg.state.get("escalation_risk", 0) * 0.08
                if rng.random() < prob:
                    events.append(InteractionEvent(
                        class_id=class_id,
                        turn=turn,
                        actor=agg.student_id,
                        target=tgt.student_id,
                        participants=[agg.student_id, tgt.student_id],
                        event_type="peer_bullying",
                        action="verbal_bullying",
                        content=f"{agg.student_id} bullying {tgt.student_id}",
                        location=getattr(context, "location", "classroom"),
                        trigger="odd_aggression",
                        outcome="negative",
                    ))
        return events

    def _helping(
        self,
        students: list[CognitiveStudent],
        relationships: RelationshipGraph,
        context: ClassroomContext,
        rng: random.Random,
        class_id: int = 0,
    ) -> list[InteractionEvent]:
        """Gifted students may help struggling peers."""
        events: list[InteractionEvent] = []
        mult = self._activity_multiplier(context)
        turn = getattr(context, "turn", 0)

        helpers = [s for s in students if s.profile_type == "gifted"]
        struggling = [s for s in students if s.state.get("attention", 1.0) < 0.4]

        for helper in helpers:
            for target in struggling:
                if helper.student_id == target.student_id:
                    continue
                prob = 0.10 * mult
                if rng.random() < prob:
                    events.append(InteractionEvent(
                        class_id=class_id,
                        turn=turn,
                        actor=helper.student_id,
                        target=target.student_id,
                        participants=[helper.student_id, target.student_id],
                        event_type="peer_help",
                        action="academic_help",
                        content=f"{helper.student_id} helping {target.student_id}",
                        location=getattr(context, "location", "classroom"),
                        trigger="gifted_prosocial",
                        outcome="positive",
                    ))
        return events

    def _apply_effects(
        self,
        event: InteractionEvent,
        students: list[CognitiveStudent],
    ) -> None:
        """Apply interaction effects to both students' emotional/cognitive states."""
        actor = _find_student(students, event.actor)
        target = _find_student(students, event.target)

        if event.event_type == "peer_contagion":
            if target:
                target.state["attention"] = _clamp(target.state.get("attention", 0.5) - 0.05)
                target.state["escalation_risk"] = _clamp(target.state.get("escalation_risk", 0.1) + 0.03)

        elif event.event_type == "peer_chat":
            if actor:
                actor.state["attention"] = _clamp(actor.state.get("attention", 0.5) - 0.03)
            if target:
                target.state["attention"] = _clamp(target.state.get("attention", 0.5) - 0.03)

        elif event.event_type == "peer_conflict":
            if actor:
                actor.state["distress_level"] = _clamp(actor.state.get("distress_level", 0.2) + 0.08)
                actor.state["escalation_risk"] = _clamp(actor.state.get("escalation_risk", 0.1) + 0.06)
            if target:
                target.state["distress_level"] = _clamp(target.state.get("distress_level", 0.2) + 0.10)
                target.state["escalation_risk"] = _clamp(target.state.get("escalation_risk", 0.1) + 0.04)

        elif event.event_type == "peer_bullying":
            if target:
                target.state["distress_level"] = _clamp(target.state.get("distress_level", 0.2) + 0.15)
                target.state["compliance"] = _clamp(target.state.get("compliance", 0.7) - 0.05)
                target.state["attention"] = _clamp(target.state.get("attention", 0.5) - 0.08)
            if actor:
                actor.state["escalation_risk"] = _clamp(actor.state.get("escalation_risk", 0.3) + 0.03)

        elif event.event_type == "peer_help":
            if target:
                target.state["attention"] = _clamp(target.state.get("attention", 0.3) + 0.06)
                target.state["distress_level"] = _clamp(target.state.get("distress_level", 0.3) - 0.04)
            if actor:
                actor.state["compliance"] = _clamp(actor.state.get("compliance", 0.8) + 0.02)


# ---------------------------------------------------------------------------
# ClassroomV2 — 950-turn environment
# ---------------------------------------------------------------------------


class ClassroomV2:
    """
    950-turn classroom environment (1 academic year).

    Episode flow:
      1. reset() generates N students with relationships.
      2. step(action) advances one period: cognitive cycles, interactions, rewards.
      3. done when turn >= 950 or externally stopped.
    """

    def __init__(
        self,
        n_students: int = 20,
        adhd_prevalence: tuple[float, float] | float | None = None,
        seed: int | None = None,
        interaction_log: InteractionLog | None = None,
        archetype: str | None = None,
    ):
        self.n_students = n_students
        self.adhd_prevalence: tuple[float, float] | float = adhd_prevalence or (0.06, 0.11)
        self.students: list[CognitiveStudent] = []
        self.relationships: RelationshipGraph = RelationshipGraph()
        self.interaction_engine: InteractionEngine = InteractionEngine()
        self.log: InteractionLog = interaction_log or InteractionLog()
        self.class_id: int = 0

        # Time tracking: 190 days x 5 periods = 950 turns
        self.turn: int = 0
        self.day: int = 1
        self.period: int = 1
        self.PERIODS_PER_DAY: int = 5
        self.TOTAL_DAYS: int = 190
        self.MAX_TURNS: int = self.TOTAL_DAYS * self.PERIODS_PER_DAY  # 950

        # Schedule: generated per day
        self.daily_schedule: list[str] = []

        # Tracking sets
        self.identified_adhd_ids: set[str] = set()
        self.managed_ids: set[str] = set()

        self._rng = random.Random(seed)

        # Classroom archetype (Phase 2 enhancement)
        self._archetype_name: str | None = archetype
        self.archetype: ClassroomArchetype | None = None
        if archetype is not None:
            self.set_archetype(archetype)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_archetype(self, name: str) -> None:
        """Set the classroom archetype by name."""
        if name not in CLASSROOM_ARCHETYPES:
            raise ValueError(
                f"Unknown archetype '{name}'. "
                f"Available: {list(CLASSROOM_ARCHETYPES.keys())}"
            )
        self._archetype_name = name
        self.archetype = CLASSROOM_ARCHETYPES[name]

    def reset(self) -> ClassroomObservation:
        """Generate a new class of students with relationships."""
        self.class_id += 1
        self.turn = 0
        self.day = 1
        self.period = 1
        self.identified_adhd_ids = set()
        self.managed_ids = set()

        # Pick archetype: use fixed one if set, otherwise random
        if self._archetype_name is not None:
            self.set_archetype(self._archetype_name)
        else:
            name = self._rng.choice(list(CLASSROOM_ARCHETYPES.keys()))
            self.set_archetype(name)

        self.students = self._generate_students()
        self._apply_archetype_effects()
        self.relationships = self._generate_relationships()
        self.daily_schedule = self._generate_daily_schedule()
        return self._make_observation()

    def step(
        self, teacher_action: TeacherAction
    ) -> tuple[ClassroomObservation, float, bool, dict[str, Any]]:
        """Advance one turn (= one class period)."""
        self.turn += 1
        self._advance_time()

        context = self._make_context(teacher_action)

        # Phase 1: Each student runs cognitive cycle
        for student in self.students:
            student.step(context, self._rng)

        # Phase 2: Student interactions
        interactions = self.interaction_engine.process_turn(
            self.students, self.relationships, context, self._rng,
            class_id=self.class_id,
        )

        # Phase 3: Log all events
        for event in interactions:
            self.log.record(event)

        # Phase 4: Apply teacher action effects
        reward = self._apply_teacher_action(teacher_action, context)

        # Phase 5: Check managed status
        self._check_managed_status()

        # Phase 6: Done check
        done = self.turn >= self.MAX_TURNS

        obs = self._make_observation(focused_id=teacher_action.student_id)
        info: dict[str, Any] = {
            "turn": self.turn,
            "day": self.day,
            "period": self.period,
            "subject": self._current_subject(),
            "location": self._current_location(),
            "interactions": interactions,
            "reward": reward,
        }
        return obs, reward, done, info

    def is_class_complete(self) -> bool:
        """True when every ADHD student is both identified and managed."""
        adhd = [s for s in self.students if s.is_adhd]
        if not adhd:
            return True
        return all(s.identified and s.managed for s in adhd)

    def ground_truth_adhd_ids(self) -> list[str]:
        return [s.student_id for s in self.students if s.is_adhd]

    def get_student(self, student_id: str) -> CognitiveStudent | None:
        return _find_student(self.students, student_id)

    # ------------------------------------------------------------------
    # Time management
    # ------------------------------------------------------------------

    def _advance_time(self) -> None:
        """Advance day/period counters based on turn number."""
        # turn is 1-indexed after increment
        self.period = ((self.turn - 1) % self.PERIODS_PER_DAY) + 1
        self.day = ((self.turn - 1) // self.PERIODS_PER_DAY) + 1

        # Generate new schedule each day (period 1)
        if self.period == 1:
            self.daily_schedule = self._generate_daily_schedule()

    def _current_subject(self) -> str:
        if not self.daily_schedule:
            return "unknown"
        return self.daily_schedule[(self.period - 1) % len(self.daily_schedule)]

    def _current_location(self) -> str:
        # Transition between days: first turn of a new day is hallway
        if self.turn > 1 and self.period == 1:
            return "hallway"
        subject = self._current_subject()
        if subject == "pe":
            return "playground"
        return "classroom"

    # ------------------------------------------------------------------
    # Student generation (Korean epidemiological distribution)
    # ------------------------------------------------------------------

    def _generate_students(self) -> list[CognitiveStudent]:
        """Generate students with realistic Korean prevalence distribution."""
        students: list[CognitiveStudent] = []

        # ADHD count (archetype may modify prevalence)
        if isinstance(self.adhd_prevalence, tuple):
            rate = self._rng.uniform(*self.adhd_prevalence)
        else:
            rate = self.adhd_prevalence
        if self.archetype is not None:
            rate = max(0.0, rate + self.archetype.adhd_prevalence_modifier)
        n_adhd = max(0, round(self.n_students * rate))

        # Confounding profile counts (Korean comorbidity data PMC5290097)
        n_anxiety = round(self.n_students * self._rng.uniform(0.05, 0.08))
        n_odd = round(self.n_students * self._rng.uniform(0.03, 0.05))
        n_gifted = round(self.n_students * self._rng.uniform(0.03, 0.05))
        n_sleep = round(self.n_students * self._rng.uniform(0.05, 0.10))

        # Cap total special profiles to avoid exceeding n_students
        n_special = n_adhd + n_anxiety + n_odd + n_gifted + n_sleep
        if n_special > self.n_students:
            scale = (self.n_students * 0.5) / n_special
            n_anxiety = round(n_anxiety * scale)
            n_odd = round(n_odd * scale)
            n_gifted = round(n_gifted * scale)
            n_sleep = round(n_sleep * scale)

        n_normal = self.n_students - n_adhd - n_anxiety - n_odd - n_gifted - n_sleep
        if n_normal < 0:
            n_normal = 0

        # Build ADHD subtype split: combined 24%, HI 24%, inattentive 52%
        n_combined = round(n_adhd * 0.24)
        n_hi = round(n_adhd * 0.24)
        n_inattentive = n_adhd - n_combined - n_hi

        profiles: list[str] = (
            ["adhd_combined"] * n_combined
            + ["adhd_hyperactive_impulsive"] * n_hi
            + ["adhd_inattentive"] * n_inattentive
            + ["anxiety"] * n_anxiety
            + ["odd"] * n_odd
            + ["gifted"] * n_gifted
            + ["sleep_deprived"] * n_sleep
            + ["normal_active"] * round(n_normal * 0.4)
            + ["normal_quiet"] * (n_normal - round(n_normal * 0.4))
        )
        self._rng.shuffle(profiles)

        # Trim or pad to exact n_students
        profiles = profiles[: self.n_students]
        while len(profiles) < self.n_students:
            profiles.append("normal_quiet")

        for i, profile in enumerate(profiles):
            sid = f"S{i + 1:02d}"

            # Gender: ADHD boys 3.19x (KCI ART001701933)
            if profile.startswith("adhd_"):
                gender = "male" if self._rng.random() < 0.761 else "female"
            else:
                gender = "male" if self._rng.random() < 0.5 else "female"

            age = self._rng.randint(7, 12)
            severity = None
            if profile.startswith("adhd_"):
                severity = self._rng.choices(
                    ["mild", "moderate", "severe"], weights=[0.45, 0.35, 0.20]
                )[0]

            student = CognitiveStudent(
                student_id=sid,
                profile_type=profile,
                age=age,
                gender=gender,
                severity=severity,
            )
            # Ensure fields that classroom_env_v2 depends on exist,
            # regardless of what cognitive_agent.py defines.
            if not hasattr(student, "identified"):
                student.identified = False  # type: ignore[attr-defined]
            if not hasattr(student, "intervention_history"):
                student.intervention_history = []  # type: ignore[attr-defined]
            if not hasattr(student, "managed"):
                student.managed = False  # type: ignore[attr-defined]
            if not hasattr(student, "managed_turns"):
                student.managed_turns = 0  # type: ignore[attr-defined]
            if not hasattr(student, "exhibited_behaviors"):
                student.exhibited_behaviors = []  # type: ignore[attr-defined]
            students.append(student)

        return students

    def _apply_archetype_effects(self) -> None:
        """Apply archetype modifiers to student baselines after generation."""
        if self.archetype is None:
            return

        arch = self.archetype

        for student in self.students:
            # Noise level affects attention baseline and escalation risk
            noise_delta = (arch.noise_level - 0.5) * 0.15
            student.state["attention"] = _clamp(
                student.state.get("attention", 0.5) - noise_delta
            )
            student.state["escalation_risk"] = _clamp(
                student.state.get("escalation_risk", 0.1) + noise_delta * 0.5
            )

            # Structure level improves compliance baseline
            structure_delta = (arch.structure_level - 0.5) * 0.10
            student.state["compliance"] = _clamp(
                student.state.get("compliance", 0.7) + structure_delta
            )

            # Exam period increases stress/distress
            if arch.name == "exam_period":
                student.state["distress_level"] = _clamp(
                    student.state.get("distress_level", 0.15) + 0.10
                )

            # Low SES: no teacher support means higher baseline distress
            if not arch.teacher_support_available:
                student.state["distress_level"] = _clamp(
                    student.state.get("distress_level", 0.15) + 0.05
                )

            # Emotional baselines: chaotic classrooms increase arousal
            if hasattr(student, "emotions") and hasattr(student.emotions, "arousal"):
                student.emotions.arousal = _clamp(
                    student.emotions.arousal + (arch.noise_level - 0.5) * 0.1
                )
                student.emotions.stress = _clamp(
                    student.emotions.stress + (1.0 - arch.structure_level) * 0.05
                )

    # ------------------------------------------------------------------
    # Relationship generation
    # ------------------------------------------------------------------

    def _generate_relationships(self) -> RelationshipGraph:
        """Generate realistic peer relationships (directed edges, both directions)."""
        graph = RelationshipGraph()

        for i, s1 in enumerate(self.students):
            for j, s2 in enumerate(self.students):
                if i >= j:
                    continue

                # Friend probability
                friend_prob = 0.15
                if s1.profile_type == s2.profile_type:
                    friend_prob += 0.1

                # Conflict probability (archetype modifies base rate)
                conflict_prob = 0.05
                if self.archetype is not None:
                    conflict_prob = max(0.0, conflict_prob + self.archetype.peer_conflict_modifier)
                if s1.profile_type == "odd":
                    conflict_prob += 0.15
                if s2.profile_type == "odd":
                    conflict_prob += 0.15
                # ODD + anxiety = potential bully dynamic (KCI ART003153213)
                if s1.profile_type == "odd" and s2.profile_type == "anxiety":
                    conflict_prob += 0.10
                if s2.profile_type == "odd" and s1.profile_type == "anxiety":
                    conflict_prob += 0.10

                r = self._rng.random()
                if r < friend_prob:
                    strength = self._rng.uniform(0.3, 0.8)
                    # Symmetric: both directions
                    graph.add(s1.student_id, s2.student_id, "friend", strength)
                    graph.add(s2.student_id, s1.student_id, "friend", strength)
                elif r < friend_prob + conflict_prob:
                    strength = self._rng.uniform(0.2, 0.6)
                    graph.add(s1.student_id, s2.student_id, "conflict", strength)
                    graph.add(s2.student_id, s1.student_id, "conflict", strength)

        # Seat neighbors: 5-column grid, neighbors = left/right/front/back
        cols = 5
        for i, student in enumerate(self.students):
            row, col = divmod(i, cols)
            neighbor_indices: list[int] = []
            if col > 0:
                neighbor_indices.append(i - 1)
            if col < cols - 1 and i + 1 < len(self.students):
                neighbor_indices.append(i + 1)
            if row > 0:
                neighbor_indices.append(i - cols)
            max_row = (len(self.students) - 1) // cols
            if row < max_row and i + cols < len(self.students):
                neighbor_indices.append(i + cols)
            for ni in neighbor_indices:
                # Directed: add both directions
                graph.add(student.student_id, self.students[ni].student_id, "neighbor", 1.0)

        return graph

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def _generate_daily_schedule(self) -> list[str]:
        """Generate 5-period daily schedule from Korean elementary subjects."""
        return self._rng.sample(SUBJECTS, min(self.PERIODS_PER_DAY, len(SUBJECTS)))

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _make_context(self, teacher_action: TeacherAction) -> ClassroomContext:
        # Compute class mood for context
        avg_distress = _mean([s.state.get("distress_level", 0.1) for s in self.students])
        avg_attention = _mean([s.state.get("attention", 0.5) for s in self.students])
        if avg_distress > 0.5:
            mood = "tense"
        elif avg_attention > 0.6:
            mood = "calm"
        elif avg_attention < 0.35:
            mood = "chaotic"
        else:
            mood = "calm"

        try:
            return ClassroomContext(
                turn=self.turn,
                period=self.period,
                day=self.day,
                subject=self._current_subject(),
                location=self._current_location(),
                current_events=[],
                class_mood=mood,
                teacher_action=teacher_action.action_type,
                teacher_target=teacher_action.student_id,
            )
        except TypeError:
            # Fallback for stub ClassroomContext (accepts **kwargs)
            return ClassroomContext(
                class_id=self.class_id,
                turn=self.turn,
                day=self.day,
                period=self.period,
                subject=self._current_subject(),
                location=self._current_location(),
                teacher_action_type=teacher_action.action_type,
                teacher_target=teacher_action.student_id,
            )

    # ------------------------------------------------------------------
    # Teacher action handling
    # ------------------------------------------------------------------

    def _apply_teacher_action(
        self, action: TeacherAction, context: ClassroomContext
    ) -> float:
        reward = 0.0

        if action.action_type == "observe":
            reward += 0.1

        elif action.action_type == "class_instruction":
            for student in self.students:
                student.state["compliance"] = _clamp(
                    student.state.get("compliance", 0.7) + self._rng.uniform(0.0, 0.05)
                )
                student.state["attention"] = _clamp(
                    student.state.get("attention", 0.5) + self._rng.uniform(0.0, 0.04)
                )
            reward += 0.05

        elif action.action_type == "individual_intervention":
            student = self.get_student(action.student_id or "")
            if student and action.strategy:
                student.intervention_history.append(action.strategy)
                student.state["distress_level"] = _clamp(
                    student.state.get("distress_level", 0.3) - self._rng.uniform(0.0, 0.10)
                )
                student.state["compliance"] = _clamp(
                    student.state.get("compliance", 0.5) + self._rng.uniform(0.0, 0.12)
                )
                student.state["attention"] = _clamp(
                    student.state.get("attention", 0.4) + self._rng.uniform(0.0, 0.10)
                )
                student.state["escalation_risk"] = _clamp(
                    student.state.get("escalation_risk", 0.2) - self._rng.uniform(0.0, 0.08)
                )
                reward += 0.3 if student.is_adhd else 0.1
            else:
                reward -= 0.1

        elif action.action_type == "private_correction":
            student = self.get_student(action.student_id or "")
            if student:
                student.state["distress_level"] = _clamp(
                    student.state.get("distress_level", 0.3) - self._rng.uniform(0.05, 0.15)
                )
                student.state["compliance"] = _clamp(
                    student.state.get("compliance", 0.5) + self._rng.uniform(0.08, 0.18)
                )
                student.state["attention"] = _clamp(
                    student.state.get("attention", 0.4) + self._rng.uniform(0.05, 0.12)
                )
                student.state["escalation_risk"] = _clamp(
                    student.state.get("escalation_risk", 0.2) - self._rng.uniform(0.06, 0.14)
                )
                student.intervention_history.append("private_correction")
                reward += 0.4 if student.is_adhd else 0.15
            else:
                reward -= 0.1

        elif action.action_type == "public_correction":
            student = self.get_student(action.student_id or "")
            if student:
                # O'Leary 1970: public correction less effective, may increase distress
                distress_delta = (
                    self._rng.uniform(-0.02, 0.12)
                    if student.is_adhd
                    else self._rng.uniform(-0.02, 0.05)
                )
                student.state["distress_level"] = _clamp(
                    student.state.get("distress_level", 0.3) + distress_delta
                )
                student.state["compliance"] = _clamp(
                    student.state.get("compliance", 0.5) + self._rng.uniform(-0.02, 0.10)
                )
                student.state["escalation_risk"] = _clamp(
                    student.state.get("escalation_risk", 0.2) + self._rng.uniform(-0.02, 0.08)
                )
                student.intervention_history.append("public_correction")
                reward += -0.05 if student.is_adhd else 0.05
            else:
                reward -= 0.1

        elif action.action_type == "identify_adhd":
            student = self.get_student(action.student_id or "")
            if student and action.student_id:
                if action.student_id not in self.identified_adhd_ids:
                    self.identified_adhd_ids.add(action.student_id)
                    student.identified = True
                    reward += 1.0 if student.is_adhd else -1.0

        elif action.action_type == "generate_report":
            student = self.get_student(action.student_id or "")
            if student:
                reward += 0.5 if (student.is_adhd and student.identified) else -0.2

        elif action.action_type == "wait":
            reward += 0.0

        return reward

    # ------------------------------------------------------------------
    # Managed status + relapse
    # ------------------------------------------------------------------

    def _check_managed_status(self) -> None:
        for student in self.students:
            if not student.is_adhd:
                continue
            if student.state.get("compliance", 0) >= MANAGED_COMPLIANCE:
                student.managed_turns += 1
            else:
                student.managed_turns = max(0, student.managed_turns - 2)  # relapse penalty
            student.managed = student.managed_turns >= MANAGED_CONSECUTIVE
            if student.managed:
                self.managed_ids.add(student.student_id)

    # ------------------------------------------------------------------
    # Observation builder (partial observability)
    # ------------------------------------------------------------------

    def _make_observation(
        self, focused_id: str | None = None
    ) -> ClassroomObservation:
        summaries: list[StudentSummary] = []
        detailed: list[DetailedObservation] = []
        cols = 5

        for i, student in enumerate(self.students):
            row, col = divmod(i, cols)

            # Profile hint: teacher sees behavior-based hints, not true labels
            if student.is_adhd and student.identified:
                hint = "identified_adhd"
            elif student.state.get("escalation_risk", 0) > 0.5:
                hint = "disruptive"
            elif student.state.get("attention", 1.0) < 0.3:
                hint = "inattentive"
            else:
                hint = "typical"

            # High-visibility behaviors only (disruptive ones visible from teacher desk)
            visible_behaviors = self._visible_behaviors(student)

            summaries.append(StudentSummary(
                student_id=student.student_id,
                profile_hint=hint,
                behaviors=visible_behaviors,
                is_identified=student.identified,
                is_managed=student.managed,
                seat_row=row,
                seat_col=col,
            ))

            # Detailed observation for focused student
            if focused_id and student.student_id == focused_id:
                emotions = (
                    student.emotions.to_dict()
                    if hasattr(student, "emotions") and hasattr(student.emotions, "to_dict")
                    else {}
                )
                recent = [
                    e.content
                    for e in self.log.get_student_history(student.student_id, self.class_id)[-3:]
                ]
                detailed.append(DetailedObservation(
                    student_id=student.student_id,
                    behaviors=list(student.exhibited_behaviors) if hasattr(student, "exhibited_behaviors") else visible_behaviors,
                    state_snapshot=dict(student.state),
                    emotional_cues=emotions,
                    recent_interactions=recent,
                ))

        # Class mood aggregate
        avg_distress = _mean([s.state.get("distress_level", 0.1) for s in self.students])
        avg_attention = _mean([s.state.get("attention", 0.5) for s in self.students])
        if avg_distress > 0.5:
            class_mood = "tense"
        elif avg_attention > 0.6:
            class_mood = "focused"
        elif avg_attention < 0.35:
            class_mood = "distracted"
        else:
            class_mood = "neutral"

        return ClassroomObservation(
            turn=self.turn,
            day=self.day,
            period=self.period,
            subject=self._current_subject(),
            location=self._current_location(),
            student_summaries=summaries,
            detailed_observations=detailed,
            class_mood=class_mood,
            identified_adhd_ids=sorted(self.identified_adhd_ids),
            managed_ids=sorted(self.managed_ids),
        )

    def _visible_behaviors(self, student: CognitiveStudent) -> list[str]:
        """Return only high-visibility behaviors the teacher can see from the front."""
        high_vis = {
            "out_of_seat", "calling_out", "interrupting", "excessive_talking",
            "running_in_classroom", "fidgeting", "emotional_outburst",
        }
        behaviors = getattr(student, "exhibited_behaviors", [])
        visible = [b for b in behaviors if b in high_vis]
        # If nothing high-vis, show a generic label
        if not visible:
            if student.state.get("attention", 1.0) < 0.3:
                visible = ["seems_inattentive"]
            elif student.state.get("compliance", 1.0) > 0.7:
                visible = ["on_task"]
            else:
                visible = ["quiet"]
        return visible


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _find_student(
    students: list[CognitiveStudent], student_id: str
) -> CognitiveStudent | None:
    for s in students:
        if s.student_id == student_id:
            return s
    return None


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
