"""
Generative Agents cognitive architecture for ADHD classroom simulation.

Implements the Stanford Generative Agents perceive-retrieve-reflect-plan-act
cycle (Park et al., 2023) adapted for Korean elementary classroom students.
ADHD and other conditions are modeled as parameter differences in the SAME
cognitive architecture, not as separate agent types.

Cognitive parameter sources:
  - Attention bandwidth: DSM-5 ADHD diagnostic criteria (APA, 2013)
  - Recency decay: Generative Agents memory scoring (Park et al., 2023)
  - Impulse override / plan consistency: Barkley's executive function model (1997)
  - Social sensitivity: Korean peer interaction studies (KCI ART002478306)

Emotional preset sources:
  - K-CBCL cluster analysis (KCI ART002650863)
  - Korean ADHD emotional dysregulation profiles (KCI ART002794420)

Behavioral categories from Korean research:
  KCI ART002794420 - 과잉행동/충동성/부주의 행동 관찰 척도
  KCI ART002478306 - ADHD 아동 교실 행동 특성 연구
"""

from __future__ import annotations

import math
import random
from copy import copy
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Cognitive Parameters
# ---------------------------------------------------------------------------


@dataclass
class CognitiveParameters:
    """Cognitive parameters that define student type. Same architecture, different values.

    All students share this structure. ADHD, anxiety, ODD, gifted, and normal
    students differ only in parameter values, not in the cognitive cycle itself.
    """

    # -- Perception --
    att_bandwidth: int = 3
    """How many events can be attended to simultaneously (DSM-5: ADHD reduces this)."""

    vision_r: int = 4
    """How wide attention scans. Higher = more distractible / broader awareness."""

    # -- Memory --
    retention: int = 5
    """How many recent events stay in working memory (Baddeley working memory model)."""

    recency_decay: float = 0.99
    """How fast memories fade. Lower = faster fade (Park et al., 2023 decay formula)."""

    # -- Reflection --
    importance_trigger: float = 150.0
    """Accumulated poignancy before reflection triggers (Generative Agents threshold)."""

    # -- Planning & Control --
    plan_consistency: float = 0.90
    """Probability of following current plan vs switching (Barkley 1997 executive fn)."""

    impulse_override: float = 0.05
    """Probability of impulsive action bypassing plan entirely (0-1)."""

    task_initiation_delay: float = 0.0
    """Extra delay before starting tasks (0-1). ADHD-inattentive: high value."""

    # -- Social --
    social_sensitivity: float = 0.5
    """How much peer actions affect this student (Korean peer studies KCI ART002478306)."""

    conflict_tendency: float = 0.1
    """Probability of initiating conflict with peers."""


# ---------------------------------------------------------------------------
# Cognitive Presets (one per student type)
# ---------------------------------------------------------------------------

COGNITIVE_PRESETS: dict[str, CognitiveParameters] = {
    # Typical quiet student: high plan consistency, low impulsivity
    "normal_quiet": CognitiveParameters(
        att_bandwidth=3, vision_r=4, retention=5, recency_decay=0.99,
        importance_trigger=150.0, plan_consistency=0.90, impulse_override=0.05,
        task_initiation_delay=0.0, social_sensitivity=0.3, conflict_tendency=0.05,
    ),
    # Typical active student: slightly more distractible, more social
    "normal_active": CognitiveParameters(
        att_bandwidth=3, vision_r=5, retention=5, recency_decay=0.98,
        importance_trigger=120.0, plan_consistency=0.80, impulse_override=0.10,
        task_initiation_delay=0.0, social_sensitivity=0.6, conflict_tendency=0.10,
    ),
    # ADHD predominantly inattentive: narrow bandwidth, fast decay, high initiation delay
    # DSM-5 314.00; Korean prevalence 5.0% (KCI ART001701933)
    "adhd_inattentive": CognitiveParameters(
        att_bandwidth=1, vision_r=3, retention=2, recency_decay=0.95,
        importance_trigger=50.0, plan_consistency=0.50, impulse_override=0.15,
        task_initiation_delay=0.4, social_sensitivity=0.4, conflict_tendency=0.08,
    ),
    # ADHD predominantly hyperactive-impulsive: wide scan, high impulse override
    # DSM-5 314.01; Korean prevalence 2.3% (KCI ART001701933)
    "adhd_hyperactive_impulsive": CognitiveParameters(
        att_bandwidth=2, vision_r=6, retention=3, recency_decay=0.96,
        importance_trigger=40.0, plan_consistency=0.40, impulse_override=0.40,
        task_initiation_delay=0.1, social_sensitivity=0.7, conflict_tendency=0.25,
    ),
    # ADHD combined type: worst of both worlds
    # DSM-5 314.01; Korean prevalence 2.3% (KCI ART001701933)
    "adhd_combined": CognitiveParameters(
        att_bandwidth=1, vision_r=6, retention=2, recency_decay=0.94,
        importance_trigger=35.0, plan_consistency=0.35, impulse_override=0.35,
        task_initiation_delay=0.3, social_sensitivity=0.6, conflict_tendency=0.20,
    ),
    # Generalized anxiety: narrow focus, high social sensitivity, low conflict
    "anxiety": CognitiveParameters(
        att_bandwidth=2, vision_r=3, retention=4, recency_decay=0.98,
        importance_trigger=50.0, plan_consistency=0.75, impulse_override=0.10,
        task_initiation_delay=0.2, social_sensitivity=0.8, conflict_tendency=0.03,
    ),
    # Oppositional Defiant Disorder: high conflict, low plan following
    "odd": CognitiveParameters(
        att_bandwidth=2, vision_r=4, retention=4, recency_decay=0.97,
        importance_trigger=80.0, plan_consistency=0.40, impulse_override=0.35,
        task_initiation_delay=0.1, social_sensitivity=0.5, conflict_tendency=0.40,
    ),
    # Gifted: high bandwidth, high retention, moderate impulse (boredom-driven)
    "gifted": CognitiveParameters(
        att_bandwidth=4, vision_r=5, retention=6, recency_decay=0.99,
        importance_trigger=200.0, plan_consistency=0.85, impulse_override=0.15,
        task_initiation_delay=0.0, social_sensitivity=0.4, conflict_tendency=0.08,
    ),
    # Sleep-deprived: reduced everything, sluggish initiation
    "sleep_deprived": CognitiveParameters(
        att_bandwidth=2, vision_r=3, retention=3, recency_decay=0.96,
        importance_trigger=100.0, plan_consistency=0.60, impulse_override=0.15,
        task_initiation_delay=0.3, social_sensitivity=0.3, conflict_tendency=0.12,
    ),
}


# ---------------------------------------------------------------------------
# Emotional State
# ---------------------------------------------------------------------------


@dataclass
class EmotionalState:
    """8-dimension dynamic emotional state.

    Values in [0, 1]. Updated each turn based on events and interactions.
    Source: K-CBCL cluster analysis (KCI ART002650863).
    """

    frustration: float = 0.15
    shame: float = 0.10
    anxiety: float = 0.15
    anger: float = 0.10
    loneliness: float = 0.10
    excitement: float = 0.20
    trust_in_teacher: float = 0.60
    self_esteem: float = 0.65


# Emotional presets derived from K-CBCL cluster profiles (KCI ART002650863).
# Each maps to an EmotionalState constructor kwargs dict.
EMOTIONAL_PRESETS: dict[str, dict[str, float]] = {
    "normal_quiet": dict(
        frustration=0.10, shame=0.08, anxiety=0.12, anger=0.05,
        loneliness=0.08, excitement=0.15, trust_in_teacher=0.70, self_esteem=0.72,
    ),
    "normal_active": dict(
        frustration=0.12, shame=0.08, anxiety=0.10, anger=0.10,
        loneliness=0.06, excitement=0.30, trust_in_teacher=0.65, self_esteem=0.70,
    ),
    # ADHD-inattentive: higher shame/frustration from repeated academic failure
    "adhd_inattentive": dict(
        frustration=0.30, shame=0.25, anxiety=0.25, anger=0.10,
        loneliness=0.20, excitement=0.10, trust_in_teacher=0.50, self_esteem=0.45,
    ),
    # ADHD-hyperactive: high excitement, low frustration tolerance, moderate anger
    "adhd_hyperactive_impulsive": dict(
        frustration=0.25, shame=0.15, anxiety=0.15, anger=0.25,
        loneliness=0.12, excitement=0.40, trust_in_teacher=0.45, self_esteem=0.50,
    ),
    # ADHD-combined: worst emotional profile across multiple dimensions
    "adhd_combined": dict(
        frustration=0.35, shame=0.25, anxiety=0.20, anger=0.25,
        loneliness=0.18, excitement=0.35, trust_in_teacher=0.40, self_esteem=0.40,
    ),
    # Anxiety: high anxiety/shame, low anger, high trust-seeking
    "anxiety": dict(
        frustration=0.15, shame=0.30, anxiety=0.45, anger=0.05,
        loneliness=0.25, excitement=0.08, trust_in_teacher=0.55, self_esteem=0.35,
    ),
    # ODD: high anger/frustration, low trust, low shame
    "odd": dict(
        frustration=0.40, shame=0.08, anxiety=0.12, anger=0.45,
        loneliness=0.15, excitement=0.20, trust_in_teacher=0.25, self_esteem=0.55,
    ),
    # Gifted: moderate frustration (boredom), high self-esteem
    "gifted": dict(
        frustration=0.20, shame=0.05, anxiety=0.10, anger=0.08,
        loneliness=0.12, excitement=0.25, trust_in_teacher=0.65, self_esteem=0.80,
    ),
    # Sleep-deprived: flat affect, low excitement, moderate frustration
    "sleep_deprived": dict(
        frustration=0.25, shame=0.12, anxiety=0.18, anger=0.15,
        loneliness=0.15, excitement=0.08, trust_in_teacher=0.55, self_esteem=0.55,
    ),
}


# ---------------------------------------------------------------------------
# Behavior Pools (Korean research behavioral categories)
# ---------------------------------------------------------------------------

# KCI ART002794420 - 과잉행동/충동성/부주의 행동 관찰 척도
BEHAVIOR_POOLS: dict[str, tuple[str, ...]] = {
    "hyperactivity": (
        "seat_leaving", "leg_swinging", "fidgeting",
        "running", "excessive_talking",
    ),
    "impulsivity": (
        "blurting_answers", "interrupting", "off_topic_comments",
        "grabbing_objects",
    ),
    "inattention": (
        "careless_mistakes", "not_following_instructions", "incomplete_tasks",
        "easily_distracted", "daydreaming",
    ),
    "normal": (
        "on_task", "listening", "writing", "reading", "collaborating",
    ),
    "anxiety": (
        "withdrawal", "avoidance", "freezing",
        "seeking_reassurance", "crying",
    ),
    "odd": (
        "defiance", "arguing", "refusing_tasks",
        "provoking_peers", "blaming_others",
    ),
    "gifted": (
        "finishing_early", "helping_others", "asking_advanced_questions",
        "boredom_fidgeting",
    ),
}

# Which behavior pools each profile type draws from (primary, secondary)
_PROFILE_BEHAVIOR_MAP: dict[str, list[str]] = {
    "normal_quiet": ["normal"],
    "normal_active": ["normal", "hyperactivity"],
    "adhd_inattentive": ["inattention", "normal"],
    "adhd_hyperactive_impulsive": ["hyperactivity", "impulsivity"],
    "adhd_combined": ["inattention", "hyperactivity", "impulsivity"],
    "anxiety": ["anxiety", "normal"],
    "odd": ["odd", "impulsivity"],
    "gifted": ["gifted", "normal"],
    "sleep_deprived": ["inattention", "normal"],
}


# ---------------------------------------------------------------------------
# Memory Node
# ---------------------------------------------------------------------------


@dataclass
class MemoryNode:
    """A single memory entry in the student's memory stream.

    Follows the Generative Agents triple format: (subject, predicate, object).
    """

    node_id: int
    created_turn: int
    node_type: str  # "event" | "thought" | "interaction"
    subject: str
    predicate: str
    object: str
    description: str
    poignancy: float  # 1-10 importance score
    keywords: list[str] = field(default_factory=list)
    embedding: list[float] | None = None  # optional for cosine similarity


# ---------------------------------------------------------------------------
# Student Memory Stream
# ---------------------------------------------------------------------------


class StudentMemoryStream:
    """Per-student memory using the Generative Agents retrieval formula.

    Retrieval score = alpha * recency + beta * relevance + gamma * importance
    (Park et al., 2023: alpha=1.0, beta=1.0, gamma=1.0, normalized per component)
    We use 0.5/3.0/2.0 weights to emphasize relevance for classroom context.
    """

    def __init__(self, retention: int = 5, recency_decay: float = 0.99) -> None:
        self.events: list[MemoryNode] = []
        self.thoughts: list[MemoryNode] = []
        self.retention = retention
        self.recency_decay = recency_decay
        self._importance_accumulated: float = 0.0
        self._next_id: int = 0

    def _alloc_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def add_event(self, node: MemoryNode) -> None:
        """Add an event node and accumulate its poignancy."""
        self.events.append(node)
        self._importance_accumulated += node.poignancy
        # Trim oldest events beyond retention window (working memory limit)
        if len(self.events) > self.retention * 10:
            self.events = self.events[-(self.retention * 10):]

    def add_thought(self, node: MemoryNode) -> None:
        """Add a reflection/thought node."""
        self.thoughts.append(node)
        if len(self.thoughts) > self.retention * 5:
            self.thoughts = self.thoughts[-(self.retention * 5):]

    def retrieve(
        self,
        query_keywords: list[str],
        current_turn: int,
        top_k: int = 5,
    ) -> list[MemoryNode]:
        """Retrieve using recency x importance x relevance (Generative Agents formula).

        Score = 0.5 * recency + 3.0 * relevance + 2.0 * importance (normalized).
        """
        all_nodes = self.events + self.thoughts
        if not all_nodes:
            return []

        query_set = set(kw.lower() for kw in query_keywords)
        scored: list[tuple[float, MemoryNode]] = []

        # Compute max values for normalization
        max_poignancy = max((n.poignancy for n in all_nodes), default=1.0) or 1.0
        max_age = max((current_turn - n.created_turn for n in all_nodes), default=1) or 1

        for node in all_nodes:
            age = current_turn - node.created_turn
            # Recency: exponential decay
            recency = self.recency_decay ** age

            # Relevance: keyword overlap (Jaccard-like)
            node_kw = set(kw.lower() for kw in node.keywords)
            if query_set and node_kw:
                relevance = len(query_set & node_kw) / len(query_set | node_kw)
            else:
                relevance = 0.0

            # Importance: normalized poignancy
            importance = node.poignancy / max_poignancy

            score = 0.5 * recency + 3.0 * relevance + 2.0 * importance
            scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:top_k]]

    def should_reflect(self, importance_trigger: float) -> bool:
        """Check if accumulated importance exceeds threshold."""
        return self._importance_accumulated >= importance_trigger

    def reset_importance(self) -> None:
        """Reset accumulated importance after reflection."""
        self._importance_accumulated = 0.0

    def recent_events(self, n: int | None = None) -> list[MemoryNode]:
        """Return the N most recent events (default: retention count)."""
        k = n if n is not None else self.retention
        return self.events[-k:]


# ---------------------------------------------------------------------------
# Relationship Graph
# ---------------------------------------------------------------------------


@dataclass
class Relationship:
    """Directed edge in the social graph."""

    target_id: str
    type: str  # "friend" | "conflict" | "neutral" | "bully_target" | "bully_source"
    strength: float  # 0-1
    history: list[str] = field(default_factory=list)  # recent interaction summaries


class RelationshipGraph:
    """Social connection graph for all students in the classroom."""

    def __init__(self) -> None:
        self._edges: dict[tuple[str, str], Relationship] = {}

    def add(self, source: str, target: str, rel_type: str, strength: float) -> None:
        key = (source, target)
        if key in self._edges:
            self._edges[key].type = rel_type
            self._edges[key].strength = _clamp(strength)
        else:
            self._edges[key] = Relationship(
                target_id=target, type=rel_type, strength=_clamp(strength),
            )

    def get(self, source: str, target: str) -> Relationship | None:
        return self._edges.get((source, target))

    def get_friends(self, student_id: str) -> list[str]:
        return [
            rel.target_id
            for (src, _), rel in self._edges.items()
            if src == student_id and rel.type == "friend"
        ]

    def get_conflicts(self, student_id: str) -> list[str]:
        return [
            rel.target_id
            for (src, _), rel in self._edges.items()
            if src == student_id and rel.type in ("conflict", "bully_target")
        ]

    def get_neighbors(self, student_id: str, seat_map: dict[str, tuple[int, int]]) -> list[str]:
        """Get students seated within distance 2 of the given student."""
        if student_id not in seat_map:
            return []
        sx, sy = seat_map[student_id]
        neighbors = []
        for sid, (nx, ny) in seat_map.items():
            if sid == student_id:
                continue
            if abs(sx - nx) <= 2 and abs(sy - ny) <= 2:
                neighbors.append(sid)
        return neighbors

    def update_after_interaction(self, source: str, target: str, outcome: str) -> None:
        """Adjust relationship based on interaction outcome."""
        rel = self.get(source, target)
        if rel is None:
            # Create new neutral relationship
            self.add(source, target, "neutral", 0.5)
            rel = self._edges[(source, target)]

        rel.history.append(outcome)
        if len(rel.history) > 10:
            rel.history = rel.history[-10:]

        if outcome == "positive":
            rel.strength = _clamp(rel.strength + 0.05)
            if rel.strength > 0.6 and rel.type == "neutral":
                rel.type = "friend"
        elif outcome == "negative":
            rel.strength = _clamp(rel.strength - 0.08)
            if rel.strength < 0.3 and rel.type != "bully_target":
                rel.type = "conflict"
        elif outcome == "escalated":
            rel.strength = _clamp(rel.strength - 0.15)
            rel.type = "conflict"


# ---------------------------------------------------------------------------
# Classroom Context
# ---------------------------------------------------------------------------


@dataclass
class ClassroomContext:
    """Shared environment state visible to all students each turn."""

    turn: int
    period: int  # 1-5 (which class period of the day)
    day: int  # 1-190 (school year)
    subject: str  # "math", "korean", "science", "art", "pe", "recess"
    location: str  # "classroom", "playground", "office", "hallway"
    current_events: list[dict[str, Any]]  # events happening this turn
    class_mood: str  # "calm", "excited", "tense", "chaotic"
    teacher_action: str  # what teacher did this turn
    teacher_target: str | None  # who teacher targeted (None = whole class)


# ---------------------------------------------------------------------------
# Cognitive Student Agent
# ---------------------------------------------------------------------------


class CognitiveStudent:
    """Generative Agents cognitive architecture for a single student.

    Each turn executes: perceive -> retrieve -> (reflect) -> plan -> act.
    ADHD and other conditions emerge from parameter differences, not from
    special-case code paths.
    """

    def __init__(
        self,
        student_id: str,
        profile_type: str,
        age: int,
        gender: str,
        severity: str | None = None,
        base_params: CognitiveParameters | None = None,
        base_emotions: EmotionalState | None = None,
    ) -> None:
        self.student_id = student_id
        self.profile_type = profile_type  # key into COGNITIVE_PRESETS
        self.is_adhd: bool = profile_type.startswith("adhd_")  # ground truth, hidden
        self.age = age
        self.gender = gender
        self.severity = severity  # mild / moderate / severe (for ADHD subtypes)

        self.base_params = base_params or COGNITIVE_PRESETS.get(
            profile_type, COGNITIVE_PRESETS["normal_quiet"]
        )
        self.emotions = base_emotions or EmotionalState(
            **EMOTIONAL_PRESETS.get(profile_type, EMOTIONAL_PRESETS["normal_quiet"])
        )
        self.memory = StudentMemoryStream(
            retention=self.base_params.retention,
            recency_decay=self.base_params.recency_decay,
        )

        # Observable state (what teacher can see)
        self.state: dict[str, float] = {
            "distress_level": 0.2,
            "compliance": 0.7,
            "attention": 0.6,
            "escalation_risk": 0.1,
        }
        self.exhibited_behaviors: list[str] = []
        self.current_action: str = "on_task"
        self.managed: bool = False
        self.managed_turns: int = 0

        # Internal (hidden from teacher)
        self.current_plan: str = ""
        self.inner_thought: str = ""

    # ------------------------------------------------------------------
    # Emotion-modified parameters
    # ------------------------------------------------------------------

    def effective_params(self) -> CognitiveParameters:
        """Emotions temporarily modify cognitive parameters."""
        p = copy(self.base_params)

        # Frustration reduces attention bandwidth
        if self.emotions.frustration > 0.5:
            p.att_bandwidth = max(1, p.att_bandwidth - 1)

        # Anxiety increases vigilance but narrows focus
        if self.emotions.anxiety > 0.5:
            p.vision_r = max(2, p.vision_r - 1)
            p.importance_trigger = max(20.0, p.importance_trigger * 0.7)

        # Anger increases impulsivity
        if self.emotions.anger > 0.5:
            p.impulse_override = min(0.8, p.impulse_override + 0.2)

        # Trust in teacher improves plan following
        if self.emotions.trust_in_teacher > 0.7:
            p.plan_consistency = min(1.0, p.plan_consistency + 0.1)

        # Low self-esteem increases task initiation delay
        if self.emotions.self_esteem < 0.3:
            p.task_initiation_delay = min(1.0, p.task_initiation_delay + 0.15)

        # High excitement broadens attention (more distractible)
        if self.emotions.excitement > 0.6:
            p.vision_r = min(8, p.vision_r + 1)

        return p

    # ------------------------------------------------------------------
    # Main cognitive cycle
    # ------------------------------------------------------------------

    def step(self, context: ClassroomContext, rng: random.Random) -> list[str]:
        """Full cognitive cycle: perceive -> retrieve -> reflect -> plan -> act.

        Returns list of exhibited (observable) behaviors for this turn.
        """
        perceived = self._perceive(context, rng)
        retrieved = self._retrieve(perceived, context.turn)

        if self.memory.should_reflect(self.effective_params().importance_trigger):
            self._reflect(retrieved, context.turn)

        plan = self._plan(context, retrieved, rng)
        behaviors = self._act(plan, context, rng)
        self._update_emotions(context, behaviors)
        self._update_state(behaviors)

        return behaviors

    # ------------------------------------------------------------------
    # Perceive
    # ------------------------------------------------------------------

    def _perceive(
        self, context: ClassroomContext, rng: random.Random,
    ) -> list[MemoryNode]:
        """Perceive events within attention bandwidth.

        Filters current_events down to att_bandwidth items, rates poignancy,
        and stores in memory stream.
        """
        params = self.effective_params()
        all_events = context.current_events

        # Sort by salience: teacher-directed events first, then proximity/relevance
        def _salience(evt: dict[str, Any]) -> float:
            s = 0.0
            # Events targeting this student are maximally salient
            if evt.get("target") == self.student_id:
                s += 10.0
            # Teacher actions are salient
            if evt.get("actor") == "teacher":
                s += 5.0
            # Peer events with friends/conflicts are salient
            if evt.get("actor") in (
                self.memory.recent_events(3)
                and [n.subject for n in self.memory.recent_events(3)]
                or []
            ):
                s += 2.0
            # Random jitter
            s += rng.random() * 0.5
            return s

        sorted_events = sorted(all_events, key=_salience, reverse=True)
        attended = sorted_events[: params.att_bandwidth]

        perceived_nodes: list[MemoryNode] = []
        for evt in attended:
            poignancy = self._rate_poignancy(evt, rng)
            node = MemoryNode(
                node_id=self.memory._alloc_id(),
                created_turn=context.turn,
                node_type="event",
                subject=str(evt.get("actor", "unknown")),
                predicate=str(evt.get("action", "did")),
                object=str(evt.get("target", "something")),
                description=str(evt.get("description", "")),
                poignancy=poignancy,
                keywords=_extract_keywords(evt),
            )
            self.memory.add_event(node)
            perceived_nodes.append(node)

        return perceived_nodes

    def _rate_poignancy(self, event: dict[str, Any], rng: random.Random) -> float:
        """Rate event importance on 1-10 scale.

        Events targeting this student, conflict events, and teacher actions
        receive higher poignancy.
        """
        base = 3.0

        # Events directly involving this student
        if event.get("target") == self.student_id:
            base += 3.0
        if event.get("actor") == self.student_id:
            base += 1.0

        # Teacher actions are important
        if event.get("actor") == "teacher":
            base += 2.0

        # Conflict/emotional events
        etype = str(event.get("type", ""))
        if "conflict" in etype or "bully" in etype or "anger" in etype:
            base += 2.0
        if "praise" in etype:
            base += 1.5

        # Social sensitivity amplifies peer events
        if event.get("actor", "").startswith("S"):
            base += self.emotions.excitement * 1.0

        # Noise
        base += rng.uniform(-0.5, 0.5)
        return _clamp(base, 1.0, 10.0)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def _retrieve(
        self, perceived: list[MemoryNode], current_turn: int,
    ) -> list[MemoryNode]:
        """Retrieve related memories using recency x importance x relevance."""
        if not perceived:
            return self.memory.recent_events()

        query_keywords: list[str] = []
        for node in perceived:
            query_keywords.extend(node.keywords)

        return self.memory.retrieve(query_keywords, current_turn, top_k=5)

    # ------------------------------------------------------------------
    # Reflect
    # ------------------------------------------------------------------

    def _reflect(self, retrieved: list[MemoryNode], current_turn: int) -> None:
        """Generate higher-level thoughts from retrieved memories.

        E.g., "The teacher always praises Mina but not me" -> loneliness++
        Reflections are stored as thought nodes for future retrieval.
        """
        if not retrieved:
            self.memory.reset_importance()
            return

        # Count patterns in retrieved memories
        teacher_praise_others = 0
        teacher_scold_me = 0
        peer_conflict_count = 0
        peer_positive_count = 0

        for node in retrieved:
            desc = node.description.lower()
            if "praise" in desc and node.object != self.student_id:
                teacher_praise_others += 1
            if ("scold" in desc or "correct" in desc) and node.object == self.student_id:
                teacher_scold_me += 1
            if "conflict" in desc or "argue" in desc:
                peer_conflict_count += 1
            if "help" in desc or "friend" in desc:
                peer_positive_count += 1

        # Generate thought based on strongest pattern
        thought_desc = ""
        thought_keywords: list[str] = []
        poignancy = 5.0

        if teacher_scold_me >= 2:
            thought_desc = "The teacher keeps correcting me. Maybe I am doing something wrong."
            thought_keywords = ["teacher", "correction", "self", "shame"]
            self.emotions.shame = _clamp(self.emotions.shame + 0.05)
            self.emotions.trust_in_teacher = _clamp(self.emotions.trust_in_teacher - 0.03)
            poignancy = 7.0
        elif teacher_praise_others >= 2:
            thought_desc = "The teacher praises others but not me."
            thought_keywords = ["teacher", "praise", "others", "loneliness"]
            self.emotions.loneliness = _clamp(self.emotions.loneliness + 0.04)
            poignancy = 6.0
        elif peer_conflict_count >= 2:
            thought_desc = "There are too many conflicts around me."
            thought_keywords = ["peers", "conflict", "stress"]
            self.emotions.anxiety = _clamp(self.emotions.anxiety + 0.04)
            poignancy = 6.5
        elif peer_positive_count >= 2:
            thought_desc = "My classmates are nice to me."
            thought_keywords = ["peers", "friendship", "positive"]
            self.emotions.loneliness = _clamp(self.emotions.loneliness - 0.03)
            self.emotions.self_esteem = _clamp(self.emotions.self_esteem + 0.02)
            poignancy = 5.0

        if thought_desc:
            thought_node = MemoryNode(
                node_id=self.memory._alloc_id(),
                created_turn=current_turn,
                node_type="thought",
                subject=self.student_id,
                predicate="reflected",
                object=thought_desc[:40],
                description=thought_desc,
                poignancy=poignancy,
                keywords=thought_keywords,
            )
            self.memory.add_thought(thought_node)
            self.inner_thought = thought_desc

        self.memory.reset_importance()

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------

    def _plan(
        self,
        context: ClassroomContext,
        retrieved: list[MemoryNode],
        rng: random.Random,
    ) -> str:
        """Decide next action based on context + memory + personality.

        Returns a plan string that _act() will interpret.
        """
        params = self.effective_params()

        # Task initiation delay: probability of not starting the task yet
        if context.turn <= 3 and rng.random() < params.task_initiation_delay:
            self.current_plan = "delay_start"
            return "delay_start"

        # Impulse override: bypass planning entirely
        if rng.random() < params.impulse_override:
            plan = self._impulsive_action(context, rng)
            self.current_plan = plan
            return plan

        # Plan consistency: follow the schedule/instruction
        if rng.random() < params.plan_consistency:
            plan = self._follow_schedule(context)
            self.current_plan = plan
            return plan

        # Otherwise: reactive action based on context and retrieved memories
        plan = self._reactive_action(context, retrieved, rng)
        self.current_plan = plan
        return plan

    def _impulsive_action(self, context: ClassroomContext, rng: random.Random) -> str:
        """Generate an impulsive action based on profile type."""
        pools = _PROFILE_BEHAVIOR_MAP.get(self.profile_type, ["normal"])
        # Prefer disruptive pools for impulse
        disruptive = [p for p in pools if p not in ("normal", "gifted")]
        if disruptive:
            pool_name = rng.choice(disruptive)
        else:
            pool_name = rng.choice(pools)
        behaviors = BEHAVIOR_POOLS.get(pool_name, BEHAVIOR_POOLS["normal"])
        return rng.choice(behaviors)

    def _follow_schedule(self, context: ClassroomContext) -> str:
        """Follow the current lesson plan / teacher instruction."""
        if context.subject == "recess":
            return "free_play"
        if context.subject == "pe":
            return "physical_activity"
        return "on_task"

    def _reactive_action(
        self,
        context: ClassroomContext,
        retrieved: list[MemoryNode],
        rng: random.Random,
    ) -> str:
        """React to context based on emotional state and memories."""
        # High frustration -> disruptive
        if self.emotions.frustration > 0.6:
            return rng.choice(["fidgeting", "seat_leaving", "off_topic_comments"])

        # High anxiety -> withdrawal
        if self.emotions.anxiety > 0.6:
            return rng.choice(["withdrawal", "avoidance", "freezing"])

        # High anger -> conflict
        if self.emotions.anger > 0.6:
            return rng.choice(["arguing", "defiance", "provoking_peers"])

        # High excitement -> overactivity
        if self.emotions.excitement > 0.7:
            return rng.choice(["excessive_talking", "blurting_answers", "running"])

        # Boredom (gifted or low excitement + on_task subject)
        if self.emotions.excitement < 0.1 and self.profile_type == "gifted":
            return rng.choice(["boredom_fidgeting", "helping_others", "asking_advanced_questions"])

        # Default: mild off-task
        return rng.choice(["daydreaming", "easily_distracted", "looking_around"])

    # ------------------------------------------------------------------
    # Act
    # ------------------------------------------------------------------

    def _act(
        self, plan: str, context: ClassroomContext, rng: random.Random,
    ) -> list[str]:
        """Execute plan and return observable behaviors."""
        params = self.effective_params()

        if plan in ("on_task", "free_play", "physical_activity"):
            self.current_action = plan
            pools = _PROFILE_BEHAVIOR_MAP.get(self.profile_type, ["normal"])
            # ADHD/disordered students leak their primary behaviors even when
            # "on_task" — the rate depends on plan_consistency (lower = more leak)
            # Normal students almost always produce normal behaviors.
            leak_rate = 1.0 - params.plan_consistency  # e.g. ADHD-I: 1-0.50 = 0.50
            if plan == "on_task" and rng.random() < leak_rate and pools[0] != "normal":
                # Primary behavior pool leaks through (e.g. inattention for ADHD-I)
                primary_pool = BEHAVIOR_POOLS.get(pools[0], BEHAVIOR_POOLS["normal"])
                return [rng.choice(primary_pool)]
            return [rng.choice(BEHAVIOR_POOLS["normal"])]

        if plan == "delay_start":
            self.current_action = "not_started"
            return [rng.choice(["daydreaming", "fidgeting", "looking_around"])]

        # Plan is a specific behavior from impulsive/reactive action
        self.current_action = plan
        behaviors = [plan]

        # High arousal states produce multiple behaviors
        arousal = (self.emotions.excitement + self.emotions.anger + self.emotions.frustration) / 3
        if arousal > 0.5 and rng.random() < arousal:
            pools = _PROFILE_BEHAVIOR_MAP.get(self.profile_type, ["normal"])
            extra_pool_name = rng.choice(pools)
            extra_pool = BEHAVIOR_POOLS.get(extra_pool_name, BEHAVIOR_POOLS["normal"])
            extra = rng.choice(extra_pool)
            if extra != plan:
                behaviors.append(extra)

        return behaviors[:3]  # cap at 3 observable behaviors

    # ------------------------------------------------------------------
    # Emotion Update
    # ------------------------------------------------------------------

    def _update_emotions(self, context: ClassroomContext, behaviors: list[str]) -> None:
        """Update emotional state based on what happened this turn."""
        # Natural decay toward baseline (emotional homeostasis)
        decay = 0.02
        baseline = EMOTIONAL_PRESETS.get(
            self.profile_type, EMOTIONAL_PRESETS["normal_quiet"]
        )
        for attr in (
            "frustration", "shame", "anxiety", "anger",
            "loneliness", "excitement", "trust_in_teacher", "self_esteem",
        ):
            current = getattr(self.emotions, attr)
            target = baseline[attr]
            delta = (target - current) * decay
            setattr(self.emotions, attr, _clamp(current + delta))

        # Teacher targeting this student
        if context.teacher_target == self.student_id:
            action = context.teacher_action.lower()
            if "praise" in action:
                self.emotions.self_esteem = _clamp(self.emotions.self_esteem + 0.05)
                self.emotions.trust_in_teacher = _clamp(self.emotions.trust_in_teacher + 0.03)
                self.emotions.excitement = _clamp(self.emotions.excitement + 0.02)
            elif "correct" in action or "scold" in action:
                self.emotions.shame = _clamp(self.emotions.shame + 0.06)
                self.emotions.frustration = _clamp(self.emotions.frustration + 0.04)
                self.emotions.trust_in_teacher = _clamp(self.emotions.trust_in_teacher - 0.03)
            elif "ignore" in action:
                self.emotions.loneliness = _clamp(self.emotions.loneliness + 0.03)

        # Being off-task increases frustration slightly (task failure feedback)
        off_task_behaviors = set(
            BEHAVIOR_POOLS["inattention"] + BEHAVIOR_POOLS["hyperactivity"]
        )
        if any(b in off_task_behaviors for b in behaviors):
            self.emotions.frustration = _clamp(self.emotions.frustration + 0.01)

        # Chaotic class mood increases anxiety for sensitive students
        if context.class_mood == "chaotic":
            sensitivity = self.base_params.social_sensitivity
            self.emotions.anxiety = _clamp(self.emotions.anxiety + 0.02 * sensitivity)

    # ------------------------------------------------------------------
    # State Update (observable by teacher)
    # ------------------------------------------------------------------

    def _update_state(self, behaviors: list[str]) -> None:
        """Update observable state from internal state + behaviors."""
        self.exhibited_behaviors = behaviors

        on_task_set = set(BEHAVIOR_POOLS["normal"])
        disruptive_set = set(
            BEHAVIOR_POOLS["hyperactivity"]
            + BEHAVIOR_POOLS["impulsivity"]
            + BEHAVIOR_POOLS["odd"]
        )
        inattentive_set = set(BEHAVIOR_POOLS["inattention"])

        n_on_task = sum(1 for b in behaviors if b in on_task_set)
        n_disruptive = sum(1 for b in behaviors if b in disruptive_set)
        n_inattentive = sum(1 for b in behaviors if b in inattentive_set)

        # Attention: high if on-task, low if inattentive/disruptive
        if n_on_task > 0 and n_disruptive == 0 and n_inattentive == 0:
            self.state["attention"] = _clamp(self.state["attention"] + 0.05)
        elif n_inattentive > 0:
            self.state["attention"] = _clamp(self.state["attention"] - 0.08)
        elif n_disruptive > 0:
            self.state["attention"] = _clamp(self.state["attention"] - 0.06)

        # Compliance: on-task raises it, disruptive lowers it
        if n_on_task > 0 and n_disruptive == 0:
            self.state["compliance"] = _clamp(self.state["compliance"] + 0.03)
        elif n_disruptive > 0:
            self.state["compliance"] = _clamp(self.state["compliance"] - 0.07)

        # Distress: mirrors frustration + shame + anxiety (observable portion)
        emotional_distress = (
            self.emotions.frustration * 0.4
            + self.emotions.shame * 0.3
            + self.emotions.anxiety * 0.3
        )
        # Smooth transition: 70% previous, 30% current emotional signal
        self.state["distress_level"] = _clamp(
            0.7 * self.state["distress_level"] + 0.3 * emotional_distress
        )

        # Escalation risk: anger + conflict behaviors
        anger_signal = self.emotions.anger * 0.5
        conflict_signal = 0.3 if n_disruptive > 1 else 0.0
        self.state["escalation_risk"] = _clamp(
            0.7 * self.state["escalation_risk"] + 0.3 * (anger_signal + conflict_signal)
        )

        # Managed tracking
        if self.managed:
            self.managed_turns += 1

    # ------------------------------------------------------------------
    # External interaction handlers
    # ------------------------------------------------------------------

    def react_to_teacher(
        self, action_type: str, strategy: str | None = None,
    ) -> None:
        """React to teacher intervention. Updates emotions and state.

        Called by the environment when the teacher targets this student.
        """
        if action_type == "praise" or strategy == "labeled_praise":
            self.emotions.self_esteem = _clamp(self.emotions.self_esteem + 0.06)
            self.emotions.trust_in_teacher = _clamp(self.emotions.trust_in_teacher + 0.05)
            self.emotions.frustration = _clamp(self.emotions.frustration - 0.04)
            self.state["compliance"] = _clamp(self.state["compliance"] + 0.08)

        elif action_type == "private_correction":
            # Private correction: less shame, more effective (O'Leary 1970)
            self.emotions.shame = _clamp(self.emotions.shame + 0.03)
            self.emotions.trust_in_teacher = _clamp(self.emotions.trust_in_teacher + 0.02)
            self.state["compliance"] = _clamp(self.state["compliance"] + 0.10)
            self.state["escalation_risk"] = _clamp(self.state["escalation_risk"] - 0.05)

        elif action_type == "public_correction":
            # Public correction: more shame, less effective, risk of escalation
            self.emotions.shame = _clamp(self.emotions.shame + 0.10)
            self.emotions.anger = _clamp(self.emotions.anger + 0.05)
            self.emotions.trust_in_teacher = _clamp(self.emotions.trust_in_teacher - 0.05)
            self.state["compliance"] = _clamp(self.state["compliance"] + 0.03)
            self.state["escalation_risk"] = _clamp(self.state["escalation_risk"] + 0.04)

        elif action_type == "individual_intervention":
            # Strategy-dependent effects
            if strategy == "break_offer":
                self.emotions.frustration = _clamp(self.emotions.frustration - 0.08)
                self.emotions.trust_in_teacher = _clamp(self.emotions.trust_in_teacher + 0.03)
            elif strategy == "empathic_acknowledgment":
                self.emotions.shame = _clamp(self.emotions.shame - 0.05)
                self.emotions.trust_in_teacher = _clamp(self.emotions.trust_in_teacher + 0.06)
                self.emotions.loneliness = _clamp(self.emotions.loneliness - 0.04)
            elif strategy == "redirect_attention":
                self.state["attention"] = _clamp(self.state["attention"] + 0.10)
            elif strategy == "firm_boundary":
                self.state["compliance"] = _clamp(self.state["compliance"] + 0.06)
                self.emotions.anger = _clamp(self.emotions.anger + 0.03)
            elif strategy == "sensory_support":
                self.emotions.frustration = _clamp(self.emotions.frustration - 0.05)
                self.state["attention"] = _clamp(self.state["attention"] + 0.05)
            elif strategy in ("transition_warning", "visual_schedule_cue", "countdown_timer"):
                self.emotions.anxiety = _clamp(self.emotions.anxiety - 0.04)
                self.state["compliance"] = _clamp(self.state["compliance"] + 0.05)
            elif strategy == "offer_choice":
                self.emotions.self_esteem = _clamp(self.emotions.self_esteem + 0.03)
                self.state["compliance"] = _clamp(self.state["compliance"] + 0.07)
            elif strategy == "collaborative_problem_solving":
                self.emotions.trust_in_teacher = _clamp(self.emotions.trust_in_teacher + 0.05)
                self.emotions.frustration = _clamp(self.emotions.frustration - 0.06)
                self.state["compliance"] = _clamp(self.state["compliance"] + 0.08)

        elif action_type == "ignore" or strategy == "ignore_wait":
            # Ignoring: may increase frustration/loneliness for attention-seeking students
            if self.base_params.social_sensitivity > 0.5:
                self.emotions.loneliness = _clamp(self.emotions.loneliness + 0.04)
                self.emotions.frustration = _clamp(self.emotions.frustration + 0.02)

    def react_to_peer(
        self, peer_id: str, event_type: str, rng: random.Random,
    ) -> None:
        """React to peer interaction. Updates emotions and may change behavior.

        Called by the environment when a peer event affects this student.
        """
        sensitivity = self.base_params.social_sensitivity

        if event_type == "peer_help":
            self.emotions.loneliness = _clamp(self.emotions.loneliness - 0.05 * sensitivity)
            self.emotions.trust_in_teacher = _clamp(self.emotions.trust_in_teacher + 0.01)
            self.emotions.self_esteem = _clamp(self.emotions.self_esteem + 0.03 * sensitivity)

        elif event_type == "peer_conflict":
            self.emotions.anger = _clamp(self.emotions.anger + 0.08 * sensitivity)
            self.emotions.anxiety = _clamp(self.emotions.anxiety + 0.05 * sensitivity)
            self.state["escalation_risk"] = _clamp(self.state["escalation_risk"] + 0.06)
            # Conflict tendency determines if student escalates
            if rng.random() < self.base_params.conflict_tendency:
                self.current_action = "arguing"
                self.exhibited_behaviors.append("arguing")

        elif event_type == "peer_bullying":
            self.emotions.shame = _clamp(self.emotions.shame + 0.10 * sensitivity)
            self.emotions.anger = _clamp(self.emotions.anger + 0.06 * sensitivity)
            self.emotions.loneliness = _clamp(self.emotions.loneliness + 0.08 * sensitivity)
            self.emotions.self_esteem = _clamp(self.emotions.self_esteem - 0.06 * sensitivity)

        elif event_type == "peer_exclusion":
            self.emotions.loneliness = _clamp(self.emotions.loneliness + 0.10 * sensitivity)
            self.emotions.self_esteem = _clamp(self.emotions.self_esteem - 0.05 * sensitivity)
            self.emotions.anxiety = _clamp(self.emotions.anxiety + 0.04 * sensitivity)

        elif event_type == "peer_contagion":
            # Behavioral contagion: social sensitivity determines probability
            if rng.random() < sensitivity * 0.5:
                pools = _PROFILE_BEHAVIOR_MAP.get(self.profile_type, ["normal"])
                pool_name = rng.choice(pools)
                pool = BEHAVIOR_POOLS.get(pool_name, BEHAVIOR_POOLS["normal"])
                self.exhibited_behaviors.append(rng.choice(pool))

        elif event_type == "peer_friendship":
            self.emotions.loneliness = _clamp(self.emotions.loneliness - 0.06 * sensitivity)
            self.emotions.excitement = _clamp(self.emotions.excitement + 0.04 * sensitivity)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float to [lo, hi]."""
    return max(lo, min(hi, float(value)))


def _extract_keywords(event: dict[str, Any]) -> list[str]:
    """Extract searchable keywords from an event dict."""
    keywords: list[str] = []
    for key in ("type", "action", "actor", "target", "subject"):
        val = event.get(key)
        if val and isinstance(val, str):
            keywords.append(val.lower())
    desc = event.get("description", "")
    if isinstance(desc, str):
        for word in desc.lower().split():
            if len(word) > 3:
                keywords.append(word)
    return keywords[:10]  # cap to avoid bloat
