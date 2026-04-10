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


# ---------------------------------------------------------------------------
# Dimensional Parameter Model (RDoC framework)
# ---------------------------------------------------------------------------
#
# Instead of defining every profile as an absolute preset, we define ONE
# normative baseline and express all profiles (ADHD, ODD, anxiety, etc.) as
# deltas from that baseline. This matches:
#
#   - Research Domain Criteria (Insel et al., 2010; Cuthbert & Insel, 2013)
#   - Dimensional psychopathology (Haslam et al., 2006; Marcus & Barry, 2011)
#   - Computational psychiatry parameter-shift models (Huys et al., 2016)
#
# Each delta entry is a float offset added to the corresponding baseline
# field. Observable-state deltas fix the bug where every profile started
# with identical first-turn observable values.
#
# Initial values are hand-seeded from the literature and serve as the
# starting point for autoresearch calibration (see docs/정리.md §25).
# ---------------------------------------------------------------------------

# Normative baseline — "population mean" normal child cognitive parameters
BASE_COGNITIVE = CognitiveParameters(
    att_bandwidth=3,
    vision_r=4,
    retention=5,
    recency_decay=0.985,
    importance_trigger=135.0,
    plan_consistency=0.85,
    impulse_override=0.075,
    task_initiation_delay=0.0,
    social_sensitivity=0.45,
    conflict_tendency=0.075,
)

# Normative baseline — emotional state
BASE_EMOTIONAL: dict[str, float] = dict(
    frustration=0.11,
    shame=0.08,
    anxiety=0.11,
    anger=0.075,
    loneliness=0.07,
    excitement=0.225,
    trust_in_teacher=0.675,
    self_esteem=0.71,
)

# Normative baseline — observable state (what teacher can see)
# NOTE: previously hardcoded identically across all profiles — now per-profile
BASE_OBSERVABLE: dict[str, float] = dict(
    distress_level=0.15,
    compliance=0.72,
    attention=0.65,
    escalation_risk=0.08,
)


# ---------------------------------------------------------------------------
# Profile Deltas — each profile = BASE + delta on named fields only
# ---------------------------------------------------------------------------
#
# Delta structure:
#   {
#     "cognitive":  {field: offset, ...}   # added to BASE_COGNITIVE
#     "emotional":  {field: offset, ...}   # added to BASE_EMOTIONAL
#     "observable": {field: offset, ...}   # added to BASE_OBSERVABLE
#     "_sources":   "citation string"      # provenance for each delta
#   }
#
# Ranges are documented alongside each delta for autoresearch calibration
# (see .harness/objective_v2.yaml — to be created).
# ---------------------------------------------------------------------------

PROFILE_DELTAS: dict[str, dict] = {
    # ── Normal variants (small delta from baseline) ────────────────────────
    "normal_quiet": {
        "cognitive": {
            "plan_consistency": +0.05, "impulse_override": -0.025,
            "social_sensitivity": -0.15, "recency_decay": +0.005,
            "importance_trigger": +15.0, "conflict_tendency": -0.025,
        },
        "emotional": {
            "frustration": -0.01, "excitement": -0.075, "self_esteem": +0.01,
            "trust_in_teacher": +0.025, "anger": -0.025,
        },
        "observable": {},
        "_sources": "Within 1 SD of normative baseline",
    },
    "normal_active": {
        "cognitive": {
            "vision_r": +1, "recency_decay": -0.005,
            "importance_trigger": -15.0, "plan_consistency": -0.05,
            "impulse_override": +0.025, "social_sensitivity": +0.15,
            "conflict_tendency": +0.025,
        },
        "emotional": {
            "frustration": +0.01, "loneliness": -0.01, "excitement": +0.075,
            "trust_in_teacher": -0.025, "anger": +0.025,
        },
        "observable": {
            "compliance": -0.02, "escalation_risk": +0.02,
        },
        "_sources": "Within 1 SD of normative baseline; social/active variant",
    },

    # ── ADHD subtypes (DSM-5; Korean prevalence KCI ART001701933) ──────────
    "adhd_inattentive": {
        "cognitive": {
            "att_bandwidth": -2,              # DSM-5 A1: sustained attention
            "vision_r": -1,                    # Narrow functional scan
            "retention": -3,                   # Working memory (Barkley 1997)
            "recency_decay": -0.035,           # Faster memory decay
            "importance_trigger": -85.0,       # Lower reflection threshold
            "plan_consistency": -0.35,         # Executive dysfunction
            "impulse_override": +0.075,        # Mild impulsivity
            "task_initiation_delay": +0.40,    # Task initiation deficit
            "social_sensitivity": -0.05,
        },
        "emotional": {
            "frustration": +0.19,              # Academic failure cumulation
            "shame": +0.17,                    # K-CBCL internalizing
            "anxiety": +0.14,
            "loneliness": +0.13,
            "excitement": -0.125,              # Blunted positive affect
            "trust_in_teacher": -0.175,
            "self_esteem": -0.26,              # Korean studies consistent
        },
        "observable": {
            "attention": -0.25,                # Visible inattention
            "compliance": -0.12,
            "distress_level": +0.05,
        },
        "_sources": "DSM-5 314.00; Barkley 1997; KCI ART002794420, ART001701933",
    },
    "adhd_hyperactive_impulsive": {
        "cognitive": {
            "att_bandwidth": -1,
            "vision_r": +2,                    # Hyper-distractible scan
            "retention": -2,
            "recency_decay": -0.025,
            "importance_trigger": -95.0,
            "plan_consistency": -0.45,
            "impulse_override": +0.325,        # Core hyperactivity-impulsivity
            "task_initiation_delay": +0.10,
            "social_sensitivity": +0.25,
            "conflict_tendency": +0.175,
        },
        "emotional": {
            "frustration": +0.14,
            "shame": +0.07,
            "anxiety": +0.04,
            "anger": +0.175,                   # Emotional dysregulation
            "loneliness": +0.05,
            "excitement": +0.175,
            "trust_in_teacher": -0.225,
            "self_esteem": -0.21,
        },
        "observable": {
            "attention": -0.15,
            "compliance": -0.17,
            "escalation_risk": +0.17,          # Visible disruption
            "distress_level": +0.03,
        },
        "_sources": "DSM-5 314.01; Sonuga-Barke 2003 dual pathway; KCI ART002794420",
    },
    "adhd_combined": {
        # Combined = additive(inattentive, hyperactive) + interaction term
        # Interaction term captures non-linear comorbidity effects
        "_base": ["adhd_inattentive", "adhd_hyperactive_impulsive"],
        "cognitive": {
            # Additional shifts on top of additive base
            "plan_consistency": +0.05,         # Slight offset vs pure sum
            "att_bandwidth": +1,               # Cap at -2 total
        },
        "emotional": {
            "frustration": -0.04,              # Slight dampening
            "anger": -0.05,
            "self_esteem": +0.01,
        },
        "observable": {
            "distress_level": +0.02,
        },
        "_sources": "DSM-5 314.01 combined; MTA Cooperative Group 1999",
    },

    # ── Anxiety disorders ──────────────────────────────────────────────────
    "anxiety": {
        "cognitive": {
            "att_bandwidth": -1,               # Narrow hypervigilance
            "vision_r": -1,
            "retention": -1,
            "recency_decay": -0.005,
            "importance_trigger": -85.0,       # Hyper-reactive reflection
            "plan_consistency": -0.10,
            "impulse_override": +0.025,
            "task_initiation_delay": +0.20,    # Avoidance delay
            "social_sensitivity": +0.35,
            "conflict_tendency": -0.045,
        },
        "emotional": {
            "frustration": +0.04,
            "shame": +0.22,
            "anxiety": +0.34,                  # Core symptom
            "anger": -0.025,
            "loneliness": +0.18,
            "excitement": -0.145,
            "trust_in_teacher": -0.125,
            "self_esteem": -0.36,
        },
        "observable": {
            "distress_level": +0.15,
            "attention": -0.08,
            "compliance": +0.03,               # Compliant but anxious
        },
        "_sources": "DSM-5 300.02 GAD; K-CBCL internalizing cluster (ART002650863)",
    },

    # ── Oppositional Defiant Disorder ──────────────────────────────────────
    "odd": {
        "cognitive": {
            "att_bandwidth": -1,
            "recency_decay": -0.015,
            "importance_trigger": -55.0,
            "plan_consistency": -0.45,         # Refuses to follow
            "impulse_override": +0.275,
            "task_initiation_delay": +0.10,
            "social_sensitivity": +0.05,
            "conflict_tendency": +0.325,       # Core: peer/authority conflict
        },
        "emotional": {
            "frustration": +0.29,
            "shame": 0.00,                     # Externalizing — low shame
            "anxiety": +0.01,
            "anger": +0.375,                   # Core
            "loneliness": +0.08,
            "trust_in_teacher": -0.425,        # Authority mistrust
            "self_esteem": -0.16,
        },
        "observable": {
            "compliance": -0.32,
            "escalation_risk": +0.27,
            "distress_level": +0.04,
        },
        "_sources": "DSM-5 313.81; Burke et al. 2002 on ODD-ADHD comorbidity",
    },

    # ── Gifted ─────────────────────────────────────────────────────────────
    "gifted": {
        "cognitive": {
            "att_bandwidth": +1,
            "vision_r": +1,
            "retention": +1,
            "recency_decay": +0.005,
            "importance_trigger": +65.0,
            "plan_consistency": 0.00,
            "impulse_override": +0.075,        # Boredom-driven
            "social_sensitivity": -0.05,
        },
        "emotional": {
            "frustration": +0.09,              # Boredom frustration
            "shame": -0.03,
            "anxiety": -0.01,
            "excitement": +0.025,
            "self_esteem": +0.09,
        },
        "observable": {
            "attention": +0.05,
            "compliance": +0.02,
        },
        "_sources": "Korean gifted education research; boredom-driven fidgeting pattern",
    },

    # ── Sleep-deprived (state, not trait) ──────────────────────────────────
    "sleep_deprived": {
        "cognitive": {
            "att_bandwidth": -1,
            "vision_r": -1,
            "retention": -2,
            "recency_decay": -0.025,
            "importance_trigger": -35.0,
            "plan_consistency": -0.25,
            "impulse_override": +0.075,
            "task_initiation_delay": +0.30,
            "social_sensitivity": -0.15,
            "conflict_tendency": +0.045,
        },
        "emotional": {
            "frustration": +0.14,
            "shame": +0.04,
            "anxiety": +0.07,
            "anger": +0.075,
            "loneliness": +0.08,
            "excitement": -0.145,              # Flat affect
            "trust_in_teacher": -0.125,
            "self_esteem": -0.16,
        },
        "observable": {
            "attention": -0.18,
            "compliance": -0.05,
            "distress_level": +0.04,
        },
        "_sources": "Sleep deprivation → inattentive-like presentation (literature review)",
    },

    # ── Comorbidity profiles (dimensional overlap) ─────────────────────────
    # Epidemiology: ADHD+ODD ≈ 35-45%, ADHD+anxiety ≈ 25-30%
    # (MTA Cooperative Group 1999; Jensen et al. 2001; Korean KCMHS)
    "adhd_i_plus_anxiety": {
        "_base": ["adhd_inattentive", "anxiety"],
        "cognitive": {
            "plan_consistency": +0.05,         # Mild offset vs pure sum
            "social_sensitivity": -0.10,
        },
        "emotional": {
            "shame": +0.05,                    # Synergistic internalizing
            "frustration": -0.03,
            "anxiety": -0.02,
        },
        "observable": {
            "distress_level": +0.03,
        },
        "_sources": "MTA 1999 ADHD-anxiety comorbid pattern; Jensen et al. 2001",
    },
    "adhd_h_plus_odd": {
        "_base": ["adhd_hyperactive_impulsive", "odd"],
        "cognitive": {
            "impulse_override": -0.05,         # Cap to avoid runaway
            "conflict_tendency": +0.05,        # Slight synergy
        },
        "emotional": {
            "anger": +0.04,                    # Amplified externalizing
            "frustration": -0.03,
        },
        "observable": {
            "escalation_risk": +0.03,
        },
        "_sources": "Burke et al. 2002; MTA 1999; common clinical presentation",
    },
}


def _combine_deltas(name: str, visited: set[str] | None = None) -> dict:
    """Recursively combine profile deltas, expanding `_base` composition.

    Returns a dict with keys {cognitive, emotional, observable} — each
    containing summed offsets across all base profiles and the local delta.
    """
    if visited is None:
        visited = set()
    if name in visited:
        raise ValueError(f"Circular profile delta reference: {name}")
    visited.add(name)

    spec = PROFILE_DELTAS.get(name, {})
    combined: dict[str, dict[str, float]] = {
        "cognitive": {}, "emotional": {}, "observable": {},
    }

    # Expand base profiles first (additive composition)
    for base_name in spec.get("_base", []):
        base_combined = _combine_deltas(base_name, visited.copy())
        for section in ("cognitive", "emotional", "observable"):
            for k, v in base_combined[section].items():
                combined[section][k] = combined[section].get(k, 0.0) + v

    # Apply local delta on top
    for section in ("cognitive", "emotional", "observable"):
        for k, v in spec.get(section, {}).items():
            combined[section][k] = combined[section].get(k, 0.0) + v

    return combined


def _apply_delta_to_cognitive(delta: dict[str, float]) -> CognitiveParameters:
    """Return a new CognitiveParameters = BASE_COGNITIVE + delta."""
    params = copy(BASE_COGNITIVE)
    for field_name, offset in delta.items():
        if hasattr(params, field_name):
            current = getattr(params, field_name)
            new_val = current + offset
            # Clamp integer fields
            if field_name in {"att_bandwidth", "vision_r", "retention"}:
                new_val = max(1, int(round(new_val)))
            # Clamp probability/float fields to [0, 1] (or reasonable range)
            elif field_name in {
                "plan_consistency", "impulse_override",
                "task_initiation_delay", "social_sensitivity", "conflict_tendency",
            }:
                new_val = max(0.0, min(1.0, float(new_val)))
            elif field_name == "recency_decay":
                new_val = max(0.80, min(1.0, float(new_val)))
            elif field_name == "importance_trigger":
                new_val = max(10.0, float(new_val))
            setattr(params, field_name, new_val)
    return params


def _apply_delta_to_dict(base: dict[str, float], delta: dict[str, float]) -> dict[str, float]:
    """Return base + delta, clamping each value to [0, 1]."""
    out = dict(base)
    for k, v in delta.items():
        out[k] = max(0.0, min(1.0, out.get(k, 0.0) + v))
    return out


# ---------------------------------------------------------------------------
# Derived presets (for backward compatibility — generated from deltas)
# ---------------------------------------------------------------------------

def _build_cognitive_presets() -> dict[str, CognitiveParameters]:
    out: dict[str, CognitiveParameters] = {}
    for name in PROFILE_DELTAS:
        combined = _combine_deltas(name)
        out[name] = _apply_delta_to_cognitive(combined["cognitive"])
    return out


def _build_emotional_presets() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name in PROFILE_DELTAS:
        combined = _combine_deltas(name)
        out[name] = _apply_delta_to_dict(BASE_EMOTIONAL, combined["emotional"])
    return out


def _build_observable_presets() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name in PROFILE_DELTAS:
        combined = _combine_deltas(name)
        out[name] = _apply_delta_to_dict(BASE_OBSERVABLE, combined["observable"])
    return out


COGNITIVE_PRESETS: dict[str, CognitiveParameters] = _build_cognitive_presets()
EMOTIONAL_PRESETS: dict[str, dict[str, float]] = _build_emotional_presets()
OBSERVABLE_PRESETS: dict[str, dict[str, float]] = _build_observable_presets()


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

        # Observable state (what teacher can see) — profile-specific initial
        # values from dimensional delta model (fixes previous bug where all
        # profiles started with identical observable state)
        self.state: dict[str, float] = dict(
            OBSERVABLE_PRESETS.get(profile_type, OBSERVABLE_PRESETS.get("normal_quiet", BASE_OBSERVABLE))
        )
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
