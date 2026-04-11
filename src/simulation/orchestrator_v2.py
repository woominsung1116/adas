"""
orchestrator_v2.py -- 950-turn classroom simulation orchestrator.

Connects:
  - ClassroomV2          (classroom_env_v2.py)   950-turn env with CognitiveStudent agents
  - TeacherMemory        (teacher_memory.py)     Case Base + Experience Base, persists across classes
  - InteractionLog       (interaction_log.py)    records all events
  - IdentificationEvaluator (identification_report.py) DSM-5 reports
  - GrowthTracker        (growth_metrics.py)     cross-class metrics
  - TeacherLLM           (teacher_llm.py)        optional LLM for teacher decisions

Usage (rule-based, no LLM):
    orch = OrchestratorV2(n_students=20, max_classes=5, seed=42)
    for result in orch.run():
        m = result["result"]["metrics"]
        print(f'Class {result["class_id"]}: TP={m.true_positives} FP={m.false_positives}')

Usage (with LLM backend):
    orch = OrchestratorV2(llm_backend=my_backend, n_students=20)
    for result in orch.run():
        ...
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Generator, Optional, Dict, List


# ---------------------------------------------------------------------------
# Teacher Emotional State — 7-dimension literature-grounded model
# ---------------------------------------------------------------------------
#
# Design principles (see 정리.md §25.11):
#   - Initial values: fixed from Korean teacher normative data (no autoresearch)
#   - Update dynamics: 100% rule-based with literature-derived coefficients
#   - LLM does NOT update teacher emotions (only reads them as decision context)
#   - Strict reproducibility — identical inputs produce identical outputs
#
# References:
#   - Maslach & Jackson (1981) MBI; Maslach et al. (1996) MBI Manual
#   - Tschannen-Moran & Woolfolk Hoy (2001) TSES
#   - Klassen & Chiu (2010) teacher stress and self-efficacy
#   - Hargreaves (2000) teacher emotions
#   - 이영만 (2012) 초등교사 심리적 소진
#   - 조환이 & 김정환 (2016) 한국 초등교사 소진
#   - Goroshit & Hen (2016) teacher empathy
# ---------------------------------------------------------------------------

# Korean elementary teacher normative baseline (literature-fixed)
BASE_TEACHER_EMOTIONAL: dict[str, float] = {
    "emotional_exhaustion":    0.42,  # MBI-EE / 이영만 2012 M=2.1/5
    "depersonalization":       0.30,  # MBI-DP / 조환이 2016 M=1.5/5
    "personal_accomplishment": 0.72,  # MBI-PA / 한국 교사 M=3.6/5
    "self_efficacy":           0.72,  # TSES / Klassen 2009 M=6.5/9
    "empathy":                 0.76,  # Goroshit 2016 M=3.8/5
    "patience":                0.72,  # Schnitker 2012 추정
    "job_stress":              0.62,  # Kyriacou / 한국 M=3.1/5
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass
class TeacherEmotionalState:
    """7-dimension literature-grounded teacher emotional state.

    Maintains strict reproducibility — all updates are deterministic
    rule-based functions of (event_type, magnitude, context). LLM agents
    may read this state as decision context but must not modify it.
    """

    # 7 dimensions (initial values from literature)
    emotional_exhaustion: float = 0.42
    depersonalization: float = 0.30
    personal_accomplishment: float = 0.72
    self_efficacy: float = 0.72
    empathy: float = 0.76
    patience: float = 0.72
    job_stress: float = 0.62

    # Legacy aliases (backward compat with older code)
    bias: dict = field(default_factory=dict)

    # -----------------------------------------------------------------
    # Event handlers — each rule's coefficients are literature-derived
    # -----------------------------------------------------------------

    def on_student_incident(self, n_incidents: int = 1) -> None:
        """Minor student incidents (off-task, disruption).

        Source: Klassen & Chiu 2010 — incidents ↔ stress/patience.
        """
        self.patience = _clamp01(self.patience - 0.02 * n_incidents)
        self.job_stress = _clamp01(self.job_stress + 0.01 * n_incidents)

    def on_identification_success(self) -> None:
        """Correct ADHD identification (TP).

        Source: Tschannen-Moran 2001 TSES validation — successful teaching tasks
        boost self-efficacy and personal accomplishment.
        """
        self.self_efficacy = _clamp01(self.self_efficacy + 0.03)
        self.personal_accomplishment = _clamp01(self.personal_accomplishment + 0.02)

    def on_identification_failure(self) -> None:
        """Missed ADHD case (FN) or false positive (FP).

        Source: Maslach 1981 — failure experiences accumulate exhaustion.
        """
        self.self_efficacy = _clamp01(self.self_efficacy - 0.02)
        self.emotional_exhaustion = _clamp01(self.emotional_exhaustion + 0.01)

    def on_chaotic_mood(self) -> None:
        """Sustained chaotic/tense classroom atmosphere (per-turn drain).

        Source: Hargreaves 2000 — negative classroom mood erodes patience.
        """
        self.patience = _clamp01(self.patience - 0.01)
        self.job_stress = _clamp01(self.job_stress + 0.01)

    def on_calm_mood(self) -> None:
        """Calm classroom — gradual patience recovery."""
        self.patience = _clamp01(self.patience + 0.005)

    def daily_recovery(self) -> None:
        """Overnight recovery (start of each day).

        Source: Maslach burnout recovery curves — partial overnight restoration.
        """
        self.emotional_exhaustion = _clamp01(self.emotional_exhaustion - 0.02)
        self.patience = _clamp01(self.patience + 0.02)
        self.job_stress = _clamp01(self.job_stress - 0.01)

    def on_semester_start(self) -> None:
        """Reset to baseline at semester start (Klassen 2009)."""
        for key, val in BASE_TEACHER_EMOTIONAL.items():
            setattr(self, key, val)

    def on_public_correction(self) -> None:
        """Public correction used — cognitive dissonance slightly erodes empathy."""
        self.empathy = _clamp01(self.empathy - 0.01)

    def on_empathic_intervention_success(self) -> None:
        """Empathic intervention worked (student compliance improved).

        Source: Goroshit 2016 — empathic success reinforces empathy capacity.
        """
        self.empathy = _clamp01(self.empathy + 0.01)
        self.personal_accomplishment = _clamp01(self.personal_accomplishment + 0.01)

    def on_major_student_crisis(self) -> None:
        """Major crisis event (severe conflict, emergency)."""
        self.emotional_exhaustion = _clamp01(self.emotional_exhaustion + 0.05)
        self.patience = _clamp01(self.patience - 0.05)

    def on_parent_communication_success(self) -> None:
        """Positive parent communication boosts self-efficacy."""
        self.self_efficacy = _clamp01(self.self_efficacy + 0.02)

    def on_conflict_resolved(self) -> None:
        """Teacher successfully resolves peer conflict."""
        self.personal_accomplishment = _clamp01(self.personal_accomplishment + 0.02)

    def on_exam_week_start(self) -> None:
        """Exam week onset — sustained job stress increase."""
        self.job_stress = _clamp01(self.job_stress + 0.05)

    def apply_burnout_acceleration(self) -> None:
        """When EE > 0.7, accelerate decay of positive emotions (Maslach clinical)."""
        if self.emotional_exhaustion > 0.7:
            self.personal_accomplishment = _clamp01(self.personal_accomplishment - 0.005)
            self.empathy = _clamp01(self.empathy - 0.003)
            self.self_efficacy = _clamp01(self.self_efficacy - 0.003)
            self.patience = _clamp01(self.patience - 0.005)

    # -----------------------------------------------------------------
    # Dispatch table — string-based event handling for simulation loops
    # -----------------------------------------------------------------

    def update(self, event_type: str, magnitude: float = 1.0, **context) -> None:
        """Dispatch an event by name.

        event_type must match one of the on_* methods (without the on_ prefix).
        magnitude scales the default coefficient for incident-type events.
        """
        if event_type == "student_incident":
            self.on_student_incident(n_incidents=int(magnitude))
        elif event_type == "identification_success":
            self.on_identification_success()
        elif event_type == "identification_failure":
            self.on_identification_failure()
        elif event_type == "chaotic_mood":
            self.on_chaotic_mood()
        elif event_type == "calm_mood":
            self.on_calm_mood()
        elif event_type == "daily_recovery":
            self.daily_recovery()
        elif event_type == "semester_start":
            self.on_semester_start()
        elif event_type == "public_correction":
            self.on_public_correction()
        elif event_type == "empathic_intervention_success":
            self.on_empathic_intervention_success()
        elif event_type == "major_student_crisis":
            self.on_major_student_crisis()
        elif event_type == "parent_communication_success":
            self.on_parent_communication_success()
        elif event_type == "conflict_resolved":
            self.on_conflict_resolved()
        elif event_type == "exam_week_start":
            self.on_exam_week_start()
        # Unknown events → no-op (silent)

        # Always apply burnout acceleration as post-hook
        self.apply_burnout_acceleration()

    # -----------------------------------------------------------------
    # Legacy compatibility (used by existing orchestrator code)
    # -----------------------------------------------------------------

    def update_after_turn(self, classroom_mood: str, n_incidents: int) -> None:
        """Legacy interface — translates to new event handlers."""
        if n_incidents > 0:
            self.on_student_incident(n_incidents)
        if classroom_mood in ("chaotic", "tense"):
            self.on_chaotic_mood()
        elif classroom_mood == "calm":
            self.on_calm_mood()
        self.apply_burnout_acceleration()

    def observation_accuracy(self) -> float:
        """How accurately the teacher reads student emotional state.

        Derived from empathy, adjusted by emotional exhaustion.
        """
        return _clamp01(self.empathy * (1.0 - self.emotional_exhaustion * 0.3))

    def is_burned_out(self) -> bool:
        """Teacher is burned out when emotional exhaustion > 0.7 or patience < 0.3."""
        return self.emotional_exhaustion > 0.7 or self.patience < 0.3

    # Legacy aliases (old code refers to .frustration / .empathy_capacity)
    @property
    def frustration(self) -> float:
        """Legacy alias — maps to emotional_exhaustion."""
        return self.emotional_exhaustion

    @frustration.setter
    def frustration(self, value: float) -> None:
        self.emotional_exhaustion = _clamp01(value)

    @property
    def empathy_capacity(self) -> float:
        """Legacy alias — maps to empathy."""
        return self.empathy

    @empathy_capacity.setter
    def empathy_capacity(self, value: float) -> None:
        self.empathy = _clamp01(value)

    def asdict(self) -> dict:
        """Export current state as dict (for logging, LLM prompts)."""
        return {
            "emotional_exhaustion": self.emotional_exhaustion,
            "depersonalization": self.depersonalization,
            "personal_accomplishment": self.personal_accomplishment,
            "self_efficacy": self.self_efficacy,
            "empathy": self.empathy,
            "patience": self.patience,
            "job_stress": self.job_stress,
        }


# ---------------------------------------------------------------------------
# Configurable phase boundaries for the 5-phase teacher strategy
# ---------------------------------------------------------------------------

@dataclass
class PhaseConfig:
    """Configurable phase boundaries for the 5-phase teacher strategy.

    Defaults match the original 950-turn timeline:
      Phase 1 (1..observation_end): Pure observation
      Phase 2 (observation_end+1..screening_end): Screening
      Phase 3 (screening_end+1..identification_end): Identification
      Phase 4 (identification_end+1..care_end): Care
      Phase 5 (care_end+1..950): Maintenance + Relapse
    """
    observation_end: int = 100      # Phase 1 -> 2
    screening_end: int = 300        # Phase 2 -> 3
    identification_end: int = 475   # Phase 3 -> 4
    care_end: int = 700             # Phase 4 -> 5
    # Phase 5 runs until class ends (950)

# ---------------------------------------------------------------------------
# Core simulation modules
# ---------------------------------------------------------------------------

try:
    from src.simulation.classroom_env_v2 import (
        ClassroomV2,
        TeacherAction,
        ClassroomObservation,
        StudentSummary,
        DetailedObservation,
        MANAGED_COMPLIANCE,
        MANAGED_CONSECUTIVE,
        CLASSROOM_ARCHETYPES,
    )
except ImportError as e:
    raise ImportError(
        f"classroom_env_v2 not found or broken: {e}. "
        "Ensure src/simulation/classroom_env_v2.py exists."
    ) from e

try:
    from src.simulation.teacher_memory import (
        TeacherMemory,
        ObservationOutcome,
        PendingObservationFeedback,
        FeedbackDelayQueue,
        HYPERACTIVITY_BEHAVIORS,
        IMPULSIVITY_BEHAVIORS,
        INATTENTION_BEHAVIORS,
    )
except ImportError as e:
    raise ImportError(
        f"teacher_memory not found or broken: {e}. "
        "Ensure src/simulation/teacher_memory.py exists."
    ) from e

try:
    from src.simulation.interaction_log import InteractionLog, InteractionEvent
except ImportError as e:
    raise ImportError(
        f"interaction_log not found or broken: {e}. "
        "Ensure src/simulation/interaction_log.py exists."
    ) from e

try:
    from src.simulation.teacher_observation import (
        TeacherObservationBatch,
        TeacherHypothesisBoard,
        build_observations_from_classroom,
        canonicalize_hypothesis_label,
    )
except ImportError as e:
    raise ImportError(
        f"teacher_observation not found or broken: {e}. "
        "Ensure src/simulation/teacher_observation.py exists."
    ) from e

try:
    from src.eval.identification_report import (
        IdentificationReport,
        IdentificationEvaluator,
        ObservedSymptom,
        DSM5_INATTENTION,
        DSM5_HYPERACTIVITY,
    )
except ImportError as e:
    raise ImportError(
        f"identification_report not found or broken: {e}. "
        "Ensure src/eval/identification_report.py exists."
    ) from e

try:
    from src.eval.growth_metrics import GrowthTracker, ClassMetrics
except ImportError as e:
    raise ImportError(
        f"growth_metrics not found or broken: {e}. "
        "Ensure src/eval/growth_metrics.py exists."
    ) from e

# Optional LLM teacher -- graceful fallback
try:
    from src.llm.teacher_llm import TeacherLLM
except ImportError:
    TeacherLLM = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Behavior -> DSM-5 criterion mapping
# ---------------------------------------------------------------------------

_BEHAVIOR_TO_DSM5: dict[str, str] = {
    # Inattention
    "careless-mistakes":           "inattention_1",
    "not-following-instructions":  "inattention_4",
    "incomplete-tasks":            "inattention_4",
    "poor-organization":           "inattention_5",
    "easily-distracted":           "inattention_8",
    "off_task":                    "inattention_2",
    "forgetting_instructions":     "inattention_9",
    "staring_out_window":          "inattention_8",
    "losing_materials":            "inattention_7",
    "not_starting_task":           "inattention_6",
    "daydreaming":                 "inattention_2",
    "loses_materials":             "inattention_7",
    "off-task":                    "inattention_2",
    # Hyperactivity / Impulsivity
    "seat-leaving":                "hyperactivity_2",
    "out_of_seat":                 "hyperactivity_2",
    "running/climbing":            "hyperactivity_3",
    "running_in_classroom":        "hyperactivity_3",
    "leg-swinging":                "hyperactivity_1",
    "paper-folding":               "hyperactivity_1",
    "excessive-talking":           "hyperactivity_6",
    "excessive_talking":           "hyperactivity_6",
    "blurting-answers":            "hyperactivity_7",
    "blurting":                    "hyperactivity_7",
    "interrupting":                "hyperactivity_9",
    "off-topic-comments":          "hyperactivity_9",
    "grabbing-objects":            "hyperactivity_9",
    "calling_out":                 "hyperactivity_7",
    "fidgeting":                   "hyperactivity_1",
    "impulsive_response":          "hyperactivity_7",
}

# Map env behavior strings -> teacher_memory ALL_BEHAVIORS vocabulary.
_ENV_TO_MEM_BEHAVIOR: dict[str, str] = {
    # v1 env behavior names
    "out_of_seat":            "seat-leaving",
    "calling_out":            "blurting-answers",
    "blurting":               "blurting-answers",
    "interrupting":           "interrupting",
    "fidgeting":              "leg-swinging",
    "fidgeting_slightly":     "leg-swinging",
    "running_in_classroom":   "running/climbing",
    "excessive_talking":      "excessive-talking",
    "off_task":               "easily-distracted",
    "daydreaming":            "easily-distracted",
    "staring_out_window":     "easily-distracted",
    "forgetting_instructions": "not-following-instructions",
    "losing_materials":       "poor-organization",
    "not_starting_task":      "incomplete-tasks",
    "impulsive_response":     "blurting-answers",
    # v2 cognitive_agent behavior names (underscore → hyphen mapping)
    "seat_leaving":              "seat-leaving",
    "leg_swinging":              "leg-swinging",
    "paper_folding":             "paper-folding",
    "running_climbing":          "running/climbing",
    "blurting_answers":          "blurting-answers",
    "off_topic_comments":        "off-topic-comments",
    "grabbing_objects":          "grabbing-objects",
    "careless_mistakes":         "careless-mistakes",
    "not_following_instructions": "not-following-instructions",
    "incomplete_tasks":          "incomplete-tasks",
    "poor_organization":         "poor-organization",
    "easily_distracted":         "easily-distracted",
    # Additional cognitive_agent behaviors
    "looking_around":            "easily-distracted",
    "boredom_fidgeting":         "leg-swinging",
}


def _translate_behaviors(behaviors: list[str]) -> list[str]:
    """Translate env behavior strings to teacher_memory vocabulary."""
    return [_ENV_TO_MEM_BEHAVIOR.get(b, b) for b in behaviors]


def _behaviors_to_dsm5(
    behaviors: list[str],
    turns_seen: dict[str, list[int]],
    current_turn: int,
) -> tuple[list[ObservedSymptom], list[ObservedSymptom]]:
    """Convert behavior strings to DSM-5 ObservedSymptom lists.

    Returns (inattention_symptoms, hyperactivity_symptoms).
    """
    inattention: dict[str, tuple[str, list[int]]] = {}
    hyperactivity: dict[str, tuple[str, list[int]]] = {}

    for b in behaviors:
        criterion = _BEHAVIOR_TO_DSM5.get(b)
        if criterion is None:
            continue
        turns = turns_seen.get(b, [current_turn])
        if criterion.startswith("inattention_"):
            if criterion not in inattention:
                inattention[criterion] = (b, list(turns))
            else:
                inattention[criterion][1].extend(turns)
        elif criterion.startswith("hyperactivity_"):
            if criterion not in hyperactivity:
                hyperactivity[criterion] = (b, list(turns))
            else:
                hyperactivity[criterion][1].extend(turns)

    def build_symptoms(mapping: dict[str, tuple[str, list[int]]]) -> list[ObservedSymptom]:
        return [
            ObservedSymptom.from_observations(criterion, behavior, turns_list)
            for criterion, (behavior, turns_list) in mapping.items()
        ]

    return build_symptoms(inattention), build_symptoms(hyperactivity)


# ---------------------------------------------------------------------------
# Per-student tracking state (internal to orchestrator)
# ---------------------------------------------------------------------------

@dataclass
class _StudentTrack:
    """Orchestrator-internal accumulator per student per class."""
    student_id: str
    all_behaviors: list[str] = field(default_factory=list)
    turns_per_behavior: dict[str, list[int]] = field(default_factory=dict)
    strategies_applied: list[str] = field(default_factory=list)
    initial_compliance: float = 0.6
    compliance_history: list[float] = field(default_factory=list)
    identification_turn: int = 0
    observation_count: int = 0


# ---------------------------------------------------------------------------
# Hypothesis-Verification Tracker (Phase 2 sub-phases)
# ---------------------------------------------------------------------------

# Intervention strategies used for hypothesis testing in Phase 2b
_HYPOTHESIS_TEST_STRATEGIES: list[str] = [
    "empathic_acknowledgment",  # anxiety differentiator
    "break_offer",              # ADHD differentiator
    "firm_boundary",            # ODD differentiator
]


@dataclass
class HypothesisTracker:
    """Track hypothesis-verification tests for a suspicious student.

    A tracker is created when a student first enters the teacher's
    WORKING SUSPICION SET — see ``_decide_action_rule_based`` Phase
    2a, which adds a student to ``_stream_suspicious`` when
    ``adhd_indicator_score >= 0.15`` with at least 3 observations
    (strong entry) or ``> 0.10`` (soft entry, no tracker created).
    This is deliberately loose — it is "teacher started paying
    closer attention", not a strong diagnostic threshold.

    Phase 2 sub-phases:
      2a: Enter working suspicion set (see above)
      2b: Apply differential interventions and record compliance deltas
      2c: After 3+ different tests, infer likely profile from response pattern
    """

    student_id: str
    suspicion_turn: int
    tests_applied: Dict[str, List[float]] = field(default_factory=dict)  # strategy -> [compliance_deltas]
    diagnosis_hypothesis: str = "unknown"  # adhd_inattentive | adhd_hyperactive | anxiety | odd | unknown

    def record_test(self, strategy: str, compliance_delta: float) -> None:
        """Record the compliance delta from a hypothesis test intervention."""
        self.tests_applied.setdefault(strategy, []).append(compliance_delta)

    def can_identify(self) -> bool:
        """Require at least 3 different intervention tests before identification."""
        return len(self.tests_applied) >= 3

    def likely_profile(self) -> str:
        """Infer likely profile from intervention response pattern.

        Decision logic:
          - empathic_acknowledgment helps a lot (avg delta > 0.05) -> anxiety
          - break_offer helps (avg delta > 0.03) -> ADHD
          - firm_boundary causes defiance (avg delta < -0.03) -> ODD
          - break helps + firm_boundary negative -> ADHD combined
          - mixed/ambiguous -> unknown
        """
        def avg_delta(strategy: str) -> float:
            deltas = self.tests_applied.get(strategy, [])
            return sum(deltas) / len(deltas) if deltas else 0.0

        empathic_avg = avg_delta("empathic_acknowledgment")
        break_avg = avg_delta("break_offer")
        firm_avg = avg_delta("firm_boundary")

        # Strong empathic response -> likely anxiety, not ADHD
        if empathic_avg > 0.05 and break_avg <= 0.03:
            self.diagnosis_hypothesis = "anxiety"
            return "anxiety"

        # Firm boundary causes defiance, break doesn't help much -> ODD
        if firm_avg < -0.03 and break_avg <= 0.02:
            self.diagnosis_hypothesis = "odd"
            return "odd"

        # Break helps, firm boundary negative or neutral -> ADHD
        if break_avg > 0.03:
            if firm_avg < -0.01:
                self.diagnosis_hypothesis = "adhd_hyperactive"
            else:
                self.diagnosis_hypothesis = "adhd_inattentive"
            return self.diagnosis_hypothesis

        self.diagnosis_hypothesis = "unknown"
        return "unknown"


# ---------------------------------------------------------------------------
# Observable-behavior proxies for strategy selection
# ---------------------------------------------------------------------------

#: High-visibility disruptive behaviors the teacher can see from the
#: front of the room. Mirrors ``ClassroomV2._visible_behaviors`` so
#: the decision path and the visibility filter stay in sync. Used by
#: Phase 5 relapse detection and by ``_default_strategy``.
_DISRUPTIVE_VISIBLE_BEHAVIORS: frozenset[str] = frozenset({
    "out_of_seat",
    "calling_out",
    "interrupting",
    "excessive_talking",
    "running_in_classroom",
    "fidgeting",
    "emotional_outburst",
})

#: Observable low-arousal inattention behaviors. Note that the
#: current ClassroomV2._visible_behaviors filter passes NONE of
#: these through (the high-visibility set is all high-arousal
#: disruption). They are kept here so that if a future pass
#: widens the visibility filter, the strategy ladder already
#: recognizes them without another edit. The deliberately-latent
#: sentinels ``seems_inattentive`` / ``on_task`` / ``quiet`` are
#: NOT in this set — they are scrubbed by the teacher observation
#: builder before the decision path ever sees them.
_INATTENTIVE_VISIBLE_BEHAVIORS: frozenset[str] = frozenset({
    "staring_out_window",
    "daydreaming",
    "off_task",
    "off-task",
})


# ---------------------------------------------------------------------------
# Intervention strategy pool
# ---------------------------------------------------------------------------

STRATEGIES = [
    "transition_warning",
    "offer_choice",
    "labeled_praise",
    "visual_schedule_cue",
    "break_offer",
    "empathic_acknowledgment",
    "redirect_attention",
    "countdown_timer",
    "collaborative_problem_solving",
    "ignore_wait",
    "firm_boundary",
    "sensory_support",
]


# ---------------------------------------------------------------------------
# OrchestratorV2
# ---------------------------------------------------------------------------


class OrchestratorV2:
    """Open-ended simulation orchestrator for 950-turn classroom.

    Connects:
    - ClassroomV2 (950-turn env with cognitive student agents)
    - TeacherMemory (Case Base + Experience Base, persists across classes)
    - InteractionLog (records all events)
    - IdentificationEvaluator (DSM-5 reports)
    - GrowthTracker (cross-class metrics)
    - TeacherLLM (optional LLM for teacher decisions)
    """

    def __init__(
        self,
        n_students: int = 20,
        llm_backend: Any = None,
        max_classes: int | None = None,
        seed: int | None = None,
        phase_config: PhaseConfig | None = None,
        feedback_rate: float = 0.30,
        feedback_delay_turns: int = 1,
    ):
        self.log = InteractionLog()
        self.classroom = ClassroomV2(
            n_students=n_students, seed=seed, interaction_log=self.log,
        )
        self.memory = TeacherMemory(
            retrieval_noise=0.20,
            principle_promotion_threshold=7,
            principle_min_classes=3,
            memory_decay_rate=0.99,
            seed=seed,
        )
        self.evaluator = IdentificationEvaluator()
        self.growth = GrowthTracker()
        self.phase_config = phase_config or PhaseConfig()
        self.teacher_llm: Any = None
        if llm_backend and TeacherLLM is not None:
            self.teacher_llm = TeacherLLM(llm_backend, self.memory)
        self.class_count = 0
        self.max_classes = max_classes
        self.feedback_rate = feedback_rate
        self._rng = random.Random(seed)
        self.teacher_emotions = TeacherEmotionalState()
        # Per-class hypothesis trackers, reset each class
        self._hypothesis_trackers: dict[str, HypothesisTracker] = {}

        # Phase 6 slice 3: delayed feedback for memory commits.
        # Observations are staged each turn and committed to teacher
        # memory `feedback_delay_turns` turns later, through the
        # same observable-only derivation path. Fixed integer delay
        # keeps the simulator deterministic; stochastic delay is
        # deferred to a later pass.
        self.feedback_delay_turns: int = max(0, int(feedback_delay_turns))
        self._feedback_queue: FeedbackDelayQueue = FeedbackDelayQueue()
        # Phase 6 slice 1: per-class teacher observation + hypothesis board.
        # Reset inside stream_class; pre-initialized here so callers can
        # access these attributes before the first class runs.
        self.hypothesis_board: TeacherHypothesisBoard = TeacherHypothesisBoard()
        self._current_teacher_obs: TeacherObservationBatch | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Generator[dict, None, None]:
        """Run open-ended simulation. Yields events per class."""
        while self.max_classes is None or self.class_count < self.max_classes:
            class_result = self.run_class()
            self.class_count += 1
            self.growth.record_class(class_result["metrics"])
            yield {
                "type": "class_complete",
                "class_id": self.class_count,
                "result": class_result,
                "growth_summary": self.growth.summary(),
            }

    # ------------------------------------------------------------------
    # Single class (950 turns)
    # ------------------------------------------------------------------

    def stream_class(self) -> Generator[dict, None, None]:
        """Generator that yields per-turn events for real-time streaming.

        Yields:
            dict with "type": "new_class" as the first event (student roster)
            dict with "type": "turn" for each turn
            dict with "type": "class_complete" as final yield
        """
        # Pick a classroom archetype for this class.
        # If the classroom already has a fixed archetype (set via
        # ClassroomV2(archetype=...) or set_archetype(...)), preserve it
        # so calibration / held-out validation can pin archetypes
        # deterministically. Otherwise pick randomly.
        if getattr(self.classroom, "_archetype_name", None) is None:
            archetype = self._rng.choice(list(CLASSROOM_ARCHETYPES.keys()))
            self.classroom.set_archetype(archetype)

        obs = self.classroom.reset()
        self.memory.new_class()

        # First yield: new_class event with student roster and archetype
        arch = self.classroom.archetype
        yield {
            "type": "new_class",
            "class_id": self.class_count + 1,
            "n_students": len(self.classroom.students),
            "max_turns": self.classroom.MAX_TURNS,
            "archetype": arch.name if arch else "unknown",
            "archetype_description": arch.description if arch else "",
            "students": [
                {
                    "id": s.student_id,
                    "profile_type": s.profile_type,
                    "gender": getattr(s, "gender", "unknown"),
                    "is_adhd": s.is_adhd,
                }
                for s in self.classroom.students
            ],
        }

        # Per-student accumulators (stored as instance attrs so they persist across yields)
        self._stream_tracks: dict[str, _StudentTrack] = {
            s.student_id: _StudentTrack(
                student_id=s.student_id,
                initial_compliance=s.state.get("compliance", 0.6),
            )
            for s in self.classroom.students
        }
        self._stream_events: list[dict] = []
        self._stream_reports: list[IdentificationReport] = []
        self._stream_identified: set[str] = set()
        self._stream_ruled_out: set[str] = set()
        self._stream_suspicious: dict[str, float] = {}
        self._stream_strategies_used: set[str] = set()
        self._stream_identification_turns: list[float] = []
        self._stream_final_turn = 0
        self._stream_obs = obs
        self._hypothesis_trackers = {}  # reset per class

        # Calibration-oriented per-turn accumulators (exported to ClassHistory).
        # - patience_log: teacher patience value sampled each turn; used by
        #   teacher.patience_end_of_day_vs_start_ratio
        # - intervention_outcomes: per-intervention pre/post compliance deltas;
        #   used by intervention.empathic_compliance_gain
        # - first_suspicion_turns: turn at which each student first
        #   entered the teacher's WORKING SUSPICION SET. The working
        #   set is the orchestrator's refresh-time structure
        #   `_stream_suspicious`, populated when
        #   `adhd_indicator_score >= 0.15` with at least 3 observations
        #   (strong entry) OR `> 0.10` (soft entry, no tracker).
        #   Semantic: this is "teacher started paying closer attention",
        #   not a strong diagnostic threshold crossing. Used by
        #   calibration metric `teacher.first_suspicion_turn_median`
        #   whose docstring now matches this definition.
        self._stream_patience_log: list[float] = []
        self._stream_intervention_outcomes: list[dict] = []
        self._stream_first_suspicion_turns: dict[str, int] = {}

        # Phase 6 slice 1: explicit teacher-facing partial observation
        # and per-student working hypothesis state. Built alongside the
        # existing decision path; future passes (hypothesis testing,
        # delayed feedback, memory noise) will consume these objects
        # instead of reading latent student state directly.
        self.hypothesis_board = TeacherHypothesisBoard()
        self._current_teacher_obs: TeacherObservationBatch | None = None

        # Phase 6 slice 3: reset the delayed-feedback queue per class
        # so stale cross-class pending items do not bleed into a
        # fresh run. Each class starts with an empty queue.
        self._feedback_queue = FeedbackDelayQueue()

        for turn in range(1, self.classroom.MAX_TURNS + 1):
            self.memory.advance_turn()
            self._stream_final_turn = turn

            # Teacher daily recovery (at start of each day, period 1)
            if turn % 5 == 1:
                self.teacher_emotions.daily_recovery()

            # 0. Phase 6 slice 3: drain the delayed-feedback queue
            # BEFORE the teacher decides. This simulates the teacher
            # "noticing yesterday's result" first — any pending
            # observation whose due_turn has arrived is now derived
            # through the observable-only feedback path and
            # committed to the case base, so the next decision is
            # informed by matured memory rather than instantaneous
            # reinforcement.
            self._drain_feedback_queue(current_turn=turn)

            # 1. Teacher decides action
            prev_suspicious_ids = set(self._stream_suspicious.keys())

            # 1a. Phase 6 slice 1: project the raw ClassroomObservation
            # into an explicit partial-observation batch. The teacher
            # decision path and hypothesis board BOTH consume this
            # batch instead of reading latent student state.
            teacher_batch = build_observations_from_classroom(self._stream_obs)
            self._current_teacher_obs = teacher_batch
            self.hypothesis_board.record_batch(teacher_batch)

            action = self._decide_action(
                self._stream_obs, turn,
                self._stream_identified, self._stream_suspicious,
                teacher_batch=teacher_batch,
            )

            # 1b. Detect students that just entered the teacher's
            # WORKING SUSPICION SET and pin their first-working-
            # suspicion turn. "Working suspicion set" here is
            # deliberately loose — it corresponds to
            # `_stream_suspicious` entries produced by the
            # refresh-time rule (score >= 0.15 with ≥3 obs, or
            # > 0.10 soft). This is NOT a strong diagnostic
            # threshold; it is "teacher started paying closer
            # attention" and is calibrated against the target
            # range in `.harness/naturalness_targets.yaml`
            # (`teacher_first_suspicion_turn`, 50-120 turns).
            # The legacy dict is kept in sync with the
            # authoritative hypothesis board.
            new_suspicious = set(self._stream_suspicious.keys()) - prev_suspicious_ids
            for sid in new_suspicious:
                if sid not in self._stream_first_suspicion_turns:
                    self._stream_first_suspicion_turns[sid] = turn

            # 1c. Capture pre-intervention compliance for outcome tracking.
            pre_intervention_compliance: float | None = None
            pre_intervention_sid: str | None = None
            if (
                action.action_type == "individual_intervention"
                and action.student_id
                and action.strategy
            ):
                s_obj = self.classroom.get_student(action.student_id)
                if s_obj is not None:
                    pre_intervention_compliance = float(
                        s_obj.state.get("compliance", 0.5)
                    )
                    pre_intervention_sid = action.student_id

            # 2. Execute in environment
            self._stream_obs, reward, done, info = self.classroom.step(action)

            # 2b. Record intervention outcome (post-compliance now available).
            if (
                pre_intervention_sid is not None
                and pre_intervention_compliance is not None
            ):
                s_obj = self.classroom.get_student(pre_intervention_sid)
                if s_obj is not None:
                    post_compliance = float(s_obj.state.get("compliance", 0.5))
                    self._stream_intervention_outcomes.append({
                        "turn": turn,
                        "student_id": pre_intervention_sid,
                        "strategy": action.strategy,
                        "pre_compliance": pre_intervention_compliance,
                        "post_compliance": post_compliance,
                    })

            # 3. Update teacher memory and tracks
            self._update_memory(self._stream_obs, action, info, self._stream_tracks, turn)

            # 3b. Synchronize the hypothesis board with the turn-end
            # suspicion / diagnosis state. This runs every turn so the
            # board reflects the *current* teacher-side hypothesis,
            # not just the first moment a student entered the
            # suspicious set. First-crossing semantics for
            # ``first_suspicion_turn`` are preserved by
            # ``record_suspicion`` (only sets once).
            self._sync_hypothesis_board(turn)

            # 4. Handle identification actions
            if action.action_type == "identify_adhd" and action.student_id:
                if action.student_id not in self._stream_identified:
                    report = self._build_report(
                        student_id=action.student_id,
                        turn=turn,
                        tracks=self._stream_tracks,
                        action=action,
                    )
                    if report:
                        self._stream_reports.append(report)
                        self._stream_identified.add(action.student_id)
                        self._stream_identification_turns.append(float(turn))
                        # Mark student identified in env
                        student = self.classroom.get_student(action.student_id)
                        if student:
                            student.identified = True
                            self.classroom.identified_adhd_ids.add(action.student_id)

            # 5. Track strategies
            if action.strategy:
                self._stream_strategies_used.add(action.strategy)
            if action.action_type in ("private_correction", "public_correction"):
                self._stream_strategies_used.add(action.action_type)

            # 6. Track compliance history
            for s in self.classroom.students:
                self._stream_tracks[s.student_id].compliance_history.append(
                    s.state.get("compliance", 0.6)
                )

            # 7. Log teacher action as interaction event
            self._log_teacher_action(action, self._stream_obs, turn)

            # 7b. Update teacher emotional state
            n_incidents = len(info.get("interactions", []))
            self.teacher_emotions.update_after_turn(
                self._stream_obs.class_mood, n_incidents
            )

            # 7c. Sample teacher patience (post-update) for calibration metrics
            self._stream_patience_log.append(float(self.teacher_emotions.patience))

            # 8. Build and yield turn event
            event = self._make_event(
                turn, self._stream_obs, action, info, reward, self._stream_identified
            )
            self._stream_events.append(event)
            yield event

            if done:
                break

        # Phase 6 slice 3: flush any pending delayed-feedback items
        # that were staged on the final turns. Without this, tail-end
        # observations (staged on turns >= MAX_TURNS - delay) would
        # never reach memory because no subsequent turn runs a drain
        # step. Matters for small unit-test classes where MAX_TURNS
        # is tiny; harmless in 950-turn runs.
        self._flush_feedback_queue()

        # Final: compile and yield class_complete
        result = self._compile_class_result(
            obs=self._stream_obs,
            turn=self._stream_final_turn,
            tracks=self._stream_tracks,
            reports=self._stream_reports,
            events=self._stream_events,
            strategies_used=self._stream_strategies_used,
            identification_turns=self._stream_identification_turns,
            identified_students=self._stream_identified,
        )
        yield {
            "type": "class_complete",
            "class_id": self.class_count + 1,
            "metrics": result["metrics"],
            "reports": result["reports"],
            "_result": result,  # carry full result for run_class() wrapper
        }

    def run_class(self) -> dict:
        """Batch-compatible wrapper. Consumes stream_class() and returns final result."""
        last_event = None
        for event in self.stream_class():
            if event.get("type") == "class_complete":
                last_event = event

        if last_event is not None:
            return last_event["_result"]

        # Fallback (should not be reached)
        from src.eval.growth_metrics import ClassMetrics
        return {
            "metrics": ClassMetrics(
                class_id=self.class_count + 1,
                n_students=len(self.classroom.students),
            ),
            "events": [],
            "reports": [],
        }

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _decide_action(
        self,
        obs: ClassroomObservation,
        turn: int,
        identified: set[str],
        suspicious: dict[str, float],
        *,
        teacher_batch: TeacherObservationBatch | None = None,
    ) -> TeacherAction:
        """Dispatch to LLM or rule-based teacher.

        ``teacher_batch`` is the Phase 6 slice 1 partial-observation
        batch for this turn. If the caller does not supply one, the
        method builds a batch from ``obs`` on the fly so legacy
        callers / test harnesses still work. The rule-based and LLM
        paths both consume the batch rather than reading latent
        student state directly.
        """
        if teacher_batch is None:
            teacher_batch = build_observations_from_classroom(obs)
        if self.teacher_llm:
            return self._decide_action_llm(obs, turn, teacher_batch=teacher_batch)
        return self._decide_action_rule_based(
            obs, turn, identified, suspicious,
            teacher_batch=teacher_batch,
        )

    def _sync_hypothesis_board(self, turn: int) -> None:
        """Mirror the turn-end suspicion/hypothesis state onto the board.

        Runs every turn (not only on first crossing) so the board
        reflects the current teacher-side hypothesis, including:

          * updated ``suspicion_score`` drawn from ``_stream_suspicious``
          * updated ``working_label`` drawn from
            ``_hypothesis_trackers[sid].diagnosis_hypothesis`` and
            canonicalized via ``canonicalize_hypothesis_label`` so
            legacy short forms (e.g. ``"adhd_hyperactive"``) do not
            crash ``TeacherHypothesis.set_working_label``
          * ``first_suspicion_turn`` preserved by the board — it is
            only set on the first call where the score meets the
            threshold (``0.0`` here, i.e. any recorded suspicion
            counts as a first crossing, matching the legacy
            set-difference semantics)

        Students that have dropped out of ``_stream_suspicious`` keep
        their last-recorded score; we do NOT silently zero them, so
        the history trail stays honest.
        """
        trackers = self._hypothesis_trackers
        for sid, score in self._stream_suspicious.items():
            tracker = trackers.get(sid)
            raw_label = (
                tracker.diagnosis_hypothesis
                if tracker and tracker.diagnosis_hypothesis
                else "unknown"
            )
            canonical_label = canonicalize_hypothesis_label(raw_label)
            self.hypothesis_board.record_suspicion(
                sid,
                turn=turn,
                score=float(score),
                threshold=0.0,
                working_label=canonical_label,
            )

    def _decide_action_rule_based(
        self,
        obs: ClassroomObservation,
        turn: int,
        identified: set[str],
        suspicious: dict[str, float],
        *,
        teacher_batch: TeacherObservationBatch | None = None,
    ) -> TeacherAction:
        """
        5-phase teacher strategy aligned with 950-turn timeline.

        Phase boundaries are configurable via self.phase_config (PhaseConfig).

        Phase 1 (Turn 1..observation_end): Pure observation
            Rotate observe() through all students. Build baseline memory.
        Phase 2 (observation_end+1..screening_end): Screening
            Focus on students with suspicious patterns. Build suspicion list.
        Phase 3 (screening_end+1..identification_end): Identification
            Formally identify students with confidence >= 0.85.
        Phase 4 (identification_end+1..care_end): Care
            Apply interventions to identified students.
        Phase 5 (care_end+1..950): Maintenance + Relapse
            Monitor managed students for relapse. Re-intervene if needed.
        """
        pc = self.phase_config
        students = self.classroom.students
        n = len(students)

        # Teacher emotional state affects identification thresholds.
        # Burned-out teachers flag students more aggressively (lower threshold)
        # and prefer firm_boundary over empathic strategies.
        _id_threshold_modifier = 1.0
        if self.teacher_emotions.is_burned_out():
            _id_threshold_modifier = 0.8  # lower threshold = more aggressive

        # Phase 6 slice 1: build a per-student lookup of teacher-visible
        # cues. Decision logic reads from this dict instead of touching
        # student.state directly, enforcing the partial-observation
        # boundary. Missing sids return an empty observation so
        # downstream proxies degrade gracefully rather than raising.
        if teacher_batch is None:
            teacher_batch = build_observations_from_classroom(obs)
        _obs_lookup = teacher_batch.by_student_id()

        # ---- Phase 1: Pure observation (turns 1..observation_end) ----
        if turn <= pc.observation_end:
            idx = (turn - 1) % n
            sid = students[idx].student_id
            return TeacherAction(
                action_type="observe",
                student_id=sid,
                reasoning=f"Phase 1: baseline observation sweep turn {turn}",
            )

        # ---- Phase 2: Screening with hypothesis-verification sub-phases ----
        if turn <= pc.screening_end:
            # Phase 2a: Update suspicion scores, flag suspicious students
            if turn % 5 == 1:  # refresh every 5 turns
                for s in students:
                    if s.student_id in identified or s.student_id in self._stream_ruled_out:
                        continue
                    profile = self.memory.get_profile(s.student_id)
                    base_score = profile.adhd_indicator_score()
                    n_obs = sum(profile.behavior_frequency_counts.values())

                    # --- Fix 1 + Fix 4: Memory-informed suspicion scoring ---
                    dominant = profile.dominant_behaviors(top_k=5)
                    score = base_score

                    # Case-base prior: only query for students with non-trivial
                    # behavioral score to avoid expensive O(N) retrieval for
                    # clearly-normal students.
                    if base_score >= 0.10 and dominant:
                        labeled_records = self._get_labeled_cases(
                            dominant, s.student_id
                        )
                        if labeled_records:
                            case_adhd_rate = sum(
                                1 for _, rec in labeled_records if rec.was_adhd
                            ) / len(labeled_records)
                            score = base_score * 0.6 + case_adhd_rate * 0.4

                    # Experience-base principles adjust score
                    for p in self.memory.experience_base.top_principles(top_k=5):
                        if self.memory._principle_applies(p.text, dominant):
                            if p.is_corrective:
                                score *= 0.8
                            else:
                                score *= 1.2

                    score = min(1.0, score)

                    if score >= 0.15 and n_obs >= 3:
                        suspicious[s.student_id] = score
                        # Create hypothesis tracker if not exists
                        if s.student_id not in self._hypothesis_trackers:
                            self._hypothesis_trackers[s.student_id] = HypothesisTracker(
                                student_id=s.student_id,
                                suspicion_turn=turn,
                            )
                    elif score > 0.10:
                        suspicious[s.student_id] = score

            # Interleave hypothesis testing (2/3 turns) with discovery (1/3 turns)
            # This prevents the teacher from spending ALL Phase 2 on testing
            # known suspicious students while missing undiscovered ADHD students.
            is_discovery_turn = (turn % 3 == 0)

            if not is_discovery_turn:
                # Phase 2b: Hypothesis testing for tracked suspicious students
                for sid, tracker in self._hypothesis_trackers.items():
                    if sid in identified or sid in self._stream_ruled_out:
                        continue
                    student_obj = self.classroom.get_student(sid)
                    if student_obj is None:
                        continue
                    untested = [
                        strat for strat in _HYPOTHESIS_TEST_STRATEGIES
                        if strat not in tracker.tests_applied
                    ]
                    if untested:
                        strategy = untested[0]
                        return TeacherAction(
                            action_type="individual_intervention",
                            student_id=sid,
                            strategy=strategy,
                            reasoning=f"Phase 2b: hypothesis test '{strategy}' for {sid} "
                                      f"(tests done: {len(tracker.tests_applied)}/3)",
                        )

                # Focus on most suspicious unidentified student
                candidate = self._most_suspicious_student(identified)
                if candidate:
                    profile = self.memory.get_profile(candidate.student_id)
                    n_obs = sum(profile.behavior_frequency_counts.values())
                    return TeacherAction(
                        action_type="observe",
                        student_id=candidate.student_id,
                        reasoning=f"Phase 2: screening {candidate.student_id} "
                                  f"(score={suspicious.get(candidate.student_id, 0):.2f}, obs={n_obs})",
                    )

            # Discovery turn (or nothing suspicious) -- observe least-observed student
            # This prevents observation bias where only visible students get attention
            least_observed = None
            min_obs = float("inf")
            for s in students:
                if s.student_id in identified or s.student_id in self._stream_ruled_out:
                    continue
                profile = self.memory.get_profile(s.student_id)
                n_obs = sum(profile.behavior_frequency_counts.values()) if profile else 0
                if n_obs < min_obs:
                    min_obs = n_obs
                    least_observed = s
            if least_observed:
                return TeacherAction(
                    action_type="observe",
                    student_id=least_observed.student_id,
                    reasoning=f"Phase 2: observe least-seen student {least_observed.student_id} (obs={min_obs})",
                )
            # Fallback: rotate
            idx = (turn - 1) % n
            return TeacherAction(
                action_type="observe",
                student_id=students[idx].student_id,
                reasoning=f"Phase 2: general sweep turn {turn}",
            )

        # ---- Phase 3: Identification (turns screening_end+1..identification_end) ----
        if turn <= pc.identification_end:
            # Phase 2c check: complete any remaining hypothesis tests first
            for sid, tracker in self._hypothesis_trackers.items():
                if sid in identified or sid in self._stream_ruled_out:
                    continue
                untested = [
                    strat for strat in _HYPOTHESIS_TEST_STRATEGIES
                    if strat not in tracker.tests_applied
                ]
                if untested:
                    strategy = untested[0]
                    return TeacherAction(
                        action_type="individual_intervention",
                        student_id=sid,
                        strategy=strategy,
                        reasoning=f"Phase 3 (completing 2c): hypothesis test '{strategy}' for {sid}",
                    )

            # --- Memory-informed identification threshold ---
            # A teacher with more case-base experience (labeled records from
            # prior classes) needs less hypothesis evidence to be confident.
            # This is the KEY mechanism where memory improves over classes.
            n_labeled = sum(
                1 for rec in self.memory.case_base._records
                if rec.was_adhd is not None
            )
            # experience_boost: ramps from 0.0 (no prior labels) to 0.20
            # (500+ labeled records = ~2-3 prior classes with feedback)
            experience_boost = min(0.20, n_labeled / 2500.0)
            # With experience, identification threshold drops from 0.40 to 0.20
            id_threshold = max(0.20, 0.40 - experience_boost)

            # Try to identify high-confidence students
            candidate = self._most_suspicious_student(identified)
            if candidate:
                profile = self.memory.get_profile(candidate.student_id)
                score = profile.adhd_indicator_score()
                n_obs = sum(profile.behavior_frequency_counts.values())

                if score >= (0.20 * _id_threshold_modifier) and n_obs >= 5:
                    # Phase 2c: differential diagnosis gate
                    tracker = self._hypothesis_trackers.get(candidate.student_id)
                    if tracker and tracker.can_identify():
                        likely = tracker.likely_profile()
                        # Only identify if hypothesis points to ADHD
                        if likely.startswith("adhd") or likely == "unknown":
                            is_adhd, confidence, reasoning = self.memory.identify_adhd(
                                candidate.student_id
                            )
                            # Hypothesis test completion boosts confidence:
                            # passing 3 differential tests is strong evidence
                            if likely.startswith("adhd"):
                                confidence = min(1.0, confidence + 0.35)
                            # Memory-informed threshold: experienced teacher
                            # can identify with less confidence
                            if confidence >= id_threshold:
                                hypo_info = (
                                    f" [hypothesis={likely}, "
                                    f"threshold={id_threshold:.2f}, "
                                    f"experience_boost={experience_boost:.3f}]"
                                )
                                return TeacherAction(
                                    action_type="identify_adhd",
                                    student_id=candidate.student_id,
                                    reasoning=reasoning + hypo_info,
                                )
                            else:
                                # Low confidence after hypothesis testing:
                                # rule out after enough observation to avoid
                                # getting stuck on the same student forever
                                if n_obs >= 15:
                                    self._stream_ruled_out.add(candidate.student_id)
                                    return TeacherAction(
                                        action_type="class_instruction",
                                        reasoning=f"Phase 3: low confidence ({confidence:.2f}), "
                                                  f"ruled out {candidate.student_id} after {n_obs} obs",
                                    )
                        else:
                            # Hypothesis says anxiety/ODD, rule out as non-ADHD
                            self._stream_ruled_out.add(candidate.student_id)
                            return TeacherAction(
                                action_type="class_instruction",
                                reasoning=f"Phase 3: hypothesis={likely}, not ADHD, ruled out {candidate.student_id}",
                            )
                    elif tracker is None:
                        # No hypothesis tracker (low-suspicion path), use old logic
                        is_adhd, confidence, reasoning = self.memory.identify_adhd(
                            candidate.student_id
                        )
                        if confidence >= 0.85:
                            return TeacherAction(
                                action_type="identify_adhd",
                                student_id=candidate.student_id,
                                reasoning=reasoning,
                            )
                        elif n_obs >= 20:
                            # Too many observations with no result, move on
                            self._stream_ruled_out.add(candidate.student_id)
                            return TeacherAction(
                                action_type="class_instruction",
                                reasoning=f"Phase 3: no tracker, low confidence, ruled out {candidate.student_id}",
                            )

                # Not enough confidence yet, keep observing
                return TeacherAction(
                    action_type="observe",
                    student_id=candidate.student_id,
                    reasoning=f"Phase 3: deepening observation (score={score:.2f}, obs={n_obs})",
                )

            # Nothing to identify -- class instruction
            return TeacherAction(
                action_type="class_instruction",
                reasoning="Phase 3: no candidates, general instruction",
            )

        # ---- Phase 4: Care (turns identification_end+1..care_end) ----
        if turn <= pc.care_end:
            # Prioritize identified-but-not-managed students
            for s in students:
                if s.student_id in identified and not s.managed:
                    # Observable distress proxy: visible emotional outburst
                    # replaces the latent ``distress_level >= 0.6`` check.
                    observable = _obs_lookup.get(s.student_id)
                    if (
                        observable is not None
                        and "emotional_outburst" in observable.visible_behaviors
                    ):
                        return TeacherAction(
                            action_type="private_correction",
                            student_id=s.student_id,
                            reasoning="Phase 4: visible emotional outburst, private correction",
                        )
                    strategy = self._choose_strategy(s.student_id, observable)
                    return TeacherAction(
                        action_type="individual_intervention",
                        student_id=s.student_id,
                        strategy=strategy,
                        reasoning=f"Phase 4: intervention with {strategy}",
                    )

            # Still have unidentified suspicious students
            candidate = self._most_suspicious_student(identified)
            if candidate:
                profile = self.memory.get_profile(candidate.student_id)
                score = profile.adhd_indicator_score()
                n_obs = sum(profile.behavior_frequency_counts.values())
                if score >= (0.75 * _id_threshold_modifier) and n_obs >= 10:
                    is_adhd, confidence, reasoning = self.memory.identify_adhd(
                        candidate.student_id
                    )
                    if confidence >= 0.75:
                        return TeacherAction(
                            action_type="identify_adhd",
                            student_id=candidate.student_id,
                            reasoning=reasoning,
                        )
                return TeacherAction(
                    action_type="observe",
                    student_id=candidate.student_id,
                    reasoning=f"Phase 4: monitoring suspicious (score={score:.2f})",
                )

            return TeacherAction(
                action_type="class_instruction",
                reasoning="Phase 4: general management",
            )

        # ---- Phase 5: Maintenance + Relapse (turns care_end+1..950) ----
        # Observable relapse: a managed student re-exhibits any of the
        # high-visibility disruptive behaviors the teacher can actually
        # see. Replaces the latent ``compliance < MANAGED_COMPLIANCE``
        # check. ``_DISRUPTIVE_VISIBLE_BEHAVIORS`` mirrors
        # ``ClassroomV2._visible_behaviors`` high-vis set.
        for s in students:
            if s.student_id in identified and s.managed:
                observable = _obs_lookup.get(s.student_id)
                if observable is not None and any(
                    b in _DISRUPTIVE_VISIBLE_BEHAVIORS
                    for b in observable.visible_behaviors
                ):
                    strategy = self._choose_strategy(s.student_id, observable)
                    return TeacherAction(
                        action_type="individual_intervention",
                        student_id=s.student_id,
                        strategy=strategy,
                        reasoning=f"Phase 5: relapse re-intervention ({strategy})",
                    )

        # Identified but not yet managed
        for s in students:
            if s.student_id in identified and not s.managed:
                observable = _obs_lookup.get(s.student_id)
                strategy = self._choose_strategy(s.student_id, observable)
                return TeacherAction(
                    action_type="individual_intervention",
                    student_id=s.student_id,
                    strategy=strategy,
                    reasoning=f"Phase 5: continued care ({strategy})",
                )

        # Final sweep for missed students
        candidate = self._most_suspicious_student(identified)
        if candidate:
            profile = self.memory.get_profile(candidate.student_id)
            score = profile.adhd_indicator_score()
            n_obs = sum(profile.behavior_frequency_counts.values())
            if score >= (0.70 * _id_threshold_modifier) and n_obs >= 10:
                is_adhd, confidence, reasoning = self.memory.identify_adhd(
                    candidate.student_id
                )
                if confidence >= (0.70 * _id_threshold_modifier):
                    return TeacherAction(
                        action_type="identify_adhd",
                        student_id=candidate.student_id,
                        reasoning=reasoning,
                    )

        return TeacherAction(
            action_type="class_instruction",
            reasoning="Phase 5: maintenance, general instruction",
        )

    def _decide_action_llm(
        self,
        obs: ClassroomObservation,
        turn: int,
        *,
        teacher_batch: TeacherObservationBatch | None = None,
    ) -> TeacherAction:
        """Use LLM (Codex CLI) for teacher decision with full memory context.

        Builds a Korean-language prompt including:
        - Current observation (all students' visible behaviors)
        - Teacher memory context (similar cases + principles from Experience Base)
        - Student profiles accumulated so far
        - Identified / ruled-out students
        - Phase guidance
        - Available actions

        Falls back to rule-based on any error.
        """
        try:
            # 1. Build student observation summary from the Phase 6
            # slice 1 partial-observation batch. The LLM prompt gets
            # the same observable-only view the rule-based path uses
            # (visible behaviors + profile hint + teacher-side
            # identified/managed flags). Memory scores are
            # derived from past observations, not latent state.
            if teacher_batch is None:
                teacher_batch = build_observations_from_classroom(obs)
            student_lines: list[str] = []
            for observation in teacher_batch:
                profile = self.memory.get_profile(observation.student_id)
                score = profile.adhd_indicator_score() if profile else 0.0
                student_lines.append(
                    f"  {observation.student_id}: "
                    f"behaviors={list(observation.visible_behaviors)}, "
                    f"hint={observation.profile_hint}, "
                    f"score={score:.2f}"
                )

            # 2. Retrieve memory context for top suspicious students
            suspicious = getattr(self, "_stream_suspicious", {})
            memory_context_lines: list[str] = []
            for sid in list(suspicious.keys())[:5]:
                profile = self.memory.get_profile(sid)
                if not profile:
                    continue
                dominant = profile.dominant_behaviors(top_k=5)
                similar = self.memory.retrieve_similar_cases(
                    dominant, top_k=3, exclude_student_id=sid,
                )
                for sim, rec in similar:
                    if sim < 0.05:
                        continue
                    label = (
                        "ADHD" if rec.was_adhd
                        else ("정상" if rec.was_adhd is False else "미확인")
                    )
                    memory_context_lines.append(
                        f"  {sid}와 유사한 과거 사례: {rec.student_id} "
                        f"(sim={sim:.2f}) -> {label}"
                    )

            # 3. Get principles from Experience Base
            principles = self.memory.experience_base.top_principles(top_k=5)
            principle_lines = (
                [f"  - {p.text}" for p in principles]
                if principles
                else ["  (아직 없음)"]
            )

            # 4. Determine current phase
            pc = self.phase_config
            if turn <= pc.observation_end:
                phase = (
                    f"Phase 1 (관찰, turn {turn}/{pc.observation_end}): "
                    "모든 학생을 순환 관찰하세요."
                )
            elif turn <= pc.screening_end:
                phase = (
                    f"Phase 2 (스크리닝, turn {turn}/{pc.screening_end}): "
                    "의심 학생을 집중 관찰하고 가설 테스트하세요."
                )
            elif turn <= pc.identification_end:
                phase = (
                    f"Phase 3 (판별, turn {turn}/{pc.identification_end}): "
                    "충분한 근거가 있으면 판별하세요."
                )
            elif turn <= pc.care_end:
                phase = (
                    f"Phase 4 (케어, turn {turn}/{pc.care_end}): "
                    "판별된 학생에게 개입하세요."
                )
            else:
                phase = (
                    f"Phase 5 (유지, turn {turn}/950): "
                    "관리 완료 학생의 재발을 모니터하세요."
                )

            # 5. Gather identified and ruled-out sets
            identified = getattr(self, "_stream_identified", set())
            ruled_out = getattr(self, "_stream_ruled_out", set())

            # 6. Build full prompt
            prompt = (
                "당신은 한국 초등학교 담임교사입니다. 20명의 학생을 관찰하며 "
                "ADHD가 의심되는 학생을 판별하고 케어합니다.\n\n"
                f"## 현재 상황\n"
                f"턴: {turn}/950 (Day {obs.day})\n"
                f"{phase}\n\n"
                f"## 학생 관찰\n"
                + "\n".join(student_lines) + "\n\n"
                f"## 이미 ADHD로 판별한 학생\n"
                f"{list(identified) if identified else '없음'}\n\n"
                f"## 제외한 학생 (ADHD 아닌 것으로 판단)\n"
                f"{list(ruled_out) if ruled_out else '없음'}\n\n"
                f"## 과거 유사 사례 (Case Base)\n"
                + ("\n".join(memory_context_lines) if memory_context_lines
                   else "  (유사 사례 없음)") + "\n\n"
                f"## 학습된 원칙 (Experience Base)\n"
                + "\n".join(principle_lines) + "\n\n"
                "## 사용 가능한 행동\n"
                "1. observe(student_id) - 특정 학생 집중 관찰\n"
                "2. class_instruction() - 전체 학급 지도\n"
                "3. individual_intervention(student_id, strategy) - 개별 개입\n"
                "   전략: transition_warning, offer_choice, labeled_praise, "
                "visual_schedule_cue,\n"
                "   break_offer, empathic_acknowledgment, redirect_attention, "
                "countdown_timer,\n"
                "   collaborative_problem_solving, ignore_wait, firm_boundary, "
                "sensory_support\n"
                "4. private_correction(student_id) - 교무실 1:1 상담\n"
                "5. public_correction(student_id) - 교실 내 공개 지적\n"
                "6. identify_adhd(student_id, reasoning) - ADHD 판별 (근거 필수)\n"
                "7. generate_report(student_id) - 판별 리포트 생성\n\n"
                "하나의 행동을 선택하세요. 과거 사례와 원칙을 참고하여 판단하세요.\n"
                '반드시 JSON으로만 응답:\n'
                '{"action_type": "...", "student_id": "...", '
                '"strategy": "...", "reasoning": "..."}'
            )

            # 7. Call LLM via generate_raw (no state schema enforcement)
            response = self.teacher_llm.backend.generate_raw(prompt)
            return self._parse_llm_response(response)
        except Exception:
            # Fallback to rule-based on any error
            return self._decide_action_rule_based(
                obs, turn,
                getattr(self, "_stream_identified", set()),
                getattr(self, "_stream_suspicious", {}),
            )

    def _parse_llm_response(self, raw: str) -> TeacherAction:
        """Parse Codex CLI JSON response into a TeacherAction."""
        import json as _json
        import re as _re

        text = raw.strip()
        # Extract JSON from markdown code fences
        fenced = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, _re.DOTALL)
        if fenced:
            text = fenced.group(1)
        else:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]

        try:
            data = _json.loads(text)
            action_type = str(data.get("action_type", "class_instruction")).strip()
            student_id = data.get("student_id") or None
            strategy = data.get("strategy") or None
            reasoning = str(data.get("reasoning", "")).strip()

            valid_actions = {
                "observe", "class_instruction", "individual_intervention",
                "private_correction", "public_correction",
                "identify_adhd", "generate_report",
            }
            if action_type not in valid_actions:
                action_type = "class_instruction"

            # Actions that need a student_id
            if action_type in {
                "observe", "individual_intervention", "private_correction",
                "public_correction", "identify_adhd", "generate_report",
            } and not student_id:
                action_type = "class_instruction"
                student_id = None
                strategy = None

            # individual_intervention needs a valid strategy
            valid_strategies = {
                "transition_warning", "offer_choice", "labeled_praise",
                "visual_schedule_cue", "break_offer", "empathic_acknowledgment",
                "redirect_attention", "countdown_timer",
                "collaborative_problem_solving", "ignore_wait",
                "firm_boundary", "sensory_support",
            }
            if action_type == "individual_intervention":
                if strategy not in valid_strategies:
                    strategy = "redirect_attention"

            return TeacherAction(
                action_type=action_type,
                student_id=student_id,
                strategy=strategy,
                reasoning=reasoning,
            )
        except (_json.JSONDecodeError, KeyError):
            return TeacherAction(
                action_type="class_instruction",
                reasoning="LLM parse error, fallback",
            )

    # ------------------------------------------------------------------
    # Helper: cached labeled case retrieval (avoids repeated O(N) scans)
    # ------------------------------------------------------------------

    def _get_labeled_cases(
        self, behaviors: list[str], exclude_student_id: str,
    ) -> list[tuple[float, Any]]:
        """Retrieve similar cases with known ADHD labels, with per-turn caching."""
        # Cache key: turn + behaviors + excluded student
        cache_key = (self.memory._turn, tuple(sorted(behaviors)), exclude_student_id)
        if not hasattr(self, "_labeled_cache"):
            self._labeled_cache: dict = {}
        if cache_key in self._labeled_cache:
            return self._labeled_cache[cache_key]

        similar = self.memory.retrieve_similar_cases(
            behaviors, top_k=5, exclude_student_id=exclude_student_id,
        )
        labeled = [
            (sim, rec) for sim, rec in similar if rec.was_adhd is not None
        ]
        self._labeled_cache[cache_key] = labeled
        # Limit cache size to prevent memory bloat
        if len(self._labeled_cache) > 200:
            self._labeled_cache.clear()
        return labeled

    # ------------------------------------------------------------------
    # Helper: most suspicious student
    # ------------------------------------------------------------------

    def _most_suspicious_student(
        self, identified: set[str],
    ) -> Any:
        """Return the unidentified student with highest suspicion score.

        Uses precomputed memory-blended scores from _stream_suspicious when
        available (populated in Phase 2a). Falls back to behavioral score only.
        """
        best_score = -1.0
        best_student = None
        ruled_out = getattr(self, "_stream_ruled_out", set())
        suspicious = getattr(self, "_stream_suspicious", {})

        for s in self.classroom.students:
            if s.student_id in identified or s.student_id in ruled_out:
                continue
            # Use precomputed memory-blended score if available
            if s.student_id in suspicious:
                score = suspicious[s.student_id]
            else:
                profile = self.memory.get_profile(s.student_id)
                score = profile.adhd_indicator_score()

            if score > best_score:
                best_score = score
                best_student = s
        return best_student

    # ------------------------------------------------------------------
    # Helper: choose intervention strategy
    # ------------------------------------------------------------------

    def _choose_strategy(
        self,
        student_id: str,
        observable: Any = None,
    ) -> str:
        """Pick best intervention strategy from teacher-visible signals.

        Phase 6 slice 1: replaces the previous
        ``_choose_strategy(student)`` signature that read latent
        student state directly. The decision path now passes either
        a ``StudentObservation`` or ``None`` and this routine only
        consults:

          * teacher memory (profile history, case base, experience base)
          * teacher emotional state
          * the supplied observation's ``visible_behaviors`` /
            ``profile_hint`` — never the student's latent state

        ``observable`` is typed as ``Any`` to avoid a forward import
        cycle; at runtime it is a ``StudentObservation`` instance or
        ``None`` when the caller has no turn-level observation yet.
        """
        profile = self.memory.get_profile(student_id)
        history = profile.response_to_interventions

        # Teacher emotional state biases strategy choice
        # Burned out: prefer firm_boundary, avoid empathic strategies
        if self.teacher_emotions.is_burned_out():
            return "firm_boundary"

        # --- Fix 3: Check what worked for similar students in case base ---
        # Only query if no local intervention history yet, and only if there
        # are labeled records (from prior class outcomes) to learn from.
        if not history and len(self.memory.case_base) > 0:
            dominant = profile.dominant_behaviors(top_k=5)
            # Check if any labeled records exist before doing expensive retrieval
            has_labeled = any(
                rec.was_adhd is not None
                for rec in self.memory.case_base._records[-500:]  # sample recent
            )
            if dominant and has_labeled:
                similar = self.memory.retrieve_similar_cases(
                    dominant, top_k=5, exclude_student_id=student_id,
                )
                strategy_scores: dict[str, float] = {}
                for sim, record in similar:
                    if record.action_taken and record.outcome == "positive":
                        if record.was_adhd is True:
                            strategy_scores[record.action_taken] = (
                                strategy_scores.get(record.action_taken, 0.0) + sim * 1.5
                            )
                        elif record.was_adhd is None:
                            strategy_scores[record.action_taken] = (
                                strategy_scores.get(record.action_taken, 0.0) + sim
                            )
                        # skip was_adhd=False records
                if strategy_scores:
                    best_from_memory = max(strategy_scores, key=strategy_scores.get)  # type: ignore[arg-type]
                    if best_from_memory in STRATEGIES:
                        return best_from_memory

        # Fall back to this student's own intervention history
        if history:
            best = max(history, key=lambda k: history[k])
            return best

        # Fall back to observation-derived heuristics
        return self._default_strategy(observable)

    def _default_strategy(self, observable: Any = None) -> str:
        """Observable-only fallback strategy selection.

        Consumes a ``StudentObservation`` (or None). Derives proxies
        from visible behaviors and the coarse ``profile_hint`` label,
        with no access to latent scalars. The previous version read
        student state directly; this replacement uses only the
        partial-observation layer.

        Decision ladder (behavior-only, no ``profile_hint`` reads
        since the teacher observation builder now emits only
        ``identified_adhd`` / ``disruptive`` / ``unknown`` and
        those are not informative for strategy selection):
          1. visible ``emotional_outburst`` with a low-empathy
             teacher → ``firm_boundary`` (legacy misread branch,
             reproduced with observables)
          2. visible ``emotional_outburst`` →
             ``empathic_acknowledgment``
          3. any disruptive visible behavior → ``break_offer``
          4. any inattentive visible behavior →
             ``redirect_attention`` (currently unreachable under
             the standard ClassroomV2 visibility filter; kept
             future-proof)
          5. nothing observable → random strategy
        """
        visible: tuple[str, ...] = tuple()
        if observable is not None:
            visible = tuple(getattr(observable, "visible_behaviors", ()) or ())

        has_outburst = "emotional_outburst" in visible
        has_disruption = any(b in _DISRUPTIVE_VISIBLE_BEHAVIORS for b in visible)
        has_inattention = any(
            b in _INATTENTIVE_VISIBLE_BEHAVIORS for b in visible
        )

        # Low empathy: misreads visible outburst as defiance,
        # escalates to firm_boundary.
        if (
            has_outburst
            and self.teacher_emotions.observation_accuracy() < 0.5
        ):
            return "firm_boundary"

        if has_outburst:
            return "empathic_acknowledgment"
        if has_disruption:
            return "break_offer"
        if has_inattention:
            return "redirect_attention"

        return self._rng.choice(STRATEGIES)

    # ------------------------------------------------------------------
    # Memory update
    # ------------------------------------------------------------------

    def _drain_feedback_queue(self, current_turn: int) -> int:
        """Commit every pending memory observation whose delay has matured.

        Phase 6 slice 3. Called once at the start of each turn, and
        once more at class end (via ``_flush_feedback_queue``) so
        tail-end observations still land in memory. Returns the
        number of items committed for observability in tests.

        For each due pending item, re-derive an ``ObservationOutcome``
        through the observable-only feedback path and push it onto
        the case base via ``memory.append_record``. This is the only
        place where queued pending items convert into real records.
        """
        due = self._feedback_queue.pop_due(current_turn)
        for pending in due:
            feedback = self._derive_feedback_outcome(
                student_id=pending.student_id,
                teacher_action=pending.teacher_action,
            )
            self.memory.append_record(
                student_id=pending.student_id,
                turn=pending.observed_turn,
                observed_behaviors=pending.observed_behaviors,
                feedback=feedback,
            )
        return len(due)

    def _flush_feedback_queue(self) -> int:
        """Commit every pending memory observation, regardless of due_turn.

        Called once at class end so observations staged on the final
        turns still reach memory. Uses the same observable-only
        feedback path as ``_drain_feedback_queue``.
        """
        remaining = self._feedback_queue.flush_all()
        for pending in remaining:
            feedback = self._derive_feedback_outcome(
                student_id=pending.student_id,
                teacher_action=pending.teacher_action,
            )
            self.memory.append_record(
                student_id=pending.student_id,
                turn=pending.observed_turn,
                observed_behaviors=pending.observed_behaviors,
                feedback=feedback,
            )
        return len(remaining)

    def _derive_feedback_outcome(
        self,
        student_id: str,
        teacher_action: str,
    ) -> ObservationOutcome:
        """Phase 6 slice 2 feedback-translation step.

        Builds an explicit ``ObservationOutcome`` payload for one
        memory commit, pulling the raw latent compliance ONCE here
        to produce a coarse teacher-visible outcome label. The
        scalar itself is discarded immediately; only the label,
        the teacher_action, and any post-action visible behaviors
        are stored in the case base. This is the single point
        where compliance is read for memory feedback — callers
        must not bypass it.

        The exact coarse thresholds (``>= 0.75`` → positive,
        ``<= 0.35`` → negative, otherwise neutral) preserve the
        legacy label distribution the calibration stack was tuned
        against; a future delayed-feedback pass can replace this
        immediate read with a queued look-back without changing
        the ObservationOutcome schema.
        """
        student_obj = self.classroom.get_student(student_id)
        if student_obj is None:
            return ObservationOutcome(
                outcome="neutral",
                teacher_action=teacher_action,
            )
        compliance = float(student_obj.state.get("compliance", 0.5))
        if compliance >= 0.75:
            label = "positive"
        elif compliance <= 0.35:
            label = "negative"
        else:
            label = "neutral"

        # Post-action visible behaviors (teacher-visible only; the
        # same visibility filter the observation builder uses).
        post_behaviors: tuple[str, ...] = ()
        visible = getattr(student_obj, "exhibited_behaviors", None) or []
        post_behaviors = tuple(
            b for b in visible if b in _DISRUPTIVE_VISIBLE_BEHAVIORS
        )

        return ObservationOutcome(
            outcome=label,
            teacher_action=teacher_action,
            post_behaviors=post_behaviors,
        )

    def _update_memory(
        self,
        obs: ClassroomObservation,
        action: TeacherAction,
        info: dict,
        tracks: dict[str, _StudentTrack],
        turn: int,
    ) -> None:
        """Record observations and outcomes in TeacherMemory and local tracks.

        Detailed observations are processed and committed first so that
        subsequent summary processing cannot overwrite them.
        Also records compliance deltas for hypothesis testing interventions.
        """
        committed_this_turn: set[str] = set()

        # Record hypothesis test results: measure compliance delta from intervention
        if (
            action.student_id
            and action.strategy in _HYPOTHESIS_TEST_STRATEGIES
            and action.student_id in self._hypothesis_trackers
        ):
            student_obj = self.classroom.get_student(action.student_id)
            track = tracks.get(action.student_id)
            if student_obj and track and track.compliance_history:
                prev_compliance = track.compliance_history[-1]
                curr_compliance = student_obj.state.get("compliance", prev_compliance)
                delta = curr_compliance - prev_compliance
                self._hypothesis_trackers[action.student_id].record_test(
                    action.strategy, delta
                )

        # 1. Detailed observations first -- observe now, ENQUEUE
        # the commit for the delayed-feedback queue.
        #
        # Phase 6 slice 2: the latent-state dump from the detailed
        # observation block is deliberately NOT passed into teacher
        # memory.
        #
        # Phase 6 slice 3: the ObservationOutcome is no longer
        # derived and committed on the same turn. Instead the
        # observation is enqueued on `self._feedback_queue` with a
        # `due_turn = turn + self.feedback_delay_turns`, and the
        # turn-start drain step calls `_derive_feedback_outcome`
        # + `memory.append_record` when the item matures.
        for detail in obs.detailed_observations:
            sid = detail.student_id
            behaviors = detail.behaviors

            track = tracks.get(sid)
            if track:
                track.all_behaviors.extend(behaviors)
                track.observation_count += 1
                for b in behaviors:
                    track.turns_per_behavior.setdefault(b, []).append(turn)
                if action.student_id == sid and action.strategy:
                    track.strategies_applied.append(action.strategy)

            translated = _translate_behaviors(behaviors)
            self.memory.observe(
                student_id=sid,
                behaviors=translated,
                # state explicitly dropped — Phase 6 slice 2 boundary
                action_taken=action.action_type,
            )
            self._feedback_queue.enqueue(
                PendingObservationFeedback(
                    student_id=sid,
                    observed_behaviors=list(translated),
                    teacher_action=action.action_type,
                    observed_turn=turn,
                    due_turn=turn + self.feedback_delay_turns,
                )
            )
            committed_this_turn.add(sid)

        # 2. Student summaries -- skip students already handled this turn
        for summary in obs.student_summaries:
            sid = summary.student_id
            if sid in committed_this_turn:
                continue

            behaviors = summary.behaviors

            track = tracks.get(sid)
            if track:
                track.all_behaviors.extend(behaviors)
                for b in behaviors:
                    track.turns_per_behavior.setdefault(b, []).append(turn)

            # Only feed to memory if this student has visible disruptive behaviors
            if behaviors and any(b not in ("on_task", "listening", "writing") for b in behaviors):
                translated = _translate_behaviors(behaviors)
                self.memory.observe(
                    student_id=sid,
                    behaviors=translated,
                    # state explicitly dropped — Phase 6 slice 2 boundary
                    action_taken="passive_observation",
                )
                self._feedback_queue.enqueue(
                    PendingObservationFeedback(
                        student_id=sid,
                        observed_behaviors=list(translated),
                        teacher_action="passive_observation",
                        observed_turn=turn,
                        due_turn=turn + self.feedback_delay_turns,
                    )
                )
                committed_this_turn.add(sid)

    # ------------------------------------------------------------------
    # Log teacher action
    # ------------------------------------------------------------------

    def _log_teacher_action(
        self,
        action: TeacherAction,
        obs: ClassroomObservation,
        turn: int,
    ) -> None:
        """Record the teacher action as an InteractionEvent in the log."""
        event_type_map = {
            "observe": "teacher_observe",
            "class_instruction": "teacher_class_instruction",
            "individual_intervention": "teacher_intervene",
            "private_correction": "teacher_correct_private",
            "public_correction": "teacher_correct_public",
            "identify_adhd": "teacher_identify",
        }

        event = InteractionEvent(
            class_id=self.classroom.class_id,
            turn=turn,
            actor="teacher",
            target=action.student_id or "class",
            participants=["teacher"] + ([action.student_id] if action.student_id else []),
            event_type=event_type_map.get(action.action_type, "teacher_observe"),
            action=action.action_type,
            content=action.reasoning,
            location=obs.location,
        )
        self.log.record(event)

    # ------------------------------------------------------------------
    # Report builder
    # ------------------------------------------------------------------

    def _build_report(
        self,
        student_id: str,
        turn: int,
        tracks: dict[str, _StudentTrack],
        action: TeacherAction,
    ) -> Optional[IdentificationReport]:
        """Build and evaluate an IdentificationReport for a newly identified student."""
        student = self.classroom.get_student(student_id)
        if student is None:
            return None

        track = tracks.get(student_id)
        behaviors = track.all_behaviors if track else []
        turns_per = track.turns_per_behavior if track else {}

        inattention_symptoms, hyperactivity_symptoms = _behaviors_to_dsm5(
            behaviors, turns_per, turn
        )

        # Determine subtype from symptom counts
        n_inatt = len(inattention_symptoms)
        n_hyper = len(hyperactivity_symptoms)
        if n_inatt >= n_hyper and n_hyper < 3:
            subtype = "inattentive"
        elif n_hyper >= n_inatt and n_inatt < 3:
            subtype = "hyperactive-impulsive"
        else:
            subtype = "combined"

        _, confidence, reasoning = self.memory.identify_adhd(student_id)

        report = IdentificationReport(
            student_id=student_id,
            teacher_class_id=self.class_count + 1,
            turn_identified=turn,
            observed_inattention_symptoms=inattention_symptoms,
            observed_hyperactivity_symptoms=hyperactivity_symptoms,
            identified_subtype=subtype,
            confidence=round(confidence, 3),
            reasoning=action.reasoning or reasoning,
        )

        # Ground-truth subtype mapping
        gt_subtype_map = {
            "adhd_inattentive": "inattentive",
            "adhd_hyperactive_impulsive": "hyperactive-impulsive",
            "adhd_combined": "combined",
        }
        gt_subtype = gt_subtype_map.get(student.profile_type, None)

        # Delayed feedback: only feedback_rate fraction of identifications
        # get confirmed with ground truth. The rest remain unconfirmed.
        if self._rng.random() < self.feedback_rate:
            # Confirmed: teacher learns from outcome
            report.evaluate(
                ground_truth_adhd=student.is_adhd,
                ground_truth_subtype=gt_subtype,
            )
            self.evaluator.add_report(report)
            self.memory.record_outcome(student_id, was_correct=bool(report.is_correct))
        else:
            # Unconfirmed: teacher doesn't know if they were right.
            # No outcome recorded -> Experience Base doesn't update for this case.
            # Case Base still has the observation.
            report.ground_truth_is_adhd = None
            report.is_correct = None

        if track:
            track.identification_turn = turn

        return report

    # ------------------------------------------------------------------
    # Event builder
    # ------------------------------------------------------------------

    def _make_event(
        self,
        turn: int,
        obs: ClassroomObservation,
        action: TeacherAction,
        info: dict,
        reward: float,
        identified_students: set[str],
    ) -> dict:
        """Create event dict for WebSocket streaming."""
        location = info.get("location", "classroom")
        if action.action_type == "private_correction":
            location = "office"
        return {
            "type": "turn",
            "class_id": self.class_count + 1,
            "turn": turn,
            "day": info.get("day", 1),
            "period": info.get("period", 1),
            "subject": info.get("subject", ""),
            "location": location,
            "students": [
                {
                    "id": s.student_id,
                    "state": dict(s.state),
                    "behaviors": list(s.exhibited_behaviors),
                    "is_identified": s.student_id in identified_students,
                    "is_managed": s.managed,
                }
                for s in self.classroom.students
            ],
            "teacher_action": {
                "action_type": action.action_type,
                "student_id": action.student_id,
                "strategy": action.strategy,
                "reasoning": action.reasoning,
            },
            "interactions": [
                {
                    "actor": ev.actor,
                    "target": ev.target,
                    "event_type": ev.event_type,
                    "content": ev.content[:80] if ev.content else "",
                }
                for ev in info.get("interactions", [])
                if ev.event_type.startswith("peer_")
            ][:5],
            "reward": round(float(reward), 4),
            "memory_summary": self._compact_memory_summary(),
        }

    def _compact_memory_summary(self) -> str:
        """Build a compact summary string from growth_report() data."""
        gr = self.memory.growth_report()
        return (
            f"classes={gr.get('classes_seen', 0)}, "
            f"cases={gr.get('case_base_size', 0)}, "
            f"principles={gr.get('experience_base_size', 0)}, "
            f"precision={gr.get('precision', 0.0):.2f}, "
            f"recall={gr.get('recall', 0.0):.2f}"
        )

    # ------------------------------------------------------------------
    # Class result compiler
    # ------------------------------------------------------------------

    def _compile_class_result(
        self,
        obs: ClassroomObservation,
        turn: int,
        tracks: dict[str, _StudentTrack],
        reports: list[IdentificationReport],
        events: list[dict],
        strategies_used: set[str],
        identification_turns: list[float],
        identified_students: set[str],
    ) -> dict:
        """Compile ClassMetrics for GrowthTracker."""
        adhd_students = [s for s in self.classroom.students if s.is_adhd]
        normal_students = [s for s in self.classroom.students if not s.is_adhd]

        ground_truth = set(self.classroom.ground_truth_adhd_ids())
        tp = len(identified_students & ground_truth)
        fp = len(identified_students - ground_truth)
        fn = len(ground_truth - identified_students)
        tn = len(normal_students) - fp

        # Supply FN/TN counts to evaluator
        self.evaluator.add_missed(fn)
        self.evaluator.add_true_negative(max(0, tn))

        avg_id_turn = (
            sum(identification_turns) / len(identification_turns)
            if identification_turns else 0.0
        )

        # Behavior improvement: (final_compliance - initial_compliance) per ADHD student
        improvement_rates: list[float] = []
        for s in adhd_students:
            track = tracks.get(s.student_id)
            if track and track.compliance_history:
                rate = track.compliance_history[-1] - track.initial_compliance
                improvement_rates.append(max(0.0, rate))

        # Average care turns per managed ADHD student
        managed_adhd = [s for s in adhd_students if s.managed]
        avg_care_turns = (
            sum(
                len(tracks[s.student_id].strategies_applied)
                for s in managed_adhd
                if s.student_id in tracks
            ) / len(managed_adhd)
            if managed_adhd else 0.0
        )

        n_managed = sum(1 for s in self.classroom.students if s.is_adhd and s.managed)

        # Per-category breakdown for Macro-F1
        # ADHD TP/FP/FN counted from identified vs ground truth sets
        adhd_tp = tp
        adhd_fp = fp
        adhd_fn = fn
        # Confounder FP: normal students with confounding profiles wrongly identified
        # Actual profile_type values: anxiety, odd, gifted, sleep_deprived
        _CONFOUNDER_PROFILES = {"anxiety", "odd", "gifted", "sleep_deprived"}
        confounder_fp = 0
        for sid in (identified_students - ground_truth):
            s_obj = self.classroom.get_student(sid)
            if s_obj and hasattr(s_obj, "profile_type"):
                if s_obj.profile_type in _CONFOUNDER_PROFILES:
                    confounder_fp += 1

        metrics = ClassMetrics(
            class_id=self.class_count + 1,
            n_students=len(self.classroom.students),
            n_adhd=len(adhd_students),
            n_identified=len(identified_students),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=max(0, tn),
            avg_identification_turn=round(avg_id_turn, 2),
            avg_care_turns=round(avg_care_turns, 2),
            strategies_used=list(strategies_used),
            behavior_improvement_rates=improvement_rates,
            n_managed=n_managed,
            class_completion_turn=turn,
            adhd_tp=adhd_tp,
            adhd_fp=adhd_fp,
            adhd_fn=adhd_fn,
            confounder_fp=confounder_fp,
        )

        return {
            "metrics": metrics,
            "events": events,
            "reports": reports,
            # Calibration-oriented exports (§25.13 bridge data):
            "teacher_patience_log": list(self._stream_patience_log),
            "intervention_outcomes": list(self._stream_intervention_outcomes),
            "first_suspicion_turns": dict(self._stream_first_suspicion_turns),
        }
