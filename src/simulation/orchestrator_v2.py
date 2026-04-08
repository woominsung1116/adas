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
# Teacher Emotional State (Phase 2 enhancement)
# ---------------------------------------------------------------------------


@dataclass
class TeacherEmotionalState:
    """Teacher emotional state that affects observation and decision quality.

    Patience depletes with difficult students/incidents, partially recovers
    between days. Frustration accumulates and reduces observation accuracy.
    Burnout (low patience) degrades decision quality.
    """

    patience: float = 0.80
    empathy_capacity: float = 0.70
    frustration: float = 0.10
    bias: dict = field(default_factory=dict)

    def update_after_turn(self, classroom_mood: str, n_incidents: int) -> None:
        """Update emotional state based on turn events."""
        self.patience = max(0.1, self.patience - 0.01 * n_incidents)
        self.frustration = min(0.9, self.frustration + 0.005 * n_incidents)

        # Chaotic/tense mood drains patience faster
        if classroom_mood in ("chaotic", "tense"):
            self.patience = max(0.1, self.patience - 0.005)

    def daily_recovery(self) -> None:
        """Partial recovery at the start of each day."""
        self.patience = min(0.9, self.patience + 0.03)
        self.frustration = max(0.0, self.frustration - 0.01)

    def observation_accuracy(self) -> float:
        """How accurately the teacher reads student emotional state."""
        return min(1.0, self.empathy_capacity * (1.0 - self.frustration * 0.3))

    def is_burned_out(self) -> bool:
        """Teacher is burned out when patience drops below 0.3."""
        return self.patience < 0.3


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

    Phase 2 sub-phases:
      2a: Flag as suspicious (adhd_indicator_score >= 0.5, observations >= 5)
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
        # Randomly pick a classroom archetype for this class
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

        for turn in range(1, self.classroom.MAX_TURNS + 1):
            self.memory.advance_turn()
            self._stream_final_turn = turn

            # Teacher daily recovery (at start of each day, period 1)
            if turn % 5 == 1:
                self.teacher_emotions.daily_recovery()

            # 1. Teacher decides action
            action = self._decide_action(
                self._stream_obs, turn,
                self._stream_identified, self._stream_suspicious,
            )

            # 2. Execute in environment
            self._stream_obs, reward, done, info = self.classroom.step(action)

            # 3. Update teacher memory and tracks
            self._update_memory(self._stream_obs, action, info, self._stream_tracks, turn)

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

            # 8. Build and yield turn event
            event = self._make_event(
                turn, self._stream_obs, action, info, reward, self._stream_identified
            )
            self._stream_events.append(event)
            yield event

            if done:
                break

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
    ) -> TeacherAction:
        """Dispatch to LLM or rule-based teacher."""
        if self.teacher_llm:
            return self._decide_action_llm(obs, turn)
        return self._decide_action_rule_based(obs, turn, identified, suspicious)

    def _decide_action_rule_based(
        self,
        obs: ClassroomObservation,
        turn: int,
        identified: set[str],
        suspicious: dict[str, float],
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
                    score = profile.adhd_indicator_score()
                    n_obs = sum(profile.behavior_frequency_counts.values())
                    if score >= 0.5 and n_obs >= 5:
                        suspicious[s.student_id] = score
                        # Create hypothesis tracker if not exists
                        if s.student_id not in self._hypothesis_trackers:
                            self._hypothesis_trackers[s.student_id] = HypothesisTracker(
                                student_id=s.student_id,
                                suspicion_turn=turn,
                            )
                    elif score > 0.3:
                        suspicious[s.student_id] = score

            # Phase 2b: Hypothesis testing for tracked suspicious students
            for sid, tracker in self._hypothesis_trackers.items():
                if sid in identified or sid in self._stream_ruled_out:
                    continue
                student_obj = self.classroom.get_student(sid)
                if student_obj is None:
                    continue

                # Determine which test strategy hasn't been tried yet
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

            # Focus on most suspicious unidentified student (still in 2a observation)
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

            # Nothing suspicious -- rotate observation
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

            # Try to identify high-confidence students
            candidate = self._most_suspicious_student(identified)
            if candidate:
                profile = self.memory.get_profile(candidate.student_id)
                score = profile.adhd_indicator_score()
                n_obs = sum(profile.behavior_frequency_counts.values())

                if score >= (0.85 * _id_threshold_modifier) and n_obs >= 10:
                    # Phase 2c: differential diagnosis gate
                    tracker = self._hypothesis_trackers.get(candidate.student_id)
                    if tracker and tracker.can_identify():
                        likely = tracker.likely_profile()
                        # Only identify if hypothesis points to ADHD
                        if likely.startswith("adhd") or likely == "unknown":
                            is_adhd, confidence, reasoning = self.memory.identify_adhd(
                                candidate.student_id
                            )
                            if confidence >= 0.85:
                                hypo_info = f" [hypothesis={likely}]"
                                return TeacherAction(
                                    action_type="identify_adhd",
                                    student_id=candidate.student_id,
                                    reasoning=reasoning + hypo_info,
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
                    # High distress -> private correction
                    if s.state.get("distress_level", 0.0) >= 0.6:
                        return TeacherAction(
                            action_type="private_correction",
                            student_id=s.student_id,
                            reasoning="Phase 4: high distress, private correction",
                        )
                    strategy = self._choose_strategy(s)
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
        # Check managed students for relapse
        for s in students:
            if s.student_id in identified and s.managed:
                # Relapse detection: compliance dropped below threshold
                if s.state.get("compliance", 0.8) < MANAGED_COMPLIANCE:
                    strategy = self._choose_strategy(s)
                    return TeacherAction(
                        action_type="individual_intervention",
                        student_id=s.student_id,
                        strategy=strategy,
                        reasoning=f"Phase 5: relapse re-intervention ({strategy})",
                    )

        # Identified but not yet managed
        for s in students:
            if s.student_id in identified and not s.managed:
                strategy = self._choose_strategy(s)
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
        self, obs: ClassroomObservation, turn: int
    ) -> TeacherAction:
        """Use LLM backend for teacher decision. Falls back to rule-based on error."""
        try:
            action = self.teacher_llm.decide_action(obs, turn)
            if isinstance(action, TeacherAction):
                return action
        except Exception:
            pass
        return self._decide_action_rule_based(obs, turn, set(), {})

    # ------------------------------------------------------------------
    # Helper: most suspicious student
    # ------------------------------------------------------------------

    def _most_suspicious_student(
        self, identified: set[str],
    ) -> Any:
        """Return the unidentified student with highest ADHD indicator score."""
        best_score = -1.0
        best_student = None
        ruled_out = getattr(self, "_stream_ruled_out", set())

        for s in self.classroom.students:
            if s.student_id in identified or s.student_id in ruled_out:
                continue
            profile = self.memory.get_profile(s.student_id)
            score = profile.adhd_indicator_score()
            if score > best_score:
                best_score = score
                best_student = s
        return best_student

    # ------------------------------------------------------------------
    # Helper: choose intervention strategy
    # ------------------------------------------------------------------

    def _choose_strategy(self, student: Any) -> str:
        """Pick best intervention strategy based on memory, state, and teacher emotions."""
        profile = self.memory.get_profile(student.student_id)
        history = profile.response_to_interventions

        # Teacher emotional state biases strategy choice
        # Burned out: prefer firm_boundary, avoid empathic strategies
        if self.teacher_emotions.is_burned_out():
            return "firm_boundary"

        if history:
            best = max(history, key=lambda k: history[k])
            return best

        distress = student.state.get("distress_level", 0.5)
        escalation = student.state.get("escalation_risk", 0.3)
        attention = student.state.get("attention", 0.5)

        # Low empathy: misreads anxiety as defiance, uses firm_boundary
        if (
            distress >= 0.6
            and self.teacher_emotions.observation_accuracy() < 0.5
        ):
            return "firm_boundary"

        if distress >= 0.6:
            return "empathic_acknowledgment"
        if escalation >= 0.6:
            return "break_offer"
        if attention < 0.3:
            return "redirect_attention"

        return self._rng.choice(STRATEGIES)

    # ------------------------------------------------------------------
    # Memory update
    # ------------------------------------------------------------------

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

        # 1. Detailed observations first -- observe AND commit immediately
        for detail in obs.detailed_observations:
            sid = detail.student_id
            behaviors = detail.behaviors
            state = detail.state_snapshot or {}

            track = tracks.get(sid)
            if track:
                track.all_behaviors.extend(behaviors)
                track.observation_count += 1
                for b in behaviors:
                    track.turns_per_behavior.setdefault(b, []).append(turn)
                if action.student_id == sid and action.strategy:
                    track.strategies_applied.append(action.strategy)

            self.memory.observe(
                student_id=sid,
                behaviors=_translate_behaviors(behaviors),
                state=state,
                action_taken=action.action_type,
            )

            # Commit immediately so summaries cannot overwrite
            student_obj = self.classroom.get_student(sid)
            if student_obj:
                compliance = student_obj.state.get("compliance", 0.5)
                if compliance >= 0.75:
                    outcome = "positive"
                elif compliance <= 0.35:
                    outcome = "negative"
                else:
                    outcome = "neutral"
                self.memory.commit_observation(sid, outcome=outcome)
                committed_this_turn.add(sid)

        # 2. Student summaries -- skip students already committed this turn
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
                self.memory.observe(
                    student_id=sid,
                    behaviors=_translate_behaviors(behaviors),
                    state={},
                    action_taken="passive_observation",
                )
                # Commit summary observations too
                student_obj = self.classroom.get_student(sid)
                if student_obj:
                    compliance = student_obj.state.get("compliance", 0.5)
                    if compliance >= 0.75:
                        outcome = "positive"
                    elif compliance <= 0.35:
                        outcome = "negative"
                    else:
                        outcome = "neutral"
                    self.memory.commit_observation(sid, outcome=outcome)
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
        }
