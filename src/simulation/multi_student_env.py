"""
multi_student_env.py

Multi-student classroom environment for ADHD detection and intervention simulation.

Korean epidemiological data source: KCI ART001701933
  - ADHD prevalence: 6-11% (default 9%)
  - Male:Female ratio: 3.19:1 for ADHD
  - Subtypes: inattentive 5.0%, hyperactive-impulsive 2.3%, combined 2.3%

Korean behavioral frequency source: KCI ART002478306
  - Inattention most common (18 studies), then hyperactivity (11), impulsivity (9), aggression (8)

O'Leary 1970: private correction more effective than public correction.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.environment.child_profiles import ChildProfile, load_profiles
from src.environment.scenarios import Scenario, load_scenarios
from src.environment.transition_constraints import (
    DEFAULT_ACTION_CONSTRAINTS,
    constrain_transition,
)

# ---------------------------------------------------------------------------
# Constants
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

# Korean prevalence data (KCI ART001701933)
ADHD_SUBTYPES = ["inattentive", "hyperactive_impulsive", "combined"]

# Relative proportion weights derived from 5.0 / 2.3 / 2.3
_SUBTYPE_WEIGHTS = [5.0, 2.3, 2.3]

# Severity distribution within ADHD population
_SEVERITY_WEIGHTS = {"mild": 0.45, "moderate": 0.35, "severe": 0.20}

# Normal-student behavior pool (observable, mild noise)
_NORMAL_BEHAVIORS: list[str] = [
    "on_task",
    "listening",
    "writing",
    "whispering_briefly",
    "looking_around",
    "fidgeting_slightly",
]

# Mild ADHD-like behaviors that normal students can occasionally show (PPV=0.19 per AAFP 2019)
_NORMAL_NOISE_BEHAVIORS: list[str] = [
    "easily_distracted",
    "fidgeting",
]

# ADHD behavior pool keyed by subtype
_ADHD_BEHAVIORS: dict[str, list[str]] = {
    "inattentive": [
        "daydreaming",
        "losing_materials",
        "forgetting_instructions",
        "staring_out_window",
        "not_starting_task",
        "off_task",
    ],
    "hyperactive_impulsive": [
        "out_of_seat",
        "calling_out",
        "interrupting",
        "excessive_talking",
        "fidgeting",
        "running_in_classroom",
    ],
    "combined": [
        "daydreaming",
        "out_of_seat",
        "calling_out",
        "off_task",
        "forgetting_instructions",
        "interrupting",
        "fidgeting",
    ],
}

# Severity multipliers on deviation amplitude
_SEVERITY_AMPLITUDE = {"mild": 0.08, "moderate": 0.15, "severe": 0.25}

# Management success threshold: compliance must exceed this for K consecutive turns
# Raised from 0.75/3 to 0.80/5 — harder to sustain improvement (more realistic)
MANAGED_COMPLIANCE_THRESHOLD = 0.80
MANAGED_CONSECUTIVE_TURNS = 5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StudentState:
    student_id: str
    is_adhd: bool
    adhd_subtype: str | None  # inattentive / hyperactive_impulsive / combined
    severity: str | None  # mild / moderate / severe
    gender: str  # male / female
    age: int
    state: dict[str, float]  # distress / compliance / attention / escalation_risk
    exhibited_behaviors: list[str] = field(default_factory=list)
    intervention_history: list[str] = field(default_factory=list)
    managed: bool = False
    identified: bool = False  # teacher has formally identified this student
    _consecutive_compliant_turns: int = field(default=0, repr=False)

    def copy_state(self) -> dict[str, float]:
        return dict(self.state)


@dataclass
class TeacherAction:
    action_type: str  # observe / class_instruction / individual_intervention /
                      # private_correction / public_correction / identify_adhd /
                      # generate_report
    student_id: str | None = None  # None means whole-class
    strategy: str | None = None    # for individual_intervention
    reasoning: str = ""            # for identify_adhd


@dataclass
class StudentObservation:
    student_id: str
    behaviors: list[str]
    state_snapshot: dict[str, float] | None = None  # only after focused observe()


@dataclass
class ClassroomObservation:
    turn: int
    student_observations: list[StudentObservation]
    class_context: str  # e.g. current scenario name
    identified_adhd_ids: list[str]
    managed_ids: list[str]
    all_complete: bool


# ---------------------------------------------------------------------------
# Core environment
# ---------------------------------------------------------------------------


class MultiStudentClassroom:
    """
    Open-ended multi-student classroom environment.

    Episode flow:
      1. reset() generates N students (mixed ADHD/normal).
      2. step(action) updates all students and returns ClassroomObservation.
      3. When all ADHD students are identified AND managed, is_class_complete() is True.
      4. Call reset() again to start a new class (experience persists externally).
    """

    def __init__(
        self,
        n_students: int = 20,
        adhd_prevalence: float | tuple[float, float] | None = None,
        profiles: list[ChildProfile] | None = None,
        scenarios: list[Scenario] | None = None,
        profiles_path: str | None = None,
        scenarios_path: str | None = None,
        action_constraints: dict | None = None,
        seed: int | None = None,
    ):
        self.n_students = n_students
        # ADHD prevalence: randomized within Korean research range (6-11%)
        # Source: KCI ART001701933 (Korean elementary prevalence 6-11%)
        if adhd_prevalence is None:
            # Default: random within Korean range each class reset
            self._prevalence_range = (0.06, 0.11)
            self.adhd_prevalence = 0.09  # initial, re-sampled on reset()
        elif isinstance(adhd_prevalence, tuple):
            self._prevalence_range = adhd_prevalence
            self.adhd_prevalence = sum(adhd_prevalence) / 2
        else:
            self._prevalence_range = None
            self.adhd_prevalence = adhd_prevalence
        self.action_constraints = action_constraints or DEFAULT_ACTION_CONSTRAINTS

        # Load profiles / scenarios if paths provided and objects not given
        if profiles is not None:
            self.profiles = profiles
        elif profiles_path is not None:
            self.profiles = load_profiles(profiles_path)
        else:
            self.profiles = []

        if scenarios is not None:
            self.scenarios = scenarios
        elif scenarios_path is not None:
            self.scenarios = load_scenarios(scenarios_path)
        else:
            self.scenarios = []

        self._rng = random.Random(seed)

        # Runtime state — populated by reset()
        self.students: list[StudentState] = []
        self.current_scenario: Scenario | None = None
        self.turn: int = 0
        self.identified_adhd_ids: set[str] = set()
        self.managed_ids: set[str] = set()
        self.class_history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> ClassroomObservation:
        """Generate a new classroom and return initial observation."""
        self.turn = 0
        self.identified_adhd_ids = set()
        self.managed_ids = set()
        self.class_history = []

        if self.scenarios:
            self.current_scenario = self._rng.choice(self.scenarios)
        else:
            self.current_scenario = None

        self.students = self._generate_students()
        return self._make_observation(detail_student_id=None)

    def step(
        self, action: TeacherAction
    ) -> tuple[ClassroomObservation, float, bool, dict[str, Any]]:
        """
        Apply teacher action, advance all students one turn.

        Returns:
            obs: ClassroomObservation
            reward: float
            done: bool  (True when class is complete)
            info: dict with per-student states and narrative
        """
        self.turn += 1
        reward = 0.0
        info: dict[str, Any] = {"turn": self.turn, "action": action, "student_updates": {}}

        detail_student_id: str | None = None

        if action.action_type == "observe":
            detail_student_id = action.student_id
            reward += self._reward_observe(action.student_id)

        elif action.action_type == "class_instruction":
            reward += self._apply_class_instruction()

        elif action.action_type == "individual_intervention":
            if action.student_id and action.strategy:
                reward += self._apply_individual_intervention(
                    action.student_id, action.strategy, public=False
                )

        elif action.action_type == "private_correction":
            if action.student_id:
                reward += self._apply_private_correction(action.student_id)

        elif action.action_type == "public_correction":
            if action.student_id:
                reward += self._apply_public_correction(action.student_id)

        elif action.action_type == "identify_adhd":
            if action.student_id:
                reward += self._apply_identify(action.student_id, action.reasoning)

        elif action.action_type == "generate_report":
            if action.student_id:
                reward += self._apply_generate_report(action.student_id)

        # Advance all students (behavior update each turn regardless of action)
        for student in self.students:
            self._update_student(student, action)
            info["student_updates"][student.student_id] = dict(student.state)

        # Update managed flags
        self._update_managed_flags()

        obs = self._make_observation(detail_student_id=detail_student_id)
        done = self.is_class_complete()
        if done:
            reward += 10.0  # class-completion bonus

        self.class_history.append(
            {"turn": self.turn, "action": action, "reward": reward}
        )
        return obs, reward, done, info

    def is_class_complete(self) -> bool:
        """True when every ADHD student is both identified and managed."""
        adhd_students = [s for s in self.students if s.is_adhd]
        if not adhd_students:
            return True
        return all(s.identified and s.managed for s in adhd_students)

    def ground_truth_adhd_ids(self) -> list[str]:
        """Return the true list of ADHD student IDs (for evaluation only)."""
        return [s.student_id for s in self.students if s.is_adhd]

    def get_student(self, student_id: str) -> StudentState | None:
        for s in self.students:
            if s.student_id == student_id:
                return s
        return None

    # ------------------------------------------------------------------
    # Student generation
    # ------------------------------------------------------------------

    def _generate_students(self) -> list[StudentState]:
        students = []

        # Determine how many students have ADHD
        # If range set, resample prevalence each class for realistic variation
        if self._prevalence_range is not None:
            lo, hi = self._prevalence_range
            self.adhd_prevalence = self._rng.uniform(lo, hi)
        n_adhd = max(0, round(self.n_students * self.adhd_prevalence))
        n_adhd = min(n_adhd, self.n_students)

        adhd_indices = set(self._rng.sample(range(self.n_students), n_adhd))

        for i in range(self.n_students):
            sid = f"S{i+1:02d}"
            is_adhd = i in adhd_indices

            if is_adhd:
                # Gender: boys 3.19x more likely — P(male|ADHD) = 3.19 / (3.19 + 1) ≈ 0.761
                gender = "male" if self._rng.random() < 0.761 else "female"
                subtype = self._rng.choices(ADHD_SUBTYPES, weights=_SUBTYPE_WEIGHTS, k=1)[0]
                severity = self._rng.choices(
                    list(_SEVERITY_WEIGHTS.keys()),
                    weights=list(_SEVERITY_WEIGHTS.values()),
                    k=1,
                )[0]
                age = self._rng.randint(7, 12)
                state = self._adhd_initial_state(subtype, severity)
            else:
                gender = "male" if self._rng.random() < 0.5 else "female"
                subtype = None
                severity = None
                age = self._rng.randint(7, 12)
                state = self._normal_initial_state()

            students.append(
                StudentState(
                    student_id=sid,
                    is_adhd=is_adhd,
                    adhd_subtype=subtype,
                    severity=severity,
                    gender=gender,
                    age=age,
                    state=state,
                )
            )

        return students

    def _normal_initial_state(self) -> dict[str, float]:
        rng = self._rng
        return {
            "distress_level": _clamp(0.10 + rng.gauss(0, 0.04)),
            "compliance": _clamp(0.80 + rng.gauss(0, 0.06)),
            "attention": _clamp(0.72 + rng.gauss(0, 0.06)),
            "escalation_risk": _clamp(0.05 + rng.gauss(0, 0.03)),
        }

    def _adhd_initial_state(self, subtype: str, severity: str) -> dict[str, float]:
        """Initial state from the closest matching ChildProfile, with subtype/severity nudge."""
        rng = self._rng
        # Try to match a profile by severity
        matched = self._match_profile(severity, subtype)
        if matched:
            base = dict(matched.state_priors) if matched.state_priors else {
                "distress_level": 0.30,
                "compliance": 0.25,
                "attention": 0.25,
                "escalation_risk": 0.25,
            }
        else:
            base = {
                "distress_level": 0.30,
                "compliance": 0.25,
                "attention": 0.25,
                "escalation_risk": 0.25,
            }

        amp = _SEVERITY_AMPLITUDE.get(severity, 0.10)

        # Apply scenario initial_state bias if available
        scenario_bias: dict[str, float] = {}
        if self.current_scenario:
            for k, v in self.current_scenario.initial_state.items():
                scenario_bias[k] = v

        state: dict[str, float] = {}
        for key in ("distress_level", "compliance", "attention", "escalation_risk"):
            val = float(base.get(key, 0.3))
            # Blend 30% scenario bias
            if scenario_bias:
                val = 0.70 * val + 0.30 * float(scenario_bias.get(key, val))
            val += rng.gauss(0, amp)
            state[key] = _clamp(val)

        # Subtype-specific nudges (Korean frequency data: KCI ART002478306)
        if subtype == "inattentive":
            state["attention"] = _clamp(state["attention"] - 0.10)
        elif subtype == "hyperactive_impulsive":
            state["escalation_risk"] = _clamp(state["escalation_risk"] + 0.10)
            state["distress_level"] = _clamp(state["distress_level"] + 0.05)
        elif subtype == "combined":
            state["attention"] = _clamp(state["attention"] - 0.07)
            state["escalation_risk"] = _clamp(state["escalation_risk"] + 0.07)

        return state

    def _match_profile(self, severity: str, subtype: str) -> ChildProfile | None:
        """Find best matching profile from loaded profiles."""
        if not self.profiles:
            return None

        # Prefer exact severity match with compatible subtype
        subtype_map = {
            "inattentive": "inattentive",
            "hyperactive_impulsive": "hyperactive",
            "combined": "combined",
        }
        subtype_kw = subtype_map.get(subtype, "")

        candidates = [p for p in self.profiles if p.severity == severity]
        if not candidates:
            candidates = self.profiles

        # Prefer profile whose name contains the subtype keyword
        for p in candidates:
            if subtype_kw in p.name.lower():
                return p
        return candidates[0]

    # ------------------------------------------------------------------
    # Student behavior update
    # ------------------------------------------------------------------

    def _update_student(self, student: StudentState, action: TeacherAction) -> None:
        """Advance one student's state for this turn."""
        rng = self._rng

        if student.is_adhd:
            amp = _SEVERITY_AMPLITUDE.get(student.severity or "mild", 0.08)
            prev = student.copy_state()
            proposed = {
                "distress_level": _clamp(prev["distress_level"] + rng.gauss(0, amp)),
                "compliance": _clamp(prev["compliance"] + rng.gauss(0, amp)),
                "attention": _clamp(prev["attention"] + rng.gauss(0, amp)),
                "escalation_risk": _clamp(prev["escalation_risk"] + rng.gauss(0, amp)),
            }

            # Apply transition constraints for actions targeting this student
            if (
                action.action_type in ("individual_intervention", "private_correction",
                                        "public_correction")
                and action.student_id == student.student_id
                and action.strategy in self.action_constraints
            ):
                proposed = constrain_transition(
                    action.strategy or action.action_type,
                    prev,
                    proposed,
                    constraints=self.action_constraints,
                )
            elif action.action_type == "class_instruction":
                # Class instruction has mild positive effect on ADHD students
                proposed["compliance"] = _clamp(proposed["compliance"] + rng.uniform(0.0, 0.05))
                proposed["attention"] = _clamp(proposed["attention"] + rng.uniform(0.0, 0.04))

            student.state = proposed
            student.exhibited_behaviors = self._sample_adhd_behaviors(
                student, current_turn=self.turn
            )
        else:
            # Normal student: small random walk around stable baseline
            student.state = {
                "distress_level": _clamp(
                    student.state["distress_level"] + rng.gauss(0, 0.02)
                ),
                "compliance": _clamp(
                    0.80 + rng.gauss(0, 0.05)
                ),
                "attention": _clamp(
                    0.72 + rng.gauss(0, 0.05)
                ),
                "escalation_risk": _clamp(
                    student.state["escalation_risk"] + rng.gauss(0, 0.01)
                ),
            }
            # ~15% chance of ONE mild ADHD-like behavior (Korean PPV=0.19, AAFP 2019)
            # This creates false-positive potential and makes identification harder
            if rng.random() < 0.15:
                student.exhibited_behaviors = [
                    self._rng.choice(_NORMAL_NOISE_BEHAVIORS)
                ]
            elif rng.random() < 0.15:
                # Mild off-task (non-ADHD noise)
                student.exhibited_behaviors = [
                    self._rng.choice(_NORMAL_BEHAVIORS[2:])
                ]
            else:
                student.exhibited_behaviors = [
                    self._rng.choice(_NORMAL_BEHAVIORS[:2])
                ]

    def _sample_adhd_behaviors(
        self, student: StudentState, current_turn: int = 0
    ) -> list[str]:
        """Sample 1-3 behaviors based on subtype, severity, and turn (warming-up period).

        Early turns (1-5): ADHD students show fewer distinctive behaviors.
        Mild ADHD: high stochastic suppression probability — hard to distinguish from normal.
        """
        pool = _ADHD_BEHAVIORS.get(student.adhd_subtype or "combined", _ADHD_BEHAVIORS["combined"])

        # Warming-up period: early turns suppress behavior expression
        # Turns 1-5 → 40% chance of showing NO distinctive behavior (use normal pool instead)
        if current_turn <= 5:
            if self._rng.random() < 0.40:
                return [self._rng.choice(_NORMAL_BEHAVIORS)]

        # Mild ADHD: stochastic suppression — 30% chance of hiding symptoms
        severity = student.severity or "mild"
        if severity == "mild" and self._rng.random() < 0.30:
            return [self._rng.choice(_NORMAL_BEHAVIORS)]

        # Moderate ADHD: 15% suppression chance
        if severity == "moderate" and self._rng.random() < 0.15:
            return [self._rng.choice(_NORMAL_BEHAVIORS)]

        # More severe state → more behaviors shown
        n_behaviors = 1
        if student.state["distress_level"] > 0.5 or student.state["escalation_risk"] > 0.5:
            n_behaviors = 2
        if student.state["escalation_risk"] > 0.7:
            n_behaviors = 3
        n_behaviors = min(n_behaviors, len(pool))
        return self._rng.sample(pool, n_behaviors)

    # ------------------------------------------------------------------
    # Action handlers — return incremental reward
    # ------------------------------------------------------------------

    def _reward_observe(self, student_id: str | None) -> float:
        """Focused observation: small positive reward for gathering information."""
        return 0.1

    def _apply_class_instruction(self) -> float:
        """Whole-class instruction: small benefit for all."""
        return 0.05

    def _apply_individual_intervention(
        self, student_id: str, strategy: str, public: bool
    ) -> float:
        student = self.get_student(student_id)
        if student is None:
            return -0.1

        student.intervention_history.append(strategy)
        prev = student.copy_state()

        # Build proposed state with small positive nudge
        amp = _SEVERITY_AMPLITUDE.get(student.severity or "mild", 0.08) if student.is_adhd else 0.04
        rng = self._rng
        proposed = {
            "distress_level": _clamp(prev["distress_level"] - rng.uniform(0, 0.10)),
            "compliance": _clamp(prev["compliance"] + rng.uniform(0, 0.12)),
            "attention": _clamp(prev["attention"] + rng.uniform(0, 0.10)),
            "escalation_risk": _clamp(prev["escalation_risk"] - rng.uniform(0, 0.08)),
        }

        if strategy in self.action_constraints:
            proposed = constrain_transition(strategy, prev, proposed, self.action_constraints)

        student.state = proposed

        # Correct intervention on an unidentified ADHD student yields higher reward
        base_reward = 0.3 if student.is_adhd else 0.1
        return base_reward

    def _apply_private_correction(self, student_id: str) -> float:
        """
        Private 1:1 correction — more effective (O'Leary 1970).
        Lower distress impact, higher compliance gain.
        """
        student = self.get_student(student_id)
        if student is None:
            return -0.1

        rng = self._rng
        prev = student.copy_state()
        student.state = {
            "distress_level": _clamp(prev["distress_level"] - rng.uniform(0.05, 0.15)),
            "compliance": _clamp(prev["compliance"] + rng.uniform(0.08, 0.18)),
            "attention": _clamp(prev["attention"] + rng.uniform(0.05, 0.12)),
            "escalation_risk": _clamp(prev["escalation_risk"] - rng.uniform(0.06, 0.14)),
        }
        student.intervention_history.append("private_correction")
        return 0.4 if student.is_adhd else 0.15

    def _apply_public_correction(self, student_id: str) -> float:
        """
        Public classroom correction — less effective, may increase distress.
        O'Leary 1970: public reprimands less effective and can backfire.
        """
        student = self.get_student(student_id)
        if student is None:
            return -0.1

        rng = self._rng
        prev = student.copy_state()
        # May increase distress, especially for ADHD
        distress_delta = rng.uniform(-0.02, 0.12) if student.is_adhd else rng.uniform(-0.02, 0.05)
        student.state = {
            "distress_level": _clamp(prev["distress_level"] + distress_delta),
            "compliance": _clamp(prev["compliance"] + rng.uniform(-0.02, 0.10)),
            "attention": _clamp(prev["attention"] + rng.uniform(0.0, 0.06)),
            "escalation_risk": _clamp(prev["escalation_risk"] + rng.uniform(-0.02, 0.08)),
        }
        student.intervention_history.append("public_correction")
        # Penalize if used on an ADHD student (research-based: less effective)
        return -0.05 if student.is_adhd else 0.05

    def _apply_identify(self, student_id: str, reasoning: str) -> float:
        """
        Teacher formally identifies a student as ADHD.
        Reward: +1 if correct, -1 if wrong (false positive).
        """
        student = self.get_student(student_id)
        if student is None:
            return -0.1

        if student_id in self.identified_adhd_ids:
            return 0.0  # already identified, no additional reward

        self.identified_adhd_ids.add(student_id)
        student.identified = True

        if student.is_adhd:
            return 1.0   # correct identification
        else:
            return -1.0  # false positive

    def _apply_generate_report(self, student_id: str) -> float:
        """Generate identification report. Reward only if student is truly ADHD."""
        student = self.get_student(student_id)
        if student is None:
            return -0.1
        return 0.5 if (student.is_adhd and student.identified) else -0.2

    # ------------------------------------------------------------------
    # Managed flag update
    # ------------------------------------------------------------------

    def _update_managed_flags(self) -> None:
        """
        Mark a student as managed if they exceed the compliance threshold
        for MANAGED_CONSECUTIVE_TURNS consecutive turns.
        """
        for student in self.students:
            if not student.is_adhd or student.managed:
                continue
            if student.state["compliance"] >= MANAGED_COMPLIANCE_THRESHOLD:
                student._consecutive_compliant_turns += 1
            else:
                student._consecutive_compliant_turns = 0

            if student._consecutive_compliant_turns >= MANAGED_CONSECUTIVE_TURNS:
                student.managed = True
                self.managed_ids.add(student.student_id)

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _make_observation(self, detail_student_id: str | None) -> ClassroomObservation:
        """
        Build ClassroomObservation visible to the teacher.

        Normal info: list of exhibited behaviors per student.
        Detail (after focused observe): also includes state_snapshot.
        """
        obs_list: list[StudentObservation] = []
        for student in self.students:
            state_snap = dict(student.state) if student.student_id == detail_student_id else None
            obs_list.append(
                StudentObservation(
                    student_id=student.student_id,
                    behaviors=list(student.exhibited_behaviors),
                    state_snapshot=state_snap,
                )
            )

        return ClassroomObservation(
            turn=self.turn,
            student_observations=obs_list,
            class_context=self.current_scenario.name if self.current_scenario else "free",
            identified_adhd_ids=list(self.identified_adhd_ids),
            managed_ids=list(self.managed_ids),
            all_complete=self.is_class_complete(),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))
