"""Calibration metric extraction — bridge between simulation results and YAML targets.

This module provides the `MetricRegistry`, a string-keyed dispatcher that
computes scalar metrics from a `CalibrationResultBundle`. It connects the
declarative target YAMLs (naturalness_targets.yaml, epidemiology_targets.yaml)
to actual numbers from simulation outputs.

Design (정리.md §25.13):
  - Post-hoc extraction from class histories and per-turn events
  - Callable registry keyed by YAML metric path strings
  - Unsupported metrics raise NotImplementedError explicitly (no silent fakes)
  - Approximations are documented inline

Supported metrics in this pass (see bottom of file for registry):

Epidemiology side (8):
  - prevalence.adhd_total
  - prevalence.adhd_inattentive
  - prevalence.adhd_hyperactive
  - prevalence.adhd_combined
  - prevalence.adhd_male_to_female_ratio
  - teacher.identification_fraction_by_turn[475]
  - teacher.identification_fraction_by_turn[950]
  - behavior.normal_on_task_percentage
  - behavior.adhd_on_task_relative_to_normal

Naturalness side (4):
  - behavior.seat_leaving_adhd_normal_ratio
  - teacher.first_suspicion_turn_median
  - teacher.patience_end_of_day_vs_start_ratio
  - intervention.empathic_compliance_gain

All other YAML metric paths raise NotImplementedError when invoked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Simulation result contract
# ---------------------------------------------------------------------------


@dataclass
class ClassHistory:
    """Post-hoc snapshot of one classroom simulation's observable data.

    This is the minimal contract required by the supported metric extractors.
    Producers (orchestrator_v2._compile_class_result, etc.) can fill in only
    the fields needed by the metrics they want to evaluate.

    Fields:
      metrics: ClassMetrics instance from eval.growth_metrics (prevalence,
               identification counts, etc.). None if not available.
      events:  list of per-turn event dicts from orchestrator stream_class,
               each with keys {"turn", "students", "teacher_action", ...}.
               Used for behavior frequency + teacher timing metrics.
      students_roster: list of dicts with {"id", "profile_type", "gender",
               "is_adhd"} — used for prevalence and gender-ratio metrics.
      reports: list of IdentificationReport objects (for subtype accuracy).
      teacher_patience_log: optional per-turn list of teacher patience values
               (for patience_end_of_day_vs_start_ratio). Optional.
      intervention_outcomes: list of dicts {"strategy": str, "pre_compliance":
               float, "post_compliance": float, "student_id": str}. Optional.
    """

    metrics: Any = None
    events: list[dict] = field(default_factory=list)
    students_roster: list[dict] = field(default_factory=list)
    reports: list = field(default_factory=list)
    teacher_patience_log: list[float] | None = None
    intervention_outcomes: list[dict] = field(default_factory=list)
    # Map of {student_id: first-suspicion turn} from the orchestrator's
    # suspicion tracker (adhd_indicator_score >= 0.5 crossing). Preferred
    # input for teacher.first_suspicion_turn_median when present.
    first_suspicion_turns: dict[str, int] = field(default_factory=dict)


@dataclass
class CalibrationResultBundle:
    """Aggregate of N classroom histories for calibration evaluation.

    Metric extractors aggregate across all histories in the bundle.
    For single-class tests, N=1 is valid.
    """

    histories: list[ClassHistory] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.histories)

    def iter_students(self):
        """Yield every (history, student_dict) pair across all histories."""
        for h in self.histories:
            for s in h.students_roster:
                yield h, s

    def iter_events(self):
        """Yield every (history, event_dict) pair across all histories."""
        for h in self.histories:
            for ev in h.events:
                yield h, ev


# ---------------------------------------------------------------------------
# Metric extractors (each returns a float or raises)
# ---------------------------------------------------------------------------


def _collect_all_students(bundle: CalibrationResultBundle) -> list[dict]:
    return [s for _, s in bundle.iter_students()]


def _total_student_count(bundle: CalibrationResultBundle) -> int:
    return sum(len(h.students_roster) for h in bundle.histories)


# ----- Prevalence --------------------------------------------------------


def _prevalence_adhd_total(bundle: CalibrationResultBundle) -> float:
    total = _total_student_count(bundle)
    if total == 0:
        raise ValueError("empty bundle for prevalence metric")
    n_adhd = sum(1 for s in _collect_all_students(bundle) if s.get("is_adhd"))
    return n_adhd / total


def _prevalence_subtype(subtype_prefixes: tuple[str, ...]) -> Callable:
    """Build a subtype prevalence extractor.

    A student counts toward the subtype if profile_type starts with any
    of the given prefixes. This handles both pure subtypes and comorbid
    variants (e.g., adhd_inattentive matches "adhd_inattentive",
    adhd_i_plus_anxiety also matches "adhd_i_").
    """
    def _extract(bundle: CalibrationResultBundle) -> float:
        total = _total_student_count(bundle)
        if total == 0:
            raise ValueError("empty bundle for prevalence metric")
        n_match = sum(
            1 for s in _collect_all_students(bundle)
            if any(s.get("profile_type", "").startswith(p) for p in subtype_prefixes)
        )
        return n_match / total
    return _extract


# ADHD-I bucket includes pure inattentive and inattentive-based comorbidities
_prevalence_adhd_inattentive = _prevalence_subtype(
    ("adhd_inattentive", "adhd_i_plus_")
)

# ADHD-H bucket includes pure hyperactive-impulsive and H-based comorbidities
_prevalence_adhd_hyperactive = _prevalence_subtype(
    ("adhd_hyperactive_impulsive", "adhd_h_plus_")
)

# ADHD-combined bucket includes combined and combined-based comorbidities
_prevalence_adhd_combined = _prevalence_subtype(
    ("adhd_combined", "adhd_c_plus_")
)


def _prevalence_male_female_ratio(bundle: CalibrationResultBundle) -> float:
    """Compute male:female ADHD ratio. Returns 0 if no female ADHD cases.

    Uses the profile_type.startswith("adhd_") test so all comorbid ADHD
    variants are included in the ratio calculation.
    """
    male = 0
    female = 0
    for s in _collect_all_students(bundle):
        if not s.get("profile_type", "").startswith("adhd_"):
            continue
        gender = s.get("gender", "unknown")
        if gender == "male":
            male += 1
        elif gender == "female":
            female += 1
    if female == 0:
        # Edge case: no female ADHD — return male count as ratio (cap high).
        # Prefer this to ZeroDivisionError so calibration loop can proceed.
        return float(male) if male > 0 else 0.0
    return male / female


# ----- Teacher identification timing ------------------------------------


def _identification_fraction_by_turn(target_turn: int) -> Callable:
    """Fraction of ADHD students identified by `target_turn`.

    Scans per-turn events for `identifications` or `teacher_action.action_type ==
    "identify_adhd"` entries and counts unique student_ids that were flagged
    on or before target_turn.
    """
    def _extract(bundle: CalibrationResultBundle) -> float:
        total_adhd = 0
        flagged_by_turn: set[str] = set()

        for h in bundle.histories:
            # Count ADHD ground truth
            adhd_ids = {
                s["id"] for s in h.students_roster
                if s.get("is_adhd") and s.get("id") is not None
            }
            total_adhd += len(adhd_ids)

            # Scan events
            for ev in h.events:
                turn = ev.get("turn", 0)
                if turn > target_turn:
                    break  # events are turn-ordered
                action = ev.get("teacher_action") or {}
                if action.get("action_type") == "identify_adhd":
                    sid = action.get("student_id")
                    if sid in adhd_ids:
                        # Namespace by history index to avoid cross-class collisions
                        hid = id(h)
                        flagged_by_turn.add(f"{hid}:{sid}")

        if total_adhd == 0:
            raise ValueError("no ADHD students in bundle")
        return len(flagged_by_turn) / total_adhd
    return _extract


_teacher_identification_fraction_475 = _identification_fraction_by_turn(475)
_teacher_identification_fraction_950 = _identification_fraction_by_turn(950)


# ----- Behavior frequencies ---------------------------------------------


def _count_behavior_events(
    events: list[dict],
    student_ids: set[str],
    behavior_names: set[str],
) -> int:
    """Count turns in which any student in the set exhibited any of the
    given behaviors.
    """
    count = 0
    for ev in events:
        for st in ev.get("students", []) or []:
            if st.get("id") in student_ids:
                behaviors = st.get("behaviors", []) or []
                if any(b in behavior_names for b in behaviors):
                    count += 1
    return count


_ON_TASK_BEHAVIORS = {
    "on_task", "listening", "writing", "reading", "collaborating",
}

_SEAT_LEAVING_BEHAVIORS = {"seat_leaving", "running"}


def _behavior_normal_on_task(bundle: CalibrationResultBundle) -> float:
    """Fraction of turn-observations where normal students show on-task behavior.

    Approximation: counts across all (student, turn) observations where
    is_adhd=False. Returns 0-1 fraction.
    """
    total_obs = 0
    on_task_obs = 0
    for h in bundle.histories:
        normal_ids = {
            s["id"] for s in h.students_roster
            if not s.get("is_adhd") and s.get("id") is not None
        }
        for ev in h.events:
            for st in ev.get("students", []) or []:
                if st.get("id") in normal_ids:
                    total_obs += 1
                    behaviors = st.get("behaviors", []) or []
                    if any(b in _ON_TASK_BEHAVIORS for b in behaviors):
                        on_task_obs += 1
    if total_obs == 0:
        raise ValueError("no normal-student observations in bundle")
    return on_task_obs / total_obs


def _behavior_adhd_on_task_relative(bundle: CalibrationResultBundle) -> float:
    """ADHD students' on-task rate divided by normal students' on-task rate.

    Returns a fraction in [0, ~1.2]. Values below 1.0 mean ADHD students
    are less on-task than normal peers (expected direction).
    """
    total_normal = 0
    on_normal = 0
    total_adhd = 0
    on_adhd = 0
    for h in bundle.histories:
        normal_ids = {
            s["id"] for s in h.students_roster
            if not s.get("is_adhd") and s.get("id") is not None
        }
        adhd_ids = {
            s["id"] for s in h.students_roster
            if s.get("is_adhd") and s.get("id") is not None
        }
        for ev in h.events:
            for st in ev.get("students", []) or []:
                sid = st.get("id")
                behaviors = st.get("behaviors", []) or []
                is_on_task = any(b in _ON_TASK_BEHAVIORS for b in behaviors)
                if sid in normal_ids:
                    total_normal += 1
                    if is_on_task:
                        on_normal += 1
                elif sid in adhd_ids:
                    total_adhd += 1
                    if is_on_task:
                        on_adhd += 1

    if total_normal == 0 or total_adhd == 0:
        raise ValueError("insufficient observations for ratio")
    normal_rate = on_normal / total_normal
    adhd_rate = on_adhd / total_adhd
    if normal_rate == 0:
        return 0.0
    return adhd_rate / normal_rate


def _behavior_seat_leaving_ratio(bundle: CalibrationResultBundle) -> float:
    """ADHD seat-leaving rate / normal seat-leaving rate.

    Expected range: 2.5 - 5.5 (KCI ART002794420).
    Returns ratio; if normal rate is 0, returns ADHD count (capped) to
    avoid division by zero while still reflecting direction.
    """
    total_normal_turns = 0
    seat_normal = 0
    total_adhd_turns = 0
    seat_adhd = 0
    for h in bundle.histories:
        normal_ids = {
            s["id"] for s in h.students_roster
            if not s.get("is_adhd") and s.get("id") is not None
        }
        adhd_ids = {
            s["id"] for s in h.students_roster
            if s.get("is_adhd") and s.get("id") is not None
        }
        for ev in h.events:
            for st in ev.get("students", []) or []:
                sid = st.get("id")
                behaviors = st.get("behaviors", []) or []
                has_seat_leaving = any(
                    b in _SEAT_LEAVING_BEHAVIORS for b in behaviors
                )
                if sid in normal_ids:
                    total_normal_turns += 1
                    if has_seat_leaving:
                        seat_normal += 1
                elif sid in adhd_ids:
                    total_adhd_turns += 1
                    if has_seat_leaving:
                        seat_adhd += 1

    if total_normal_turns == 0 or total_adhd_turns == 0:
        raise ValueError("insufficient turns for seat-leaving ratio")
    normal_rate = seat_normal / total_normal_turns
    adhd_rate = seat_adhd / total_adhd_turns
    if normal_rate < 1e-9:
        # Avoid division blowup; cap at a large sentinel value
        return min(adhd_rate * 100, 10.0) if adhd_rate > 0 else 1.0
    return adhd_rate / normal_rate


# ----- Teacher timing / patience ----------------------------------------


def _teacher_first_suspicion_turn_median(bundle: CalibrationResultBundle) -> float:
    """Median turn at which teacher first began suspecting any ADHD student.

    Prefers the real suspicion signal (orchestrator's `first_suspicion_turns`
    map, populated when a student first crosses the adhd_indicator_score
    suspicion threshold). Falls back to `identify_adhd` action timing for
    histories that lack suspicion data (e.g., legacy or synthetic bundles).

    "First suspicion" is taken per-class as the earliest turn at which any
    true-ADHD student entered suspicion. The median across classes is
    returned.
    """
    first_turns: list[int] = []
    for h in bundle.histories:
        adhd_ids = {
            s["id"] for s in h.students_roster
            if s.get("is_adhd") and s.get("id") is not None
        }
        if not adhd_ids:
            continue

        # Preferred: real suspicion tracker data from orchestrator
        earliest_turn: int | None = None
        if h.first_suspicion_turns:
            adhd_suspicion_turns = [
                t for sid, t in h.first_suspicion_turns.items()
                if sid in adhd_ids
            ]
            if adhd_suspicion_turns:
                earliest_turn = min(adhd_suspicion_turns)

        # Fallback: identify_adhd action as proxy (legacy synthetic bundles)
        if earliest_turn is None:
            for ev in h.events:
                action = ev.get("teacher_action") or {}
                if action.get("action_type") == "identify_adhd":
                    sid = action.get("student_id")
                    if sid in adhd_ids:
                        earliest_turn = ev.get("turn", 0)
                        break

        if earliest_turn is not None:
            first_turns.append(earliest_turn)

    if not first_turns:
        raise ValueError(
            "no suspicion or identification data found in bundle"
        )
    return float(median(first_turns))


def _teacher_patience_end_vs_start_ratio(bundle: CalibrationResultBundle) -> float:
    """Ratio of teacher patience at end of day vs start of day.

    REQUIRES: teacher_patience_log in each ClassHistory (per-turn patience
    values). If absent, raises ValueError.

    Approximation: averages the ratio across days and classes. Assumes
    periods_per_day=5 (matches ClassroomV2.PERIODS_PER_DAY).
    """
    periods_per_day = 5
    day_ratios: list[float] = []

    for h in bundle.histories:
        log = h.teacher_patience_log
        if not log:
            continue
        # Walk by day boundaries
        n_days = len(log) // periods_per_day
        for day_idx in range(n_days):
            start = log[day_idx * periods_per_day]
            end = log[day_idx * periods_per_day + periods_per_day - 1]
            if start > 1e-9:
                day_ratios.append(end / start)

    if not day_ratios:
        raise ValueError("no teacher_patience_log data in bundle")
    return sum(day_ratios) / len(day_ratios)


# ----- Intervention outcomes --------------------------------------------


def _intervention_empathic_compliance_gain(bundle: CalibrationResultBundle) -> float:
    """Mean compliance gain attributable to empathic interventions.

    REQUIRES: intervention_outcomes in each ClassHistory with strategy name
    matching "empathic" or "empathic_intervention".
    """
    gains: list[float] = []
    for h in bundle.histories:
        for outcome in h.intervention_outcomes:
            strategy = (outcome.get("strategy") or "").lower()
            if "empath" not in strategy:
                continue
            pre = outcome.get("pre_compliance")
            post = outcome.get("post_compliance")
            if pre is None or post is None:
                continue
            gains.append(post - pre)

    if not gains:
        raise ValueError("no empathic intervention outcomes in bundle")
    return sum(gains) / len(gains)


# ---------------------------------------------------------------------------
# Registry — string metric path → extractor callable
# ---------------------------------------------------------------------------


METRIC_REGISTRY: dict[str, Callable[[CalibrationResultBundle], float]] = {
    # Epidemiology
    "prevalence.adhd_total": _prevalence_adhd_total,
    "prevalence.adhd_inattentive": _prevalence_adhd_inattentive,
    "prevalence.adhd_hyperactive": _prevalence_adhd_hyperactive,
    "prevalence.adhd_combined": _prevalence_adhd_combined,
    "prevalence.adhd_male_to_female_ratio": _prevalence_male_female_ratio,
    "teacher.identification_fraction_by_turn[475]": _teacher_identification_fraction_475,
    "teacher.identification_fraction_by_turn[950]": _teacher_identification_fraction_950,
    "behavior.normal_on_task_percentage": _behavior_normal_on_task,
    "behavior.adhd_on_task_relative_to_normal": _behavior_adhd_on_task_relative,

    # Naturalness
    "behavior.seat_leaving_adhd_normal_ratio": _behavior_seat_leaving_ratio,
    "teacher.first_suspicion_turn_median": _teacher_first_suspicion_turn_median,
    "teacher.patience_end_of_day_vs_start_ratio": _teacher_patience_end_vs_start_ratio,
    "intervention.empathic_compliance_gain": _intervention_empathic_compliance_gain,
}


def resolve_metric(metric_path: str) -> Callable[[CalibrationResultBundle], float]:
    """Look up a metric by YAML path string.

    Raises NotImplementedError for unknown paths, listing available metrics
    in the error message for easier debugging.
    """
    if metric_path not in METRIC_REGISTRY:
        supported = "\n  ".join(sorted(METRIC_REGISTRY.keys()))
        raise NotImplementedError(
            f"Metric path {metric_path!r} is not implemented.\n"
            f"Supported metrics:\n  {supported}"
        )
    return METRIC_REGISTRY[metric_path]


def supported_metrics() -> list[str]:
    """Return the sorted list of currently supported metric paths."""
    return sorted(METRIC_REGISTRY.keys())
