from __future__ import annotations
from dataclasses import dataclass, field


STATE_KEYS = ("distress_level", "compliance", "attention", "escalation_risk")


@dataclass(frozen=True)
class ActionConstraint:
    action_name: str
    delta_bounds: dict[str, tuple[float, float]]
    evidence: list[str] = field(default_factory=list)
    rationale: str = ""

    def clip_state(self, prev_state: dict[str, float], proposed_state: dict[str, float]) -> dict[str, float]:
        clipped = {}
        for key in STATE_KEYS:
            prev_value = float(prev_state.get(key, 0.5))
            proposed_value = float(proposed_state.get(key, prev_value))
            lo, hi = self.delta_bounds.get(key, (-1.0, 1.0))
            delta = max(lo, min(hi, proposed_value - prev_value))
            clipped[key] = max(0.0, min(1.0, prev_value + delta))
        return clipped


DEFAULT_ACTION_CONSTRAINTS = {
    "transition_warning": ActionConstraint(
        action_name="transition_warning",
        delta_bounds={
            "distress_level": (-0.12, 0.08),
            "compliance": (-0.05, 0.18),
            "attention": (-0.04, 0.20),
            "escalation_risk": (-0.12, 0.08),
        },
        evidence=[
            "Gaastra et al., 2016",
            "Staff et al., 2021",
            "Staff et al., 2023",
        ],
        rationale="Antecedent structure should usually increase task orientation without causing a large distress spike.",
    ),
    "offer_choice": ActionConstraint(
        action_name="offer_choice",
        delta_bounds={
            "distress_level": (-0.15, 0.06),
            "compliance": (0.00, 0.22),
            "attention": (-0.05, 0.18),
            "escalation_risk": (-0.14, 0.05),
        },
        evidence=["DuPaul et al., 2022", "Staff et al., 2021"],
        rationale="Choice can support autonomy and improve compliance when the transition demand is clear.",
    ),
    "labeled_praise": ActionConstraint(
        action_name="labeled_praise",
        delta_bounds={
            "distress_level": (-0.16, 0.04),
            "compliance": (0.00, 0.18),
            "attention": (0.00, 0.18),
            "escalation_risk": (-0.16, 0.04),
        },
        evidence=["Gaastra et al., 2016", "Staff et al., 2021"],
        rationale="Positive reinforcement should not sharply increase distress or escalation risk.",
    ),
    "visual_schedule_cue": ActionConstraint(
        action_name="visual_schedule_cue",
        delta_bounds={
            "distress_level": (-0.14, 0.06),
            "compliance": (-0.03, 0.18),
            "attention": (0.00, 0.24),
            "escalation_risk": (-0.16, 0.05),
        },
        evidence=["DuPaul et al., 2022", "Gaastra et al., 2016"],
        rationale="Visual structure should help orient attention during transitions.",
    ),
    "break_offer": ActionConstraint(
        action_name="break_offer",
        delta_bounds={
            "distress_level": (-0.24, 0.05),
            "compliance": (-0.08, 0.14),
            "attention": (-0.10, 0.12),
            "escalation_risk": (-0.20, 0.05),
        },
        evidence=["DuPaul et al., 2022", "Yegencik et al., 2025"],
        rationale="A brief break can reduce distress, but immediate task compliance may improve more slowly.",
    ),
    "empathic_acknowledgment": ActionConstraint(
        action_name="empathic_acknowledgment",
        delta_bounds={
            "distress_level": (-0.20, 0.04),
            "compliance": (-0.04, 0.16),
            "attention": (-0.04, 0.14),
            "escalation_risk": (-0.18, 0.04),
        },
        evidence=["DuPaul et al., 2022", "Staff et al., 2021"],
        rationale="Emotion validation should primarily dampen distress and escalation before large compliance gains appear.",
    ),
    "redirect_attention": ActionConstraint(
        action_name="redirect_attention",
        delta_bounds={
            "distress_level": (-0.14, 0.08),
            "compliance": (-0.04, 0.16),
            "attention": (0.00, 0.26),
            "escalation_risk": (-0.12, 0.08),
        },
        evidence=["Gaastra et al., 2016", "Yegencik et al., 2025"],
        rationale="Redirecting to the next task should mostly increase engagement but may not fully resolve resistance.",
    ),
    "countdown_timer": ActionConstraint(
        action_name="countdown_timer",
        delta_bounds={
            "distress_level": (-0.14, 0.10),
            "compliance": (-0.02, 0.18),
            "attention": (0.00, 0.22),
            "escalation_risk": (-0.14, 0.08),
        },
        evidence=["DuPaul et al., 2022", "Staff et al., 2023"],
        rationale="Temporal predictability should improve transition readiness while limiting sharp escalation.",
    ),
    "collaborative_problem_solving": ActionConstraint(
        action_name="collaborative_problem_solving",
        delta_bounds={
            "distress_level": (-0.18, 0.05),
            "compliance": (0.00, 0.20),
            "attention": (0.00, 0.20),
            "escalation_risk": (-0.18, 0.05),
        },
        evidence=["Gaastra et al., 2016", "DuPaul et al., 2022"],
        rationale="Self-regulation and collaborative supports should produce gradual improvements rather than instant jumps.",
    ),
    "ignore_wait": ActionConstraint(
        action_name="ignore_wait",
        delta_bounds={
            "distress_level": (-0.10, 0.14),
            "compliance": (-0.08, 0.10),
            "attention": (-0.10, 0.08),
            "escalation_risk": (-0.08, 0.14),
        },
        evidence=["Staff et al., 2021", "Staff et al., 2023"],
        rationale="Planned waiting may help some transitions but idle time can also increase hyperactivity.",
    ),
    "firm_boundary": ActionConstraint(
        action_name="firm_boundary",
        delta_bounds={
            "distress_level": (-0.08, 0.18),
            "compliance": (-0.06, 0.20),
            "attention": (-0.08, 0.16),
            "escalation_risk": (-0.08, 0.16),
        },
        evidence=["Staff et al., 2021", "Gaastra et al., 2016"],
        rationale="Clear boundaries can increase compliance but may also raise distress for high-reactivity profiles.",
    ),
    "sensory_support": ActionConstraint(
        action_name="sensory_support",
        delta_bounds={
            "distress_level": (-0.22, 0.05),
            "compliance": (-0.06, 0.16),
            "attention": (-0.04, 0.16),
            "escalation_risk": (-0.20, 0.05),
        },
        evidence=["DuPaul et al., 2022", "Gaastra et al., 2016"],
        rationale="Sensory support should mainly reduce distress and escalation, with compliance gains following more gradually.",
    ),
}


def constrain_transition(
    action_name: str,
    prev_state: dict[str, float],
    proposed_state: dict[str, float],
    constraints: dict[str, ActionConstraint] | None = None,
) -> dict[str, float]:
    constraint_map = constraints or DEFAULT_ACTION_CONSTRAINTS
    constraint = constraint_map.get(action_name)
    if constraint is None:
        return {
            key: max(0.0, min(1.0, float(proposed_state.get(key, prev_state.get(key, 0.5)))))
            for key in STATE_KEYS
        }
    return constraint.clip_state(prev_state, proposed_state)
