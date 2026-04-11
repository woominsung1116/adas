"""Held-out / cross-scenario validation for autoresearch calibration.

Provides a minimal, explicit layer that separates:
  - **fitting scenarios**: the simulation configurations used during
    autoresearch to optimize parameters
  - **held-out scenarios**: a disjoint set of simulation configurations
    used only for post-hoc validation, never for fitting

This is not a full cross-validation framework. It is a deliberately
narrow "evaluate a config on specified scenarios without optimizing"
helper that uses the existing real simulation path
(`OrchestratorV2` + `run_real_bundle` + `compute_combined_loss`).

A scenario is represented by a small dataclass that specifies the
simulation knobs relevant to calibration splits:
  - classroom archetype (pinned so it does not randomize)
  - seed
  - max_turns
  - n_students

These map directly to existing `OrchestratorV2` / `ClassroomV2` fields.
No new simulator behavior is introduced — only controlled determinism
over fields that already exist.

Typical usage:

    from src.calibration import (
        build_default_autoresearch_setup,
        ValidationScenario,
        evaluate_config_on_scenarios,
    )

    setup = build_default_autoresearch_setup(n_iterations=20)
    result = setup.orchestrator.run()

    heldout = [
        ValidationScenario(name="calm_1",    archetype="calm",     seed=100),
        ValidationScenario(name="chaotic_1", archetype="chaotic",  seed=200),
    ]
    report = evaluate_config_on_scenarios(
        config=result.global_best_config,
        scenarios=heldout,
        naturalness_targets=setup.evaluator.naturalness_targets,
        epidemiology_targets=setup.evaluator.epidemiology_targets,
    )
    print(report.summary())

Or via the setup convenience method:

    report = setup.validate_best_on_heldout(
        best_config=result.global_best_config,
        scenarios=heldout,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from .applier import parameter_override, ParameterOverrideError
from .constraints import (
    SupportedRule,
    UnsupportedRule,
    check_constraints,
)
from .loss import LossResult, compute_combined_loss
from .metrics import CalibrationResultBundle


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------


@dataclass
class ValidationScenario:
    """A single deterministic simulation scenario.

    Fields:
      name:       short identifier used in reports (e.g. "calm_seed100")
      archetype:  classroom archetype to PIN (must exist in
                  CLASSROOM_ARCHETYPES). If None, the simulator falls
                  back to its randomized default — but for validation
                  splits you almost always want to pin an archetype so
                  the scenario is reproducible.
      seed:       RNG seed for the simulation
      max_turns:  cap on turns per class
      n_students: students per class
      n_classes:  how many classes to run in this scenario
      note:       optional free-form tag (e.g. "training" / "heldout_1")
    """

    name: str
    archetype: str | None = None
    seed: int = 0
    max_turns: int = 200
    n_students: int = 20
    n_classes: int = 1
    note: str = ""


# ---------------------------------------------------------------------------
# Per-scenario result
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Outcome of evaluating a config on a single scenario."""

    scenario: ValidationScenario
    loss: LossResult
    bundle: CalibrationResultBundle | None = None
    error: str | None = None  # populated only on evaluation failure


@dataclass
class HeldOutValidationReport:
    """Aggregate report across multiple held-out scenarios."""

    config: dict[str, Any]
    training_scenarios: list[ValidationScenario] = field(default_factory=list)
    heldout_scenarios: list[ValidationScenario] = field(default_factory=list)
    training_results: list[ValidationResult] = field(default_factory=list)
    heldout_results: list[ValidationResult] = field(default_factory=list)

    def training_losses(self) -> list[float]:
        return [r.loss.total for r in self.training_results if r.error is None]

    def heldout_losses(self) -> list[float]:
        return [r.loss.total for r in self.heldout_results if r.error is None]

    def aggregate_training_loss(self) -> float | None:
        losses = self.training_losses()
        if not losses:
            return None
        return mean(losses)

    def aggregate_heldout_loss(self) -> float | None:
        losses = self.heldout_losses()
        if not losses:
            return None
        return mean(losses)

    def train_vs_heldout_gap(self) -> float | None:
        """Held-out loss minus training loss.

        Positive value indicates worse performance on held-out scenarios
        (possible overfitting). Negative is better on held-out. Returns
        None if either side has no valid results.
        """
        t = self.aggregate_training_loss()
        h = self.aggregate_heldout_loss()
        if t is None or h is None:
            return None
        return h - t

    def summary(self) -> str:
        """Multi-line human-readable report."""
        lines = [
            "=== Held-out Validation Report ===",
            f"Config: {len(self.config)} parameter overrides",
            f"Training scenarios: {len(self.training_scenarios)}",
            f"Held-out scenarios: {len(self.heldout_scenarios)}",
        ]

        t_agg = self.aggregate_training_loss()
        h_agg = self.aggregate_heldout_loss()
        gap = self.train_vs_heldout_gap()

        if t_agg is not None:
            lines.append(f"Training loss (mean): {t_agg:.6f}")
        if h_agg is not None:
            lines.append(f"Held-out loss (mean): {h_agg:.6f}")
        if gap is not None:
            direction = (
                "worse on held-out" if gap > 0 else
                "better on held-out" if gap < 0 else
                "equal"
            )
            lines.append(f"Gap (heldout - training): {gap:+.6f} ({direction})")

        lines.append("")
        if self.training_results:
            lines.append("Training per-scenario:")
            for r in self.training_results:
                _append_scenario_line(lines, r)
        if self.heldout_results:
            lines.append("")
            lines.append("Held-out per-scenario:")
            for r in self.heldout_results:
                _append_scenario_line(lines, r)
        return "\n".join(lines)


def _append_scenario_line(lines: list[str], result: ValidationResult) -> None:
    s = result.scenario
    tag = f"{s.name}"
    if s.archetype:
        tag += f" [{s.archetype}]"
    if result.error:
        lines.append(f"  {tag:30s} ERROR: {result.error}")
    else:
        lines.append(f"  {tag:30s} loss={result.loss.total:.6f}")


# ---------------------------------------------------------------------------
# Scenario split helpers
# ---------------------------------------------------------------------------


class ScenarioSplitError(ValueError):
    """Raised when a training/held-out split is not disjoint.

    A held-out validation report is only meaningful if the held-out
    scenarios were not also used for fitting. This error fails loudly
    instead of silently producing a misleading gap metric.
    """

    def __init__(
        self,
        overlapping_keys: list[tuple],
        training: list,
        heldout: list,
    ) -> None:
        self.overlapping_keys = list(overlapping_keys)
        self.training = list(training)
        self.heldout = list(heldout)
        key_str = ", ".join(repr(k) for k in overlapping_keys)
        super().__init__(
            f"training and held-out scenario splits overlap on "
            f"{len(overlapping_keys)} key(s): {key_str}. "
            f"Every held-out scenario must differ from every training "
            f"scenario in at least one of (archetype, seed, max_turns, "
            f"n_students, n_classes)."
        )


def _scenario_overlap_keys(
    a: list,
    b: list,
) -> list[tuple]:
    """Return the shared identity keys between two scenario lists."""
    def _key(s) -> tuple:
        return (s.archetype, s.seed, s.max_turns, s.n_students, s.n_classes)

    a_keys = {_key(s) for s in a}
    return sorted(k for k in a_keys if k in {_key(s) for s in b})


def scenarios_overlap(
    a: list[ValidationScenario],
    b: list[ValidationScenario],
) -> bool:
    """Return True if any (archetype, seed) pair appears in both lists.

    Used by tests to ensure training/held-out splits are actually disjoint.
    Names are ignored (they are labels, not identity). An unpinned
    archetype (None) is treated as its own value so None ≠ "calm".
    """
    def _key(s: ValidationScenario) -> tuple:
        return (s.archetype, s.seed, s.max_turns, s.n_students, s.n_classes)

    a_keys = {_key(s) for s in a}
    b_keys = {_key(s) for s in b}
    return bool(a_keys & b_keys)


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------


def _run_scenario_once(
    scenario: ValidationScenario,
    config: dict[str, Any],
    naturalness_targets: list,
    epidemiology_targets: list,
    supported_rules: list[SupportedRule] | None = None,
    unsupported_rules: list[UnsupportedRule] | None = None,
) -> ValidationResult:
    """Evaluate a config once on a single scenario and return a result.

    This is evaluation-only: no search, no proposer, no checkpointing.
    Uses the same real simulator path as calibration
    (OrchestratorV2 + run_real_bundle + compute_combined_loss) so loss
    semantics are identical.

    Supported constraints, if provided, are checked BEFORE simulation
    so a violating config produces a ValidationResult with error set
    rather than silently running the simulator.
    """
    from src.simulation.orchestrator_v2 import OrchestratorV2
    from .adapters import run_real_bundle

    # Constraint pre-check: evaluation should also respect the loader's
    # enforcement policy (same as orchestrator pre-evaluation check).
    if supported_rules:
        check = check_constraints(
            config, supported_rules, unsupported_rules or []
        )
        if not check.valid:
            msgs = check.describe_violations()
            return ValidationResult(
                scenario=scenario,
                loss=LossResult(total=float("inf")),
                bundle=None,
                error=f"constraint_violation: {'; '.join(msgs)}",
            )

    bundle = CalibrationResultBundle(histories=[])
    try:
        with parameter_override(config) as apply_errors:
            if apply_errors:
                return ValidationResult(
                    scenario=scenario,
                    loss=LossResult(total=float("inf")),
                    bundle=None,
                    error=f"parameter_override: {'; '.join(apply_errors)}",
                )
            for class_idx in range(scenario.n_classes):
                class_seed = scenario.seed + class_idx
                orch = OrchestratorV2(
                    n_students=scenario.n_students,
                    max_classes=1,
                    seed=class_seed,
                )
                if scenario.archetype is not None:
                    orch.classroom.set_archetype(scenario.archetype)
                orch.classroom.MAX_TURNS = scenario.max_turns
                sub = run_real_bundle(orch, n_classes=1)
                bundle.histories.extend(sub.histories)

            loss_result = compute_combined_loss(
                bundle,
                naturalness_targets=naturalness_targets,
                epidemiology_targets=epidemiology_targets,
            )
    except Exception as exc:
        return ValidationResult(
            scenario=scenario,
            loss=LossResult(total=float("inf")),
            bundle=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    return ValidationResult(
        scenario=scenario,
        loss=loss_result,
        bundle=bundle,
    )


def evaluate_config_on_scenarios(
    *,
    config: dict[str, Any],
    scenarios: list[ValidationScenario],
    naturalness_targets: list,
    epidemiology_targets: list,
    supported_rules: list[SupportedRule] | None = None,
    unsupported_rules: list[UnsupportedRule] | None = None,
) -> list[ValidationResult]:
    """Run a config on each scenario and return per-scenario results."""
    out: list[ValidationResult] = []
    for scenario in scenarios:
        result = _run_scenario_once(
            scenario=scenario,
            config=config,
            naturalness_targets=naturalness_targets,
            epidemiology_targets=epidemiology_targets,
            supported_rules=supported_rules,
            unsupported_rules=unsupported_rules,
        )
        out.append(result)
    return out


def build_held_out_report(
    *,
    config: dict[str, Any],
    training_scenarios: list[ValidationScenario],
    heldout_scenarios: list[ValidationScenario],
    naturalness_targets: list,
    epidemiology_targets: list,
    supported_rules: list[SupportedRule] | None = None,
    unsupported_rules: list[UnsupportedRule] | None = None,
) -> HeldOutValidationReport:
    """Run both sides of a train/heldout split and build an aggregate report.

    Enforces split disjointness: if any held-out scenario shares an
    identity key (archetype, seed, max_turns, n_students, n_classes)
    with a training scenario, `ScenarioSplitError` is raised before
    any simulator work runs. If `training_scenarios` is empty, no
    overlap check is performed (there is nothing to overlap with).
    """
    if training_scenarios and heldout_scenarios:
        overlap = _scenario_overlap_keys(training_scenarios, heldout_scenarios)
        if overlap:
            raise ScenarioSplitError(
                overlapping_keys=overlap,
                training=training_scenarios,
                heldout=heldout_scenarios,
            )

    training_results = evaluate_config_on_scenarios(
        config=config,
        scenarios=training_scenarios,
        naturalness_targets=naturalness_targets,
        epidemiology_targets=epidemiology_targets,
        supported_rules=supported_rules,
        unsupported_rules=unsupported_rules,
    )
    heldout_results = evaluate_config_on_scenarios(
        config=config,
        scenarios=heldout_scenarios,
        naturalness_targets=naturalness_targets,
        epidemiology_targets=epidemiology_targets,
        supported_rules=supported_rules,
        unsupported_rules=unsupported_rules,
    )
    return HeldOutValidationReport(
        config=dict(config),
        training_scenarios=list(training_scenarios),
        heldout_scenarios=list(heldout_scenarios),
        training_results=training_results,
        heldout_results=heldout_results,
    )
