"""Default autoresearch setup — harness YAML → ready-to-run orchestrator.

This module wires the Phase 4.5 YAML loader into the Phase 4 autoresearch
orchestrator so real calibration runs can be launched from the canonical
`.harness/student_parameter_ranges.yaml` file with minimal boilerplate.

Before this module, a calibration run required ~10 lines of manual
assembly:

    space = SearchSpace([ParameterSpec(...), ...])  # or load manually
    ev = build_default_evaluator(n_classes=3, max_turns=200)
    orch = AutoresearchOrchestrator(
        space=space, evaluator=ev, proposer_kind="random",
        n_iterations=10, n_starts=3, seed=42,
        results_dir=Path(".harness"),
    )
    result = orch.run()

After:

    setup = build_default_autoresearch_setup()
    result = setup.orchestrator.run()
    print(setup.loaded_space.constraints)  # still accessible

Both the `LoadedSearchSpace` and the assembled `AutoresearchOrchestrator`
are preserved on the returned bundle so downstream code can access
constraints, metadata, and unsupported section markers without calling
the loader again.

This module does NOT:
  - enforce constraints (still deferred to a future pass)
  - implement new proposer algorithms or metric extractors
  - provide a CLI (keeping scope tight)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .orchestrator import AutoresearchOrchestrator
from .applier import DefaultEvaluator, build_default_evaluator
from .constraints import (
    SupportedRule,
    UnsupportedRule,
    parse_constraints,
)
from .search_space_loader import (
    LoadedSearchSpace,
    load_default_search_space,
)


# ---------------------------------------------------------------------------
# Result bundle
# ---------------------------------------------------------------------------


@dataclass
class DefaultAutoresearchSetup:
    """Everything a caller needs to run and introspect a default calibration.

    Fields:
      loaded_space: full LoadedSearchSpace (constraints + metadata + sources
                    + unsupported_sections all preserved)
      evaluator:    DefaultEvaluator already configured with target YAMLs
      orchestrator: AutoresearchOrchestrator ready to `.run()`
      supported_rules:  parsed SupportedRule list (forwarded to orchestrator
                        for pre-evaluation filtering)
      unsupported_rules: parsed UnsupportedRule list (surfaced for
                         transparency, never enforced)

    Access patterns:
        setup = build_default_autoresearch_setup(n_iterations=10)
        setup.orchestrator.run()
        print(setup.loaded_space.constraints)   # raw YAML dicts
        print(setup.supported_rules)             # parsed + enforced
        print(setup.unsupported_rules)           # parsed + never enforced
    """

    loaded_space: LoadedSearchSpace
    evaluator: DefaultEvaluator
    orchestrator: AutoresearchOrchestrator
    supported_rules: list[SupportedRule] = field(default_factory=list)
    unsupported_rules: list[UnsupportedRule] = field(default_factory=list)

    def summary(self) -> str:
        """One-line human-readable summary of the setup."""
        ls = self.loaded_space
        o = self.orchestrator
        return (
            f"DefaultAutoresearchSetup("
            f"n_params={len(ls.space)}, "
            f"n_constraints={len(ls.constraints)}, "
            f"supported_rules={len(self.supported_rules)}, "
            f"unsupported_rules={len(self.unsupported_rules)}, "
            f"unsupported_sections={len(ls.unsupported_sections)}, "
            f"proposer={o.proposer_kind}, "
            f"n_starts={o.n_starts}, "
            f"n_iterations={o.n_iterations}, "
            f"results_dir={o.results_dir})"
        )

    def validate_best_on_heldout(
        self,
        *,
        best_config: dict[str, Any],
        scenarios: list,
        training_scenarios: list | None = None,
    ):
        """Convenience method: run held-out validation on `best_config`.

        Uses the evaluator's loaded naturalness / epidemiology targets
        and, if present, the orchestrator's supported/unsupported
        constraints so validation respects the same enforcement policy
        as calibration.

        Args:
            best_config: the config to validate (typically
                         `result.global_best_config` from an orchestrator run)
            scenarios: held-out `ValidationScenario` list
            training_scenarios: optional training-side scenarios;
                                if provided, the returned report also
                                includes train-vs-heldout gap

        Returns:
            HeldOutValidationReport
        """
        from .validation import build_held_out_report
        return build_held_out_report(
            config=best_config,
            training_scenarios=list(training_scenarios or []),
            heldout_scenarios=list(scenarios),
            naturalness_targets=self.evaluator.naturalness_targets,
            epidemiology_targets=self.evaluator.epidemiology_targets,
            supported_rules=self.orchestrator.supported_constraints,
            unsupported_rules=self.orchestrator.unsupported_constraints,
        )

    def report(self) -> str:
        """Multi-line detailed report for debugging / logging."""
        ls = self.loaded_space
        o = self.orchestrator
        lines = [
            "=== Default Autoresearch Setup ===",
            f"Search space: {len(ls.space)} parameters",
            f"Constraints: {len(ls.constraints)} raw "
            f"(supported: {len(self.supported_rules)}, "
            f"unsupported: {len(self.unsupported_rules)})",
            f"Metadata keys: {sorted(ls.metadata.keys())}",
            f"Unsupported sections: {ls.unsupported_sections}",
            "",
            "Orchestrator:",
            f"  proposer_kind: {o.proposer_kind}",
            f"  n_starts: {o.n_starts}",
            f"  n_iterations: {o.n_iterations}",
            f"  seed: {o.seed}",
            f"  results_dir: {o.results_dir}",
            f"  constraint enforcement: "
            f"{'active' if o.supported_constraints else 'inactive'}",
            "",
            "Evaluator:",
            f"  n_classes: {self.evaluator.n_classes}",
            f"  max_turns: {self.evaluator.max_turns}",
            f"  n_students: {self.evaluator.n_students}",
            f"  naturalness targets: {len(self.evaluator.naturalness_targets)}",
            f"  epidemiology targets: {len(self.evaluator.epidemiology_targets)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_default_autoresearch_setup(
    *,
    # Orchestrator overrides
    n_iterations: int = 10,
    n_starts: int = 1,
    seed: int | None = 42,
    proposer_kind: str = "random",
    results_dir: Path | str | None = None,
    early_stop_patience: int | None = None,
    proposer_kwargs: dict | None = None,
    # Evaluator overrides
    n_classes: int = 3,
    max_turns: int = 200,
    n_students: int = 20,
    # YAML overrides (for tests or alternative runs)
    search_space_yaml: Path | str | None = None,
    naturalness_yaml: Path | str | None = None,
    epidemiology_yaml: Path | str | None = None,
    # Constraint enforcement
    enforce_constraints: bool = True,
) -> DefaultAutoresearchSetup:
    """Assemble a ready-to-run autoresearch setup from harness YAMLs.

    Defaults pull from:
      - `.harness/student_parameter_ranges.yaml` (search space)
      - `.harness/naturalness_targets.yaml` (Tier 1 targets)
      - `.harness/epidemiology_targets.yaml` (Tier 2 targets)

    All knobs are keyword-only so callers must opt in explicitly; this
    prevents accidental misconfiguration from positional argument order
    changes.

    Args:
        n_iterations: trials per start (default 10)
        n_starts: independent random starts for multi-start identifiability
        seed: master seed (each start derives its own from this)
        proposer_kind: "random" | "lhs" | "grid" | "bayes"
        results_dir: where to write `results.tsv` + checkpoint JSON.
                     Defaults to `.harness/` next to the YAMLs.
        early_stop_patience: stop a start early if this many iterations
                             pass without improvement
        proposer_kwargs: extra kwargs forwarded to `make_proposer`
        n_classes: classes per evaluation (simulation fanout)
        max_turns: cap on turns per class (200 reaches phase 2-3)
        n_students: students per class
        search_space_yaml: override harness search-space file path
        naturalness_yaml: override Tier 1 target file path
        epidemiology_yaml: override Tier 2 target file path

    Returns:
        DefaultAutoresearchSetup bundle with loaded_space, evaluator,
        orchestrator all wired together.
    """
    # Load search space (LoadedSearchSpace preserves constraints + metadata)
    if search_space_yaml is None:
        loaded_space = load_default_search_space()
    else:
        from .search_space_loader import load_search_space
        loaded_space = load_search_space(search_space_yaml)

    # Parse raw YAML constraint dicts into supported / unsupported rules.
    supported_rules, unsupported_rules = parse_constraints(
        loaded_space.constraints
    )

    # Build evaluator from target YAMLs
    evaluator = build_default_evaluator(
        n_classes=n_classes,
        max_turns=max_turns,
        seed=seed,
        naturalness_yaml=naturalness_yaml,
        epidemiology_yaml=epidemiology_yaml,
    )
    # Propagate n_students override (build_default_evaluator doesn't
    # take it directly, but DefaultEvaluator has the field)
    evaluator.n_students = n_students

    # Assemble orchestrator
    if results_dir is None:
        results_dir_path = Path(".harness")
    else:
        results_dir_path = Path(results_dir)

    # Only forward supported rules if enforcement is on.
    # Unsupported rules are always surfaced on the bundle for
    # transparency regardless of enforcement.
    orch_supported = supported_rules if enforce_constraints else []
    orch_unsupported = unsupported_rules if enforce_constraints else []

    orchestrator = AutoresearchOrchestrator(
        space=loaded_space.space,
        evaluator=evaluator,
        proposer_kind=proposer_kind,
        n_iterations=n_iterations,
        n_starts=n_starts,
        seed=seed,
        results_dir=results_dir_path,
        early_stop_patience=early_stop_patience,
        proposer_kwargs=proposer_kwargs,
        supported_constraints=orch_supported,
        unsupported_constraints=orch_unsupported,
    )

    return DefaultAutoresearchSetup(
        loaded_space=loaded_space,
        evaluator=evaluator,
        orchestrator=orchestrator,
        supported_rules=supported_rules,
        unsupported_rules=unsupported_rules,
    )
