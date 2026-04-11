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

    Access patterns:
        setup = build_default_autoresearch_setup(n_iterations=10)
        setup.orchestrator.run()
        print(setup.loaded_space.constraints)
        print(setup.loaded_space.metadata)
        print(setup.loaded_space.unsupported_sections)
    """

    loaded_space: LoadedSearchSpace
    evaluator: DefaultEvaluator
    orchestrator: AutoresearchOrchestrator

    def summary(self) -> str:
        """One-line human-readable summary of the setup."""
        ls = self.loaded_space
        o = self.orchestrator
        return (
            f"DefaultAutoresearchSetup("
            f"n_params={len(ls.space)}, "
            f"n_constraints={len(ls.constraints)}, "
            f"unsupported_sections={len(ls.unsupported_sections)}, "
            f"proposer={o.proposer_kind}, "
            f"n_starts={o.n_starts}, "
            f"n_iterations={o.n_iterations}, "
            f"results_dir={o.results_dir})"
        )

    def report(self) -> str:
        """Multi-line detailed report for debugging / logging."""
        ls = self.loaded_space
        o = self.orchestrator
        lines = [
            "=== Default Autoresearch Setup ===",
            f"Search space: {len(ls.space)} parameters",
            f"Constraints: {len(ls.constraints)} (parsed, not enforced)",
            f"Metadata keys: {sorted(ls.metadata.keys())}",
            f"Unsupported sections: {ls.unsupported_sections}",
            "",
            "Orchestrator:",
            f"  proposer_kind: {o.proposer_kind}",
            f"  n_starts: {o.n_starts}",
            f"  n_iterations: {o.n_iterations}",
            f"  seed: {o.seed}",
            f"  results_dir: {o.results_dir}",
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
    )

    return DefaultAutoresearchSetup(
        loaded_space=loaded_space,
        evaluator=evaluator,
        orchestrator=orchestrator,
    )
