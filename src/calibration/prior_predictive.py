"""Prior predictive check — do literature default parameters land in target ranges?

Runs the real simulator with its current default (literature-seeded) parameters
and checks whether the resulting metrics fall inside the YAML-defined target
ranges. This is the first sanity check before autoresearch calibration begins:
if priors already produce plausible outputs, the search space is well-scoped;
if most targets miss, the defaults or ranges need adjustment before tuning.

Usage:
    from src.calibration.prior_predictive import run_prior_predictive_check

    report = run_prior_predictive_check(n_classes=5, max_turns=200, seed=42)
    print(report.summary())
    for target_name, status in report.per_target.items():
        print(f"  {target_name}: {status}")

Does NOT implement parameter search — that's handled by `orchestrator.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .loader import default_harness_paths, load_targets
from .loss import compute_combined_loss, LossResult
from .metrics import CalibrationResultBundle


@dataclass
class PriorPredictiveReport:
    """Result of a prior predictive check run."""

    n_classes: int
    max_turns: int
    seed: int | None
    bundle: CalibrationResultBundle
    loss_result: LossResult
    per_target: dict[str, dict] = field(default_factory=dict)
    # per_target[name] = {"in_range": bool, "measured": float, "range": (lo, hi)}

    def n_in_range(self) -> int:
        return sum(1 for v in self.per_target.values() if v["in_range"])

    def n_total(self) -> int:
        return len(self.per_target)

    def coverage(self) -> float:
        """Fraction of targets whose measured value fell inside the prior range."""
        total = self.n_total()
        return self.n_in_range() / total if total > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"=== Prior Predictive Check ===",
            f"N classes: {self.n_classes}, max_turns: {self.max_turns}, seed: {self.seed}",
            f"Coverage: {self.n_in_range()}/{self.n_total()} "
            f"({self.coverage() * 100:.1f}%) targets in prior range",
            f"",
            self.loss_result.summary(),
            f"",
            f"Per-target status:",
        ]
        for name, v in sorted(self.per_target.items()):
            mark = "✓" if v["in_range"] else "✗"
            m = v["measured"]
            lo, hi = v["range"]
            m_str = f"{m:.4f}" if m is not None else "N/A"
            lines.append(f"  {mark} {name:50s} {m_str} ∈ [{lo}, {hi}]")
        return "\n".join(lines)


def _evaluate_targets_on_bundle(
    bundle: CalibrationResultBundle,
    naturalness_targets: list,
    epidemiology_targets: list,
) -> dict[str, dict]:
    """Measure each target on the bundle and classify as in/out of range."""
    per_target: dict[str, dict] = {}
    for target in list(naturalness_targets) + list(epidemiology_targets):
        try:
            measured = target.metric(bundle)
        except Exception as exc:
            per_target[target.name] = {
                "in_range": False,
                "measured": None,
                "range": target.target_range,
                "error": str(exc),
            }
            continue
        lo, hi = target.target_range
        per_target[target.name] = {
            "in_range": (lo <= measured <= hi),
            "measured": measured,
            "range": target.target_range,
        }
    return per_target


def run_prior_predictive_check(
    n_classes: int = 5,
    max_turns: int = 200,
    seed: int | None = 42,
    naturalness_yaml: Any = None,
    epidemiology_yaml: Any = None,
    retrieval_noise_config: Any = None,
) -> PriorPredictiveReport:
    """Run a prior predictive check with literature default parameters.

    Args:
        n_classes: number of classes to aggregate for the bundle
        max_turns: cap on turns per class (200 reaches phase 2-3, enough for
                   most metrics without running the full 950-turn cycle)
        seed: deterministic seed (defaults to 42)
        naturalness_yaml / epidemiology_yaml: override YAML paths; if None,
                   uses `.harness/naturalness_targets.yaml` and
                   `.harness/epidemiology_targets.yaml`.
        retrieval_noise_config: Phase 6 slice 11 optional
                   ``RetrievalNoiseConfig`` forwarded into every
                   orchestrator constructed by this helper. When
                   ``None`` (default), the simulator runs with the
                   legacy ``retrieval_noise`` scalar path — bit-
                   identical to the pre-slice baseline.

    Returns:
        PriorPredictiveReport with coverage stats + per-target breakdown.
    """
    # Import locally to avoid circular imports at module load time
    from src.simulation.orchestrator_v2 import OrchestratorV2
    from src.calibration.adapters import run_real_bundle

    nat_path = naturalness_yaml
    epi_path = epidemiology_yaml
    if nat_path is None or epi_path is None:
        default_nat, default_epi = default_harness_paths()
        nat_path = nat_path or default_nat
        epi_path = epi_path or default_epi

    nat_targets, epi_targets = load_targets(
        nat_path, epi_path, skip_unsupported=True
    )

    # Build the bundle from real simulation runs.
    # Use fresh orchestrators per class to avoid state leakage.
    bundle = CalibrationResultBundle(histories=[])
    for class_idx in range(n_classes):
        class_seed = (seed + class_idx) if seed is not None else None
        orch = OrchestratorV2(
            n_students=20,
            max_classes=1,
            seed=class_seed,
            retrieval_noise_config=retrieval_noise_config,
        )
        orch.classroom.MAX_TURNS = max_turns
        sub = run_real_bundle(orch, n_classes=1)
        bundle.histories.extend(sub.histories)

    # Compute combined loss (for diagnostic summary)
    loss_result = compute_combined_loss(bundle, nat_targets, epi_targets)

    # Per-target in/out status
    per_target = _evaluate_targets_on_bundle(bundle, nat_targets, epi_targets)

    return PriorPredictiveReport(
        n_classes=n_classes,
        max_turns=max_turns,
        seed=seed,
        bundle=bundle,
        loss_result=loss_result,
        per_target=per_target,
    )
