"""Autoresearch orchestrator — the main calibration loop.

Drives the search for student parameters that minimize 3-tier calibration
loss (naturalness + epidemiology + sparsity). Supports:

  - Pluggable proposer (random / LHS / grid / bayes)
  - Multi-start runs (N independent restarts for identifiability)
  - Checkpoint / resume (`.harness/checkpoints/`)
  - Per-iteration logging to `.harness/results.tsv`
  - Sensitivity analysis (perturbed best config)
  - Pluggable parameter applier (how to inject configs into simulator)

This module does NOT hold loop-specific simulation code. Instead it uses:
  - `evaluator.evaluate_config(config) -> (bundle, loss_result)`

so the evaluator decides how to apply a config and run the simulator.
See `applier.py` for the default applier.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

from .loss import LossResult
from .proposer import ProposerBase, SearchSpace, Trial, make_proposer


# ---------------------------------------------------------------------------
# Evaluator protocol
# ---------------------------------------------------------------------------


class EvaluatorProtocol:
    """Anything that can turn a config dict into a loss scalar.

    Concrete implementations live in `applier.py`.
    """

    def evaluate(
        self,
        config: dict[str, Any],
    ) -> tuple[Any, LossResult]:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Orchestrator result types
# ---------------------------------------------------------------------------


@dataclass
class RunState:
    """Single autoresearch run state (one start)."""

    start_id: int
    seed: int
    iterations: int
    best_loss: float = math.inf
    best_config: dict[str, Any] = field(default_factory=dict)
    history: list[Trial] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_id": self.start_id,
            "seed": self.seed,
            "iterations": self.iterations,
            "best_loss": self.best_loss,
            "best_config": self.best_config,
            "elapsed_seconds": self.elapsed_seconds,
            "n_trials": len(self.history),
        }


@dataclass
class OrchestratorResult:
    """Aggregate result across all restarts."""

    runs: list[RunState] = field(default_factory=list)
    global_best_loss: float = math.inf
    global_best_config: dict[str, Any] = field(default_factory=dict)
    global_best_start_id: int = -1

    def summary(self) -> str:
        lines = [
            "=== Autoresearch Orchestrator Summary ===",
            f"Number of starts: {len(self.runs)}",
            f"Global best loss: {self.global_best_loss:.6f} "
            f"(start_id={self.global_best_start_id})",
            "",
            "Per-start best loss:",
        ]
        for run in self.runs:
            mark = "★" if run.start_id == self.global_best_start_id else " "
            lines.append(
                f"  {mark} start {run.start_id}: best={run.best_loss:.6f} "
                f"iter={run.iterations} ({run.elapsed_seconds:.1f}s)"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


class AutoresearchOrchestrator:
    """Main calibration loop with multi-start + checkpointing.

    Usage:
        from src.calibration.orchestrator import AutoresearchOrchestrator
        from src.calibration.proposer import SearchSpace, ParameterSpec
        from src.calibration.applier import build_default_evaluator

        space = SearchSpace([
            ParameterSpec("adhd_inattentive.frustration", 0.12, 0.25, default=0.19),
            ...
        ])
        evaluator = build_default_evaluator(n_classes=3, max_turns=200)
        orch = AutoresearchOrchestrator(
            space=space,
            evaluator=evaluator,
            proposer_kind="random",
            n_iterations=20,
            n_starts=3,
            seed=42,
            results_dir=Path(".harness"),
        )
        result = orch.run()
        print(result.summary())
    """

    def __init__(
        self,
        space: SearchSpace,
        evaluator: EvaluatorProtocol,
        proposer_kind: str = "random",
        n_iterations: int = 10,
        n_starts: int = 1,
        seed: int | None = 42,
        results_dir: Path | None = None,
        log_filename: str = "results.tsv",
        checkpoint_filename: str = "orchestrator_checkpoint.json",
        early_stop_patience: int | None = None,
        proposer_kwargs: dict | None = None,
    ) -> None:
        self.space = space
        self.evaluator = evaluator
        self.proposer_kind = proposer_kind
        self.n_iterations = n_iterations
        self.n_starts = n_starts
        self.seed = seed
        self.early_stop_patience = early_stop_patience
        self.proposer_kwargs = proposer_kwargs or {}

        if results_dir is None:
            results_dir = Path(".harness")
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.results_dir / log_filename
        self.checkpoint_path = self.results_dir / checkpoint_filename

        self._ensure_log_header()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    _LOG_COLUMNS = [
        "timestamp",
        "start_id",
        "iteration",
        "loss_total",
        "loss_naturalness",
        "loss_epidemiology",
        "loss_sparsity",
        "config_json",
    ]

    def _ensure_log_header(self) -> None:
        if self.log_path.exists():
            return
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(self._LOG_COLUMNS)

    def _log_trial(
        self,
        start_id: int,
        iteration: int,
        loss_result: LossResult,
        config: dict[str, Any],
    ) -> None:
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(
                [
                    f"{time.time():.3f}",
                    start_id,
                    iteration,
                    f"{loss_result.total:.6f}",
                    f"{loss_result.naturalness_loss:.6f}",
                    f"{loss_result.epidemiology_loss:.6f}",
                    f"{loss_result.sparsity_penalty:.6f}",
                    json.dumps(config, ensure_ascii=False, sort_keys=True),
                ]
            )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, result: OrchestratorResult, completed_runs: int) -> None:
        payload = {
            "completed_runs": completed_runs,
            "global_best_loss": result.global_best_loss,
            "global_best_config": result.global_best_config,
            "global_best_start_id": result.global_best_start_id,
            "runs": [r.to_dict() for r in result.runs],
            "n_iterations": self.n_iterations,
            "n_starts": self.n_starts,
            "proposer_kind": self.proposer_kind,
            "timestamp": time.time(),
        }
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def load_checkpoint(self) -> Optional[dict]:
        if not self.checkpoint_path.exists():
            return None
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_single_start(
        self, start_id: int, seed: int
    ) -> RunState:
        """Run one independent optimization from a fresh proposer state."""
        proposer: ProposerBase = make_proposer(
            self.proposer_kind, self.space, seed=seed, **self.proposer_kwargs
        )

        state = RunState(start_id=start_id, seed=seed, iterations=0)
        start_time = time.time()

        best_loss_seen = math.inf
        no_improve_count = 0

        for it in range(1, self.n_iterations + 1):
            state.iterations = it
            try:
                config = proposer.propose(state.history)
                config = self.space.clip_config(config)
                _, loss_result = self.evaluator.evaluate(config)
                loss_scalar = loss_result.total
            except Exception as exc:
                # Evaluator failures are logged with infinity so the proposer
                # learns to avoid this region. Does not crash the loop.
                loss_scalar = math.inf
                loss_result = LossResult(total=math.inf)
                config = config if "config" in dir() else {}

            trial = Trial(config=config, loss=loss_scalar, iteration=it)
            state.history.append(trial)

            self._log_trial(start_id, it, loss_result, config)

            # Track best
            if loss_scalar < state.best_loss:
                state.best_loss = loss_scalar
                state.best_config = dict(config)
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Early stopping
            if (
                self.early_stop_patience is not None
                and no_improve_count >= self.early_stop_patience
            ):
                break

        state.elapsed_seconds = time.time() - start_time
        return state

    def run(self, resume: bool = False) -> OrchestratorResult:
        """Execute n_starts independent runs and return aggregate result."""
        result = OrchestratorResult()

        start_from = 0
        if resume:
            ckpt = self.load_checkpoint()
            if ckpt:
                start_from = int(ckpt.get("completed_runs", 0))
                # Re-hydrate global best
                result.global_best_loss = float(ckpt.get("global_best_loss", math.inf))
                result.global_best_config = dict(ckpt.get("global_best_config") or {})
                result.global_best_start_id = int(ckpt.get("global_best_start_id", -1))

        for start_id in range(start_from, self.n_starts):
            start_seed = (self.seed or 0) + start_id * 1000
            state = self.run_single_start(start_id, start_seed)
            result.runs.append(state)

            if state.best_loss < result.global_best_loss:
                result.global_best_loss = state.best_loss
                result.global_best_config = dict(state.best_config)
                result.global_best_start_id = start_id

            self.save_checkpoint(result, completed_runs=start_id + 1)

        return result

    # ------------------------------------------------------------------
    # Sensitivity analysis (of the best config)
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        best_config: dict[str, Any],
        perturbation: float = 0.20,
        n_samples: int = 10,
        seed: int | None = 0,
    ) -> list[dict[str, Any]]:
        """Perturb each dim of best_config by ±perturbation and evaluate.

        Returns a list of {"dim": name, "delta": +/- value, "loss": float}.
        """
        import random
        rng = random.Random(seed)
        reports: list[dict[str, Any]] = []
        for spec in self.space.specs:
            if spec.name not in best_config or spec.kind == "choice":
                continue
            base = float(best_config[spec.name])
            span = spec.hi - spec.lo
            delta = span * perturbation
            for sign in (+1, -1):
                cfg = dict(best_config)
                cfg[spec.name] = spec.clip(base + sign * delta)
                cfg = self.space.clip_config(cfg)
                try:
                    _, loss = self.evaluator.evaluate(cfg)
                    lv = loss.total
                except Exception:
                    lv = math.inf
                reports.append({
                    "dim": spec.name,
                    "sign": int(sign),
                    "delta": sign * delta,
                    "loss": lv,
                })
        return reports
