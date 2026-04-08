#!/usr/bin/env python3
"""End-to-end test: run 2 classes with mock LLM backend."""

from __future__ import annotations

import sys
import os

# Ensure project root is on sys.path when run as a script
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.llm.mock_backend import MockTeacherBackend
from src.simulation.orchestrator import SimulationOrchestrator


def main() -> None:
    print("=" * 60)
    print("ADHD Classroom Simulation — End-to-End Test (Mock LLM)")
    print("=" * 60)

    # Step 1: Create mock backend
    mock_backend = MockTeacherBackend()
    print("[OK] MockTeacherBackend created")

    # Step 2: Create orchestrator with 10 students, fixed seed for reproducibility
    orch = SimulationOrchestrator(
        llm_backend=mock_backend,
        n_students=10,
        adhd_prevalence=0.20,   # 20% so we get ~2 ADHD students in 10
        max_classes=2,
        seed=42,
    )
    print("[OK] SimulationOrchestrator created (10 students, 2 classes, seed=42)")
    print()

    # Step 3: Run 2 classes, printing per-turn events
    class_num = 0
    for class_event in orch.run(max_turns_per_class=30):
        class_num += 1
        metrics = class_event["metrics"]

        print(f"--- Class {class_event['class_id']} Complete ---")
        print(f"  Students       : {metrics.n_students}")
        print(f"  ADHD (true)    : {metrics.n_adhd}")
        print(f"  Identified     : {metrics.n_identified}")
        print(f"  True positives : {metrics.true_positives}")
        print(f"  False positives: {metrics.false_positives}")
        print(f"  False negatives: {metrics.false_negatives}")
        print(f"  Avg ID turn    : {metrics.avg_identification_turn}")
        print(f"  Strategies used: {metrics.strategies_used}")
        print(f"  Completion turn: {metrics.class_completion_turn}")
        print()

    # Step 4: Print growth summary
    print("=" * 60)
    print("GROWTH SUMMARY")
    print("=" * 60)
    print(orch.growth.summary())
    print()

    # Step 5: Print evaluator metrics
    eval_metrics = orch.evaluator.compute_metrics()
    if eval_metrics:
        print("=" * 60)
        print("EVALUATOR METRICS")
        print("=" * 60)
        for k, v in eval_metrics.items():
            if v is not None:
                print(f"  {k:<28}: {v}")
    else:
        print("[INFO] No identification reports generated (0 ADHD flagged).")

    print()
    print("[PASS] End-to-end test completed successfully.")


if __name__ == "__main__":
    main()
