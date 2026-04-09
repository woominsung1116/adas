#!/usr/bin/env python3
"""Phase 4: Run 100-class experiments with 5 comparison groups + ablation.

Usage:
    .venv/bin/python scripts/run_experiment.py [--classes 100] [--students 20] [--seed 42] [--output results/experiment]

Comparison groups:
    1. no_memory:    Rule-based teacher, no memory at all
    2. case_base:    Case Base only (no Experience Base)
    3. exp_base:     Experience Base only (no Case Base)
    4. full:         Full memory (Case Base + Experience Base) — the main condition
    5. full_no_emotion: Full memory but no teacher emotional model

Each group runs N classes × 950 turns.
Results saved as JSON + CSV for analysis.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.simulation.orchestrator_v2 import OrchestratorV2, PhaseConfig
from src.simulation.teacher_memory import TeacherMemory


def run_group(name, n_classes, n_students, seed, memory_config, emotion_enabled=True):
    """Run one experimental group."""
    print(f"\n{'='*60}")
    print(f"Group: {name} ({n_classes} classes × {n_students} students × 950 turns)")
    print(f"{'='*60}")

    orch = OrchestratorV2(
        n_students=n_students,
        max_classes=n_classes,
        seed=seed,
        feedback_rate=0.30,
    )

    # Apply memory config
    if memory_config == "no_memory":
        orch.memory = TeacherMemory(retrieval_noise=1.0, seed=seed)  # 100% noise = useless
    elif memory_config == "case_base_only":
        # Disable experience base by setting impossibly high promotion threshold
        orch.memory = TeacherMemory(
            retrieval_noise=0.20,
            principle_promotion_threshold=99999,
            seed=seed,
        )
    elif memory_config == "exp_base_only":
        # Disable case base by setting 100% retrieval noise
        orch.memory = TeacherMemory(
            retrieval_noise=1.0,
            principle_promotion_threshold=7,
            principle_min_classes=3,
            seed=seed,
        )
    elif memory_config == "full":
        orch.memory = TeacherMemory(
            retrieval_noise=0.20,
            principle_promotion_threshold=7,
            principle_min_classes=3,
            memory_decay_rate=0.99,
            seed=seed,
        )

    # Disable teacher emotion if needed
    if not emotion_enabled:
        orch.teacher_emotions.patience = 1.0
        orch.teacher_emotions.empathy_capacity = 1.0
        orch.teacher_emotions.frustration = 0.0

    class_results = []
    t0 = time.time()

    for result in orch.run():
        m = result["result"]["metrics"]
        class_id = result["class_id"]

        row = {
            "group": name,
            "class_id": class_id,
            "n_students": m.n_students,
            "n_adhd": m.n_adhd,
            "n_identified": m.n_identified,
            "n_managed": m.n_managed,
            "tp": m.true_positives,
            "fp": m.false_positives,
            "fn": m.false_negatives,
            "tn": m.true_negatives,
            "adhd_tp": m.adhd_tp,
            "adhd_fp": m.adhd_fp,
            "adhd_fn": m.adhd_fn,
            "confounder_fp": m.confounder_fp,
            "sensitivity": orch.growth.sensitivity(class_id),
            "specificity": orch.growth.specificity(class_id),
            "ppv": orch.growth.ppv(class_id),
            "f1": orch.growth.f1(class_id),
            "completion_turn": m.class_completion_turn,
            "n_reports": len(result["result"]["reports"]),
            "strategies": list(m.strategies_used) if m.strategies_used else [],
        }
        class_results.append(row)

        if class_id % 10 == 0 or class_id <= 3:
            sens = row["sensitivity"]
            ppv = row["ppv"]
            print(f"  Class {class_id:3d}: TP={m.true_positives} FP={m.false_positives} "
                  f"FN={m.false_negatives} | sens={sens:.3f} ppv={ppv:.3f}")

    elapsed = time.time() - t0
    print(f"  Done: {elapsed:.1f}s ({elapsed/n_classes:.2f}s/class)")

    # Growth curve
    growth = {
        "group": name,
        "sensitivity_curve": [r["sensitivity"] for r in class_results],
        "ppv_curve": [r["ppv"] for r in class_results],
        "f1_curve": [r["f1"] for r in class_results],
        "cumulative_sensitivity": orch.growth.sensitivity(),
        "cumulative_ppv": orch.growth.ppv(),
        "cumulative_f1": orch.growth.f1(),
        "auprc": orch.growth.auprc(),
        "macro_f1": orch.growth.macro_f1(),
        "total_time_s": elapsed,
        "vs_benchmarks": orch.growth.vs_benchmarks(),
    }

    return class_results, growth


def main():
    parser = argparse.ArgumentParser(description="ADAS Phase 4 Experiment")
    parser.add_argument("--classes", type=int, default=100, help="Classes per group")
    parser.add_argument("--students", type=int, default=20, help="Students per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="results/experiment", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    groups = [
        ("no_memory", "no_memory", True),
        ("case_base_only", "case_base_only", True),
        ("exp_base_only", "exp_base_only", True),
        ("full", "full", True),
        ("full_no_emotion", "full", False),
    ]

    all_results = []
    all_growth = []

    total_t0 = time.time()

    for name, mem_config, emotion in groups:
        results, growth = run_group(
            name=name,
            n_classes=args.classes,
            n_students=args.students,
            seed=args.seed,
            memory_config=mem_config,
            emotion_enabled=emotion,
        )
        all_results.extend(results)
        all_growth.append(growth)

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_elapsed:.1f}s for {len(groups)} groups × {args.classes} classes")
    print(f"{'='*60}")

    # Save results
    results_path = os.path.join(args.output, "class_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Saved: {results_path} ({len(all_results)} rows)")

    growth_path = os.path.join(args.output, "growth_curves.json")
    with open(growth_path, "w", encoding="utf-8") as f:
        json.dump(all_growth, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved: {growth_path}")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Group':<20} {'Sens':>6} {'PPV':>6} {'F1':>6} {'AUPRC':>6} {'MF1':>6}")
    print("-" * 60)
    for g in all_growth:
        print(f"{g['group']:<20} "
              f"{g['cumulative_sensitivity']:>6.3f} "
              f"{g['cumulative_ppv']:>6.3f} "
              f"{g['cumulative_f1']:>6.3f} "
              f"{g['auprc']:>6.3f} "
              f"{g['macro_f1']:>6.3f}")


if __name__ == "__main__":
    main()
