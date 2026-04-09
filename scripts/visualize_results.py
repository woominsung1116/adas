#!/usr/bin/env python3
"""Phase 4-8: Visualize experiment results for paper.

Usage:
    .venv/bin/python scripts/visualize_results.py [--input results/experiment_100] [--output results/figures]

Generates:
    1. Growth curves (sensitivity/PPV/F1 over classes) for all 5 groups
    2. Ablation comparison bar chart
    3. Stratified analysis (by archetype, severity)
    4. AUPRC + Macro-F1 summary
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Skipping figure generation.")


def load_results(input_dir):
    class_path = os.path.join(input_dir, "class_results.json")
    growth_path = os.path.join(input_dir, "growth_curves.json")

    with open(class_path, "r", encoding="utf-8") as f:
        class_results = json.load(f)
    with open(growth_path, "r", encoding="utf-8") as f:
        growth_curves = json.load(f)

    return class_results, growth_curves


def plot_growth_curves(growth_curves, output_dir):
    """Fig 1: Growth curves for all groups."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("sensitivity_curve", "Sensitivity", axes[0]),
        ("ppv_curve", "PPV (Positive Predictive Value)", axes[1]),
        ("f1_curve", "F1 Score", axes[2]),
    ]

    colors = {
        "no_memory": "#ef4444",
        "case_base_only": "#f59e0b",
        "exp_base_only": "#8b5cf6",
        "full": "#3b82f6",
        "full_no_emotion": "#10b981",
    }

    labels = {
        "no_memory": "No Memory",
        "case_base_only": "Case Base Only",
        "exp_base_only": "Exp. Base Only",
        "full": "Full (CB+EB)",
        "full_no_emotion": "Full (No Emotion)",
    }

    for metric_key, metric_name, ax in metrics:
        for g in growth_curves:
            name = g["group"]
            curve = g.get(metric_key, [])
            if not curve:
                continue
            # Smoothing: rolling average of 5
            smoothed = []
            for i in range(len(curve)):
                window = curve[max(0, i - 4):i + 1]
                smoothed.append(sum(window) / len(window))

            ax.plot(
                range(1, len(smoothed) + 1),
                smoothed,
                color=colors.get(name, "#666"),
                label=labels.get(name, name),
                linewidth=2,
                alpha=0.8,
            )

        ax.set_xlabel("Class #")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, "growth_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_ablation_bars(growth_curves, output_dir):
    """Fig 2: Ablation comparison bar chart."""
    if not HAS_MPL:
        return

    groups = [g["group"] for g in growth_curves]
    metrics = {
        "Sensitivity": [g["cumulative_sensitivity"] for g in growth_curves],
        "PPV": [g["cumulative_ppv"] for g in growth_curves],
        "F1": [g["cumulative_f1"] for g in growth_curves],
        "AUPRC": [g["auprc"] for g in growth_curves],
        "Macro-F1": [g["macro_f1"] for g in growth_curves],
    }

    labels = {
        "no_memory": "No Mem",
        "case_base_only": "CB Only",
        "exp_base_only": "EB Only",
        "full": "Full",
        "full_no_emotion": "No Emo",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(groups))
    width = 0.15
    colors = ["#ef4444", "#f59e0b", "#3b82f6", "#8b5cf6", "#10b981"]

    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - 2) * width
        bars = ax.bar(
            [xi + offset for xi in x],
            values,
            width,
            label=metric_name,
            color=colors[i],
            alpha=0.8,
        )

    ax.set_xlabel("Experimental Group")
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Memory Components")
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(g, g) for g in groups])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, "ablation_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_class_detail(class_results, output_dir):
    """Fig 3: Per-class detail for 'full' group."""
    if not HAS_MPL:
        return

    full = [r for r in class_results if r["group"] == "full"]
    if not full:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # TP/FP/FN over classes
    ax = axes[0][0]
    classes = [r["class_id"] for r in full]
    ax.plot(classes, [r["tp"] for r in full], "g-", label="TP", alpha=0.7)
    ax.plot(classes, [r["fp"] for r in full], "r-", label="FP", alpha=0.7)
    ax.plot(classes, [r["fn"] for r in full], "b-", label="FN", alpha=0.7)
    ax.set_title("TP / FP / FN per Class")
    ax.set_xlabel("Class #")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Sensitivity + PPV
    ax = axes[0][1]
    ax.plot(classes, [r["sensitivity"] for r in full], "b-", label="Sensitivity", linewidth=2)
    ax.plot(classes, [r["ppv"] for r in full], "r-", label="PPV", linewidth=2)
    ax.set_title("Sensitivity & PPV")
    ax.set_xlabel("Class #")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reports generated
    ax = axes[1][0]
    ax.bar(classes, [r["n_reports"] for r in full], color="#3b82f6", alpha=0.7)
    ax.set_title("DSM-5 Reports per Class")
    ax.set_xlabel("Class #")
    ax.grid(True, alpha=0.3)

    # Confounder FP
    ax = axes[1][1]
    ax.bar(classes, [r.get("confounder_fp", 0) for r in full], color="#f59e0b", alpha=0.7)
    ax.set_title("Confounder False Positives")
    ax.set_xlabel("Class #")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Full Memory Group — Detailed Analysis", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "full_group_detail.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def print_summary_table(growth_curves):
    """Print formatted summary table."""
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Group':<20} {'Sens':>7} {'PPV':>7} {'F1':>7} {'AUPRC':>7} {'MF1':>7} {'Time':>8}")
    print("-" * 70)
    for g in growth_curves:
        print(f"{g['group']:<20} "
              f"{g['cumulative_sensitivity']:>7.3f} "
              f"{g['cumulative_ppv']:>7.3f} "
              f"{g['cumulative_f1']:>7.3f} "
              f"{g['auprc']:>7.3f} "
              f"{g['macro_f1']:>7.3f} "
              f"{g['total_time_s']:>7.1f}s")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Visualize ADAS experiment results")
    parser.add_argument("--input", default="results/experiment_100", help="Input directory")
    parser.add_argument("--output", default="results/figures", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    class_results, growth_curves = load_results(args.input)
    print(f"Loaded: {len(class_results)} class results, {len(growth_curves)} groups")

    print_summary_table(growth_curves)

    plot_growth_curves(growth_curves, args.output)
    plot_ablation_bars(growth_curves, args.output)
    plot_class_detail(class_results, args.output)

    print(f"\nAll figures saved to {args.output}/")


if __name__ == "__main__":
    main()
