"""Adapters from real simulation outputs to calibration data structures.

Bridges `OrchestratorV2.run_class()` / `stream_class()` results into
`ClassHistory` and `CalibrationResultBundle` objects consumed by the
metric extractors in `metrics.py`.

Usage:
    from src.simulation.orchestrator_v2 import OrchestratorV2
    from src.calibration import run_real_bundle

    orch = OrchestratorV2(n_students=20, seed=42)
    bundle = run_real_bundle(orch, n_classes=3)

    from src.calibration import compute_combined_loss, load_targets
    nat, epi = load_targets(...)
    loss = compute_combined_loss(bundle, nat, epi)
"""

from __future__ import annotations

from typing import Any

from .metrics import ClassHistory, CalibrationResultBundle


def _student_roster_from_classroom(classroom) -> list[dict]:
    """Extract {id, profile_type, gender, is_adhd} roster from a classroom.

    Called once per class (after reset/stream completes) — the classroom's
    `students` attribute holds the current class's cognitive student objects.
    """
    roster = []
    for s in classroom.students:
        roster.append({
            "id": s.student_id,
            "profile_type": s.profile_type,
            "gender": getattr(s, "gender", "unknown"),
            "is_adhd": bool(s.is_adhd),
        })
    return roster


def class_result_to_history(
    class_result: dict,
    classroom,
) -> ClassHistory:
    """Convert one OrchestratorV2 class result into a ClassHistory.

    Args:
        class_result: dict returned by `OrchestratorV2.run_class()` containing
            keys {"metrics", "events", "reports", "teacher_patience_log",
            "intervention_outcomes", "first_suspicion_turns"}.
        classroom: the ClassroomV2 instance whose `students` attribute
            holds the student roster for this class.

    Returns:
        A ClassHistory ready for metric extraction.
    """
    return ClassHistory(
        metrics=class_result.get("metrics"),
        events=list(class_result.get("events") or []),
        students_roster=_student_roster_from_classroom(classroom),
        reports=list(class_result.get("reports") or []),
        teacher_patience_log=list(class_result.get("teacher_patience_log") or []) or None,
        intervention_outcomes=list(class_result.get("intervention_outcomes") or []),
        # Propagate first-suspicion turn map so metrics can prefer real
        # suspicion timing over identify_adhd fallback.
        first_suspicion_turns=dict(class_result.get("first_suspicion_turns") or {}),
    )


def run_real_bundle(
    orchestrator,
    n_classes: int = 1,
) -> CalibrationResultBundle:
    """Run N real classes through the orchestrator and build a bundle.

    This drives the same execution path as live simulation:
      orchestrator.run_class() → class_result dict → ClassHistory
    N histories are aggregated into a CalibrationResultBundle.

    Args:
        orchestrator: an OrchestratorV2 instance (or compatible)
        n_classes: number of classes to run (>=1)

    Returns:
        CalibrationResultBundle with `histories` list of length n_classes.
    """
    if n_classes < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}")

    histories: list[ClassHistory] = []
    for _ in range(n_classes):
        result = orchestrator.run_class()
        # run_class() increments class_count internally when called from
        # stream_class(), but the batch wrapper does not — increment here
        # so subsequent calls yield a fresh class_id.
        history = class_result_to_history(result, orchestrator.classroom)
        histories.append(history)
        orchestrator.class_count += 1

    return CalibrationResultBundle(histories=histories)
