"""Tests for calibration metric extraction + YAML loader bridge.

Covers:
  - Metric registry resolution (known + unknown)
  - ClassHistory / CalibrationResultBundle contract
  - Prevalence metrics on synthetic bundles
  - Teacher identification timing metrics from event logs
  - Behavior frequency metrics
  - Seat-leaving ratio
  - Patience end-of-day-vs-start ratio
  - Empathic intervention gain
  - YAML loader end-to-end on real .harness files
  - Combined loss running end-to-end on a mini bundle
"""

from pathlib import Path

import pytest

from src.calibration import (
    ClassHistory,
    CalibrationResultBundle,
    METRIC_REGISTRY,
    resolve_metric,
    supported_metrics,
    load_naturalness_targets,
    load_epidemiology_targets,
    load_targets,
    default_harness_paths,
    compute_combined_loss,
    NaturalnessTarget,
    EpidemiologyTarget,
)


# ==========================================================================
# Section 1 — Metric registry
# ==========================================================================


def test_metric_registry_contains_required_epidemiology_metrics():
    required = {
        "prevalence.adhd_total",
        "prevalence.adhd_inattentive",
        "prevalence.adhd_hyperactive",
        "prevalence.adhd_combined",
        "prevalence.adhd_male_to_female_ratio",
        "teacher.identification_fraction_by_turn[475]",
        "teacher.identification_fraction_by_turn[950]",
        "behavior.normal_on_task_percentage",
        "behavior.adhd_on_task_relative_to_normal",
    }
    assert required.issubset(set(METRIC_REGISTRY.keys()))


def test_metric_registry_contains_required_naturalness_metrics():
    required = {
        "behavior.seat_leaving_adhd_normal_ratio",
        "teacher.first_suspicion_turn_median",
        "teacher.patience_end_of_day_vs_start_ratio",
        "intervention.empathic_compliance_gain",
    }
    assert required.issubset(set(METRIC_REGISTRY.keys()))


def test_resolve_metric_known():
    fn = resolve_metric("prevalence.adhd_total")
    assert callable(fn)


def test_resolve_metric_unknown_raises_not_implemented():
    with pytest.raises(NotImplementedError) as exc_info:
        resolve_metric("prevalence.definitely_not_a_real_metric")
    # Error message should include list of supported metrics
    assert "prevalence.adhd_total" in str(exc_info.value)


def test_supported_metrics_returns_sorted_list():
    metrics = supported_metrics()
    assert metrics == sorted(metrics)
    assert len(metrics) >= 13  # at least the required subset


# ==========================================================================
# Section 2 — Synthetic bundle builders
# ==========================================================================


def _make_student(sid, profile, is_adhd, gender="male"):
    return {
        "id": sid,
        "profile_type": profile,
        "gender": gender,
        "is_adhd": is_adhd,
    }


def _make_event(turn, students, teacher_action=None):
    return {
        "turn": turn,
        "students": students,
        "teacher_action": teacher_action or {},
    }


def _make_observation_student(sid, behaviors):
    """Minimal student dict for events (id + behaviors)."""
    return {"id": sid, "behaviors": behaviors}


# ==========================================================================
# Section 3 — Prevalence metrics
# ==========================================================================


def test_prevalence_adhd_total_simple():
    roster = [
        _make_student("S01", "normal_quiet", is_adhd=False),
        _make_student("S02", "normal_quiet", is_adhd=False),
        _make_student("S03", "adhd_inattentive", is_adhd=True),
        _make_student("S04", "adhd_combined", is_adhd=True),
    ]
    bundle = CalibrationResultBundle(
        histories=[ClassHistory(students_roster=roster)]
    )
    fn = resolve_metric("prevalence.adhd_total")
    assert fn(bundle) == 0.5


def test_prevalence_subtype_includes_comorbidities():
    """adhd_i_plus_anxiety should count toward adhd_inattentive prevalence."""
    roster = [
        _make_student("S01", "adhd_inattentive", is_adhd=True),
        _make_student("S02", "adhd_i_plus_anxiety", is_adhd=True),
        _make_student("S03", "adhd_i_plus_ld", is_adhd=True),
        _make_student("S04", "normal_quiet", is_adhd=False),
    ]
    bundle = CalibrationResultBundle(
        histories=[ClassHistory(students_roster=roster)]
    )
    fn = resolve_metric("prevalence.adhd_inattentive")
    # 3 of 4 match
    assert fn(bundle) == 0.75


def test_prevalence_adhd_hyperactive_includes_comorbidities():
    roster = [
        _make_student("S01", "adhd_hyperactive_impulsive", is_adhd=True),
        _make_student("S02", "adhd_h_plus_odd", is_adhd=True),
        _make_student("S03", "normal_quiet", is_adhd=False),
    ]
    bundle = CalibrationResultBundle(
        histories=[ClassHistory(students_roster=roster)]
    )
    fn = resolve_metric("prevalence.adhd_hyperactive")
    assert abs(fn(bundle) - 2 / 3) < 1e-9


def test_prevalence_male_female_ratio():
    roster = [
        _make_student("S01", "adhd_inattentive", is_adhd=True, gender="male"),
        _make_student("S02", "adhd_inattentive", is_adhd=True, gender="male"),
        _make_student("S03", "adhd_combined", is_adhd=True, gender="female"),
        _make_student("S04", "normal_quiet", is_adhd=False, gender="female"),
    ]
    bundle = CalibrationResultBundle(
        histories=[ClassHistory(students_roster=roster)]
    )
    fn = resolve_metric("prevalence.adhd_male_to_female_ratio")
    assert fn(bundle) == 2.0


def test_prevalence_empty_bundle_raises():
    bundle = CalibrationResultBundle(histories=[])
    fn = resolve_metric("prevalence.adhd_total")
    with pytest.raises(ValueError):
        fn(bundle)


# ==========================================================================
# Section 4 — Teacher identification timing
# ==========================================================================


def test_identification_fraction_by_turn_475():
    roster = [
        _make_student("S01", "adhd_inattentive", is_adhd=True),
        _make_student("S02", "adhd_combined", is_adhd=True),
        _make_student("S03", "normal_quiet", is_adhd=False),
    ]
    events = [
        _make_event(100, [], teacher_action={"action_type": "observe", "student_id": "S01"}),
        _make_event(
            300,
            [],
            teacher_action={"action_type": "identify_adhd", "student_id": "S01"},
        ),
        _make_event(
            500,
            [],
            teacher_action={"action_type": "identify_adhd", "student_id": "S02"},
        ),
    ]
    bundle = CalibrationResultBundle(
        histories=[ClassHistory(students_roster=roster, events=events)]
    )
    fn = resolve_metric("teacher.identification_fraction_by_turn[475]")
    # Only S01 was identified by turn 475
    assert fn(bundle) == 0.5


def test_identification_fraction_by_turn_950_catches_all():
    roster = [
        _make_student("S01", "adhd_inattentive", is_adhd=True),
        _make_student("S02", "adhd_combined", is_adhd=True),
    ]
    events = [
        _make_event(300, [], teacher_action={"action_type": "identify_adhd", "student_id": "S01"}),
        _make_event(900, [], teacher_action={"action_type": "identify_adhd", "student_id": "S02"}),
    ]
    bundle = CalibrationResultBundle(
        histories=[ClassHistory(students_roster=roster, events=events)]
    )
    fn = resolve_metric("teacher.identification_fraction_by_turn[950]")
    assert fn(bundle) == 1.0


def test_first_suspicion_turn_median():
    roster = [
        _make_student("S01", "adhd_inattentive", is_adhd=True),
        _make_student("S02", "adhd_combined", is_adhd=True),
    ]
    # Class 1: first identify at turn 100
    events1 = [
        _make_event(100, [], teacher_action={"action_type": "identify_adhd", "student_id": "S01"}),
    ]
    # Class 2: first identify at turn 200
    events2 = [
        _make_event(200, [], teacher_action={"action_type": "identify_adhd", "student_id": "S01"}),
    ]
    bundle = CalibrationResultBundle(
        histories=[
            ClassHistory(students_roster=roster, events=events1),
            ClassHistory(students_roster=roster, events=events2),
        ]
    )
    fn = resolve_metric("teacher.first_suspicion_turn_median")
    assert fn(bundle) == 150.0  # median of [100, 200]


# ==========================================================================
# Section 5 — Behavior frequency metrics
# ==========================================================================


def test_normal_on_task_percentage_simple():
    roster = [
        _make_student("S01", "normal_quiet", is_adhd=False),
        _make_student("S02", "adhd_inattentive", is_adhd=True),
    ]
    events = [
        _make_event(1, [
            _make_observation_student("S01", ["on_task"]),
            _make_observation_student("S02", ["off_task"]),
        ]),
        _make_event(2, [
            _make_observation_student("S01", ["listening"]),
            _make_observation_student("S02", ["daydreaming"]),
        ]),
        _make_event(3, [
            _make_observation_student("S01", ["daydreaming"]),  # normal off-task once
            _make_observation_student("S02", ["fidgeting"]),
        ]),
    ]
    bundle = CalibrationResultBundle(
        histories=[ClassHistory(students_roster=roster, events=events)]
    )
    fn = resolve_metric("behavior.normal_on_task_percentage")
    # Normal student: 2/3 on-task
    assert abs(fn(bundle) - 2 / 3) < 1e-9


def test_adhd_on_task_relative_to_normal():
    roster = [
        _make_student("S01", "normal_quiet", is_adhd=False),
        _make_student("S02", "adhd_inattentive", is_adhd=True),
    ]
    events = [
        _make_event(1, [
            _make_observation_student("S01", ["on_task"]),
            _make_observation_student("S02", ["on_task"]),
        ]),
        _make_event(2, [
            _make_observation_student("S01", ["on_task"]),
            _make_observation_student("S02", ["fidgeting"]),
        ]),
    ]
    bundle = CalibrationResultBundle(
        histories=[ClassHistory(students_roster=roster, events=events)]
    )
    fn = resolve_metric("behavior.adhd_on_task_relative_to_normal")
    # normal: 2/2 = 1.0, adhd: 1/2 = 0.5, ratio 0.5
    assert fn(bundle) == 0.5


def test_seat_leaving_ratio():
    roster = [
        _make_student("S01", "normal_quiet", is_adhd=False),
        _make_student("S02", "adhd_hyperactive_impulsive", is_adhd=True),
    ]
    events = [
        _make_event(1, [
            _make_observation_student("S01", ["on_task"]),
            _make_observation_student("S02", ["seat_leaving"]),
        ]),
        _make_event(2, [
            _make_observation_student("S01", ["on_task"]),
            _make_observation_student("S02", ["seat_leaving"]),
        ]),
        _make_event(3, [
            _make_observation_student("S01", ["seat_leaving"]),  # normal rarely
            _make_observation_student("S02", ["fidgeting"]),
        ]),
        _make_event(4, [
            _make_observation_student("S01", ["on_task"]),
            _make_observation_student("S02", ["seat_leaving"]),
        ]),
    ]
    bundle = CalibrationResultBundle(
        histories=[ClassHistory(students_roster=roster, events=events)]
    )
    fn = resolve_metric("behavior.seat_leaving_adhd_normal_ratio")
    # normal: 1/4, adhd: 3/4, ratio = 3.0
    assert fn(bundle) == 3.0


# ==========================================================================
# Section 6 — Patience + intervention
# ==========================================================================


def test_patience_end_vs_start_ratio():
    # 2 days of 5 periods each = 10 values
    # Day 1: 0.80 → 0.70 (ratio 0.875)
    # Day 2: 0.80 → 0.60 (ratio 0.75)
    log = [0.80, 0.78, 0.75, 0.72, 0.70, 0.80, 0.76, 0.70, 0.65, 0.60]
    bundle = CalibrationResultBundle(
        histories=[ClassHistory(teacher_patience_log=log)]
    )
    fn = resolve_metric("teacher.patience_end_of_day_vs_start_ratio")
    expected = (0.70 / 0.80 + 0.60 / 0.80) / 2
    assert abs(fn(bundle) - expected) < 1e-9


def test_patience_no_log_raises():
    bundle = CalibrationResultBundle(histories=[ClassHistory()])
    fn = resolve_metric("teacher.patience_end_of_day_vs_start_ratio")
    with pytest.raises(ValueError):
        fn(bundle)


def test_empathic_compliance_gain():
    outcomes = [
        {
            "strategy": "empathic_intervention",
            "pre_compliance": 0.5,
            "post_compliance": 0.7,
            "student_id": "S01",
        },
        {
            "strategy": "public_correction",
            "pre_compliance": 0.5,
            "post_compliance": 0.6,
            "student_id": "S02",
        },
        {
            "strategy": "empathic_praise",
            "pre_compliance": 0.4,
            "post_compliance": 0.6,
            "student_id": "S03",
        },
    ]
    bundle = CalibrationResultBundle(
        histories=[ClassHistory(intervention_outcomes=outcomes)]
    )
    fn = resolve_metric("intervention.empathic_compliance_gain")
    # Only 2 "empath*" strategies, gains 0.2 and 0.2, mean = 0.2
    assert abs(fn(bundle) - 0.2) < 1e-9


# ==========================================================================
# Section 7 — YAML loader
# ==========================================================================


def test_yaml_loader_loads_naturalness_targets():
    nat_path, _ = default_harness_paths()
    if not nat_path.exists():
        pytest.skip("naturalness_targets.yaml not present")
    targets = load_naturalness_targets(nat_path, skip_unsupported=True)
    # At least our minimum supported set should load
    assert len(targets) >= 4
    names = {t.name for t in targets}
    # Must include the ones we implemented metric extractors for
    assert "adhd_seat_leaving_ratio" in names
    assert "teacher_first_suspicion_turn" in names


def test_yaml_loader_loads_epidemiology_targets():
    _, epi_path = default_harness_paths()
    if not epi_path.exists():
        pytest.skip("epidemiology_targets.yaml not present")
    targets = load_epidemiology_targets(epi_path, skip_unsupported=True)
    assert len(targets) >= 9
    names = {t.name for t in targets}
    assert "adhd_total_prevalence" in names
    assert "adhd_male_female_ratio" in names


def test_yaml_loader_strict_raises_on_unknown_metric(tmp_path):
    """Strict mode (skip_unsupported=False) should raise NotImplementedError."""
    fake = tmp_path / "fake_targets.yaml"
    fake.write_text(
        """
patterns:
  - name: bogus
    metric: nonexistent.fake_metric
    range: [0.0, 1.0]
    weight: 1.0
""",
        encoding="utf-8",
    )
    with pytest.raises(NotImplementedError):
        load_naturalness_targets(fake, skip_unsupported=False)


def test_yaml_loader_skip_unsupported_returns_valid_subset(tmp_path):
    """Skip mode should silently drop unsupported entries."""
    fake = tmp_path / "mixed_targets.yaml"
    fake.write_text(
        """
patterns:
  - name: bogus
    metric: nonexistent.fake_metric
    range: [0.0, 1.0]
    weight: 1.0
  - name: ok
    metric: behavior.seat_leaving_adhd_normal_ratio
    range: [2.5, 5.5]
    weight: 1.0
""",
        encoding="utf-8",
    )
    targets = load_naturalness_targets(fake, skip_unsupported=True)
    assert len(targets) == 1
    assert targets[0].name == "ok"


def test_yaml_loader_both_files():
    nat_path, epi_path = default_harness_paths()
    if not nat_path.exists() or not epi_path.exists():
        pytest.skip("harness files not present")
    nat, epi = load_targets(nat_path, epi_path, skip_unsupported=True)
    assert len(nat) >= 4
    assert len(epi) >= 5


# ==========================================================================
# Section 8 — End-to-end combined loss
# ==========================================================================


def test_combined_loss_end_to_end_on_mini_bundle():
    """Build a tiny bundle, resolve two targets from YAML, compute loss."""
    roster = [
        _make_student("S01", "normal_quiet", is_adhd=False, gender="male"),
        _make_student("S02", "normal_quiet", is_adhd=False, gender="female"),
        _make_student("S03", "adhd_inattentive", is_adhd=True, gender="male"),
    ]
    bundle = CalibrationResultBundle(
        histories=[ClassHistory(students_roster=roster)]
    )

    # One epi target in-range, one out-of-range
    in_range = EpidemiologyTarget(
        name="prev_in_range",
        metric=resolve_metric("prevalence.adhd_total"),
        target_range=(0.2, 0.5),  # 1/3 = 0.333 is inside
        weight=1.0,
    )
    out_of_range = EpidemiologyTarget(
        name="prev_out_of_range",
        metric=resolve_metric("prevalence.adhd_total"),
        target_range=(0.5, 0.9),  # 0.333 is outside
        weight=1.0,
    )

    result = compute_combined_loss(
        bundle,
        naturalness_targets=[],
        epidemiology_targets=[in_range, out_of_range],
    )
    assert result.naturalness_loss == 0.0
    assert result.epidemiology_loss > 0.0
    # Half the targets hit, half miss → average > 0 but not 1
    assert 0.0 < result.total < 1.0


def test_combined_loss_from_real_yaml():
    """Load real YAML files and run compute_combined_loss on a bundle
    with enough data to satisfy the loaded metric subset."""
    nat_path, epi_path = default_harness_paths()
    if not nat_path.exists() or not epi_path.exists():
        pytest.skip("harness files not present")

    nat, epi = load_targets(nat_path, epi_path, skip_unsupported=True)

    # Minimal bundle: just prevalence metrics
    roster = [
        _make_student("S01", "normal_quiet", is_adhd=False, gender="male"),
        _make_student("S02", "adhd_inattentive", is_adhd=True, gender="male"),
        _make_student("S03", "adhd_combined", is_adhd=True, gender="female"),
    ]
    bundle = CalibrationResultBundle(
        histories=[ClassHistory(students_roster=roster)]
    )

    # Only include targets that don't require events/logs we don't have
    epi_safe = [
        t for t in epi
        if t.name in {
            "adhd_total_prevalence",
            "adhd_male_female_ratio",
            "adhd_inattentive_prevalence",
            "adhd_hyperactive_prevalence",
            "adhd_combined_prevalence",
        }
    ]

    result = compute_combined_loss(bundle, [], epi_safe)
    # Should produce a valid scalar (exact value depends on target ranges)
    assert result.total >= 0.0
    assert result.summary()  # string rendering works
