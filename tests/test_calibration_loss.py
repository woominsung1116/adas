"""Tests for calibration loss functions."""

from dataclasses import dataclass
from src.calibration import (
    NaturalnessTarget,
    EpidemiologyTarget,
    LossResult,
    compute_naturalness_loss,
    compute_epidemiology_loss,
    compute_combined_loss,
    compute_sparsity_penalty,
)
from src.calibration.loss import range_distance, point_distance


# ---------------------------------------------------------------------------
# Mock simulation result
# ---------------------------------------------------------------------------


@dataclass
class MockSimResult:
    adhd_prevalence: float = 0.07
    seat_leaving_ratio: float = 3.5
    teacher_accuracy: float = 0.70
    odd_comorbidity_rate: float = 0.40
    anxiety_comorbidity_rate: float = 0.28


# ---------------------------------------------------------------------------
# range_distance / point_distance
# ---------------------------------------------------------------------------


def test_range_distance_inside():
    assert range_distance(0.07, (0.05, 0.09)) == 0.0
    assert range_distance(0.05, (0.05, 0.09)) == 0.0
    assert range_distance(0.09, (0.05, 0.09)) == 0.0


def test_range_distance_below():
    # 0.03 is 0.02 below 0.05, range width 0.04 → 0.5
    assert abs(range_distance(0.03, (0.05, 0.09)) - 0.5) < 1e-9


def test_range_distance_above():
    assert abs(range_distance(0.11, (0.05, 0.09)) - 0.5) < 1e-9


def test_point_distance_within_tolerance():
    assert point_distance(0.70, 0.70, tolerance=0.05) == 0.0
    assert point_distance(0.72, 0.70, tolerance=0.05) == 0.0


def test_point_distance_exceeds_tolerance():
    # diff=0.10, tolerance=0.05 → exceeds by 0.05 → 0.05/0.70 ≈ 0.071
    d = point_distance(0.80, 0.70, tolerance=0.05)
    assert d > 0


# ---------------------------------------------------------------------------
# Naturalness loss
# ---------------------------------------------------------------------------


def test_naturalness_loss_perfect_match():
    targets = [
        NaturalnessTarget(
            name="seat_leaving_ratio",
            metric=lambda s: s.seat_leaving_ratio,
            target_range=(3.0, 4.0),
            weight=1.0,
            source="KCI ART002794420",
            pattern_description="ADHD seat-leaving rate ratio",
        )
    ]
    sim = MockSimResult(seat_leaving_ratio=3.5)
    loss, per = compute_naturalness_loss(sim, targets)
    assert loss == 0.0


def test_naturalness_loss_miss():
    targets = [
        NaturalnessTarget(
            name="seat_leaving_ratio",
            metric=lambda s: s.seat_leaving_ratio,
            target_range=(3.0, 4.0),
            weight=1.0,
            source="KCI ART002794420",
        )
    ]
    sim = MockSimResult(seat_leaving_ratio=5.0)
    loss, per = compute_naturalness_loss(sim, targets)
    assert loss > 0
    assert "seat_leaving_ratio" in per


def test_naturalness_loss_multiple_targets_weighted():
    targets = [
        NaturalnessTarget(
            name="a", metric=lambda s: 1.0,
            target_range=(0.0, 0.5), weight=1.0,
        ),
        NaturalnessTarget(
            name="b", metric=lambda s: 2.0,
            target_range=(1.0, 3.0), weight=2.0,
        ),
    ]
    sim = MockSimResult()
    loss, per = compute_naturalness_loss(sim, targets)
    # a: miss (1.0 > 0.5, distance 0.5 / 0.5 = 1.0), weight 1 → 1.0
    # b: hit (2.0 in [1,3]) → 0
    # total = (1.0 * 1 + 0 * 2) / (1 + 2) = 0.333
    assert abs(loss - (1.0 / 3.0)) < 1e-9


def test_naturalness_loss_metric_exception():
    """If metric raises, target gets max loss (1.0)."""
    def bad_metric(s):
        raise ValueError("test")
    targets = [
        NaturalnessTarget(
            name="bad", metric=bad_metric,
            target_range=(0, 1), weight=1.0,
        )
    ]
    sim = MockSimResult()
    loss, per = compute_naturalness_loss(sim, targets)
    assert loss == 1.0


# ---------------------------------------------------------------------------
# Epidemiology loss
# ---------------------------------------------------------------------------


def test_epidemiology_loss_prevalence_in_range():
    targets = [
        EpidemiologyTarget(
            name="adhd_prevalence",
            metric=lambda s: s.adhd_prevalence,
            target_range=(0.05, 0.09),
            weight=1.0,
            source="JKMS 2017",
            study_sample="Korean community",
        )
    ]
    sim = MockSimResult(adhd_prevalence=0.07)
    loss, per = compute_epidemiology_loss(sim, targets)
    assert loss == 0.0


def test_epidemiology_loss_prevalence_out_of_range():
    targets = [
        EpidemiologyTarget(
            name="adhd_prevalence",
            metric=lambda s: s.adhd_prevalence,
            target_range=(0.05, 0.09),
            weight=1.0,
        )
    ]
    sim = MockSimResult(adhd_prevalence=0.20)
    loss, per = compute_epidemiology_loss(sim, targets)
    assert loss > 0


# ---------------------------------------------------------------------------
# Sparsity penalty
# ---------------------------------------------------------------------------


def test_sparsity_penalty_empty():
    assert compute_sparsity_penalty({}) == 0.0


def test_sparsity_penalty_all_zero():
    terms = {"a": 0.0, "b": 0.0}
    assert compute_sparsity_penalty(terms) == 0.0


def test_sparsity_penalty_l1():
    terms = {"a": 0.1, "b": -0.2, "c": 0.3}
    # L1 = 0.6, default weight 0.05 → 0.03
    result = compute_sparsity_penalty(terms, sparsity_weight=0.05)
    assert abs(result - 0.03) < 1e-9


def test_sparsity_penalty_custom_weight():
    terms = {"a": 0.1}
    result = compute_sparsity_penalty(terms, sparsity_weight=0.5)
    assert abs(result - 0.05) < 1e-9


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------


def test_combined_loss_perfect():
    nat = [
        NaturalnessTarget(
            name="seat", metric=lambda s: s.seat_leaving_ratio,
            target_range=(3.0, 4.0), weight=1.0,
        )
    ]
    epi = [
        EpidemiologyTarget(
            name="prev", metric=lambda s: s.adhd_prevalence,
            target_range=(0.05, 0.09), weight=1.0,
        )
    ]
    sim = MockSimResult(seat_leaving_ratio=3.5, adhd_prevalence=0.07)
    result = compute_combined_loss(sim, nat, epi)
    assert result.total == 0.0
    assert result.naturalness_loss == 0.0
    assert result.epidemiology_loss == 0.0


def test_combined_loss_mixed():
    nat = [
        NaturalnessTarget(
            name="seat", metric=lambda s: s.seat_leaving_ratio,
            target_range=(3.0, 4.0), weight=1.0,
        )
    ]
    epi = [
        EpidemiologyTarget(
            name="prev", metric=lambda s: s.adhd_prevalence,
            target_range=(0.05, 0.09), weight=1.0,
        )
    ]
    sim = MockSimResult(seat_leaving_ratio=3.5, adhd_prevalence=0.20)
    result = compute_combined_loss(sim, nat, epi)
    # naturalness = 0, epidemiology > 0
    assert result.naturalness_loss == 0.0
    assert result.epidemiology_loss > 0
    # Weighted: 0.6*0 + 0.4*epi > 0
    assert result.total > 0


def test_combined_loss_with_sparsity():
    nat = []
    epi = []
    interactions = {"a": 0.1, "b": -0.2}
    result = compute_combined_loss(
        MockSimResult(), nat, epi,
        interaction_terms=interactions,
        sparsity_weight=0.1,
    )
    # 0.3 (L1) * 0.1 = 0.03
    assert abs(result.sparsity_penalty - 0.03) < 1e-9
    assert abs(result.total - 0.03) < 1e-9


def test_loss_result_summary_string():
    result = LossResult(
        total=0.15,
        naturalness_loss=0.10,
        epidemiology_loss=0.05,
        sparsity_penalty=0.0,
        per_target={"a": 0.10, "b": 0.05},
    )
    summary = result.summary()
    assert "0.1500" in summary
    assert "Naturalness" in summary


def test_calibration_target_boundary_exact():
    """Values exactly at boundary should have zero loss."""
    target = NaturalnessTarget(
        name="test", metric=lambda s: 0.5,
        target_range=(0.3, 0.5), weight=1.0,
    )
    assert target.loss(0.3) == 0.0
    assert target.loss(0.5) == 0.0
    assert target.loss(0.4) == 0.0
