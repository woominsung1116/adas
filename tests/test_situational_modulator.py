"""Tests for SituationalModulator."""

from src.simulation.situational_modulator import (
    ModulationVector,
    SituationalModulator,
    SituationalEvent,
    academic_cycle_modulation,
    diurnal_rhythm,
    new_semester_adaptation,
    peer_conflict_event,
    substitute_teacher,
    presentation_event,
    default_korean_k6_schedule,
)


# ------------------------------------------------------------------
# Individual modulator tests
# ------------------------------------------------------------------


def test_modulation_vector_combine_additive():
    a = ModulationVector(global_anxiety=0.1, global_attention=-0.05)
    b = ModulationVector(global_anxiety=0.05, global_excitement=0.03)
    c = a.combine(b)
    assert abs(c.global_anxiety - 0.15) < 1e-9
    assert abs(c.global_attention - (-0.05)) < 1e-9
    assert abs(c.global_excitement - 0.03) < 1e-9


def test_modulation_vector_amplification_multiplicative():
    a = ModulationVector(adhd_amplification=1.5)
    b = ModulationVector(adhd_amplification=2.0)
    c = a.combine(b)
    assert abs(c.adhd_amplification - 3.0) < 1e-9


def test_modulation_vector_class_stress_max():
    a = ModulationVector(class_stress=0.3)
    b = ModulationVector(class_stress=0.7)
    c = a.combine(b)
    assert c.class_stress == 0.7  # max, not sum


def test_modulation_vector_flags_or():
    a = ModulationVector(exam_week=True)
    b = ModulationVector(peer_conflict=True)
    c = a.combine(b)
    assert c.exam_week and c.peer_conflict


# ------------------------------------------------------------------
# Academic cycle tests
# ------------------------------------------------------------------


def test_academic_cycle_baseline_mid_semester():
    """Turn 200 (mid 1st semester) = no exam, no stress."""
    v = academic_cycle_modulation(200)
    assert v.global_anxiety == 0.0
    assert not v.exam_week


def test_academic_cycle_midterm_exam_stress():
    """Turn 350 (1st midterm week) = exam stress active."""
    v = academic_cycle_modulation(350)
    assert v.exam_week
    assert v.global_anxiety > 0
    assert v.adhd_amplification > 1.0


def test_academic_cycle_final_exam_stress():
    """Turn 850 (finals prep) = peak stress."""
    v = academic_cycle_modulation(850)
    assert v.exam_week
    assert v.class_stress >= 0.5


def test_academic_cycle_fatigue_period():
    """Turn 450 (late 1st semester) = fatigue, lower excitement."""
    v = academic_cycle_modulation(450)
    assert v.global_excitement < 0
    assert not v.exam_week


# ------------------------------------------------------------------
# New semester adaptation tests
# ------------------------------------------------------------------


def test_new_semester_peaks_at_start():
    """Turn 1 = peak anxiety + excitement."""
    v = new_semester_adaptation(1)
    assert v.global_anxiety > 0.10
    assert v.global_excitement > 0


def test_new_semester_decays_to_zero():
    """Turn 100+ = back to baseline."""
    v = new_semester_adaptation(100)
    assert abs(v.global_anxiety) < 1e-9
    v2 = new_semester_adaptation(200)
    assert abs(v2.global_anxiety) < 1e-9


def test_new_semester_linear_decay():
    """Decay should be roughly linear over turns 1-100."""
    v50 = new_semester_adaptation(50)
    v_start = new_semester_adaptation(1)
    # At turn 50, should be ~half of turn 1
    ratio = v50.global_anxiety / v_start.global_anxiety
    assert 0.4 < ratio < 0.6


# ------------------------------------------------------------------
# Diurnal rhythm tests
# ------------------------------------------------------------------


def test_diurnal_morning_baseline():
    """Periods 1-2 = no modulation (peak cognition)."""
    assert diurnal_rhythm(1).global_attention == 0.0
    assert diurnal_rhythm(2).global_attention == 0.0


def test_diurnal_late_morning_slight_dip():
    """Periods 3-4 = small attention decrement."""
    v = diurnal_rhythm(4)
    assert v.global_attention < 0
    assert v.global_attention > -0.05


def test_diurnal_post_lunch_dip():
    """Period 5 = post-lunch dip (Folkard 1975)."""
    v = diurnal_rhythm(5)
    assert v.post_lunch_dip
    assert v.global_attention <= -0.06


def test_diurnal_afternoon_declining():
    """Period 6+ = sustained afternoon decline."""
    v = diurnal_rhythm(6)
    assert v.global_attention < 0
    assert v.global_excitement < 0


# ------------------------------------------------------------------
# Event modulators
# ------------------------------------------------------------------


def test_peer_conflict_event_affects_class():
    v = peer_conflict_event(active=True, severity=1.0)
    assert v.peer_conflict
    assert v.global_anxiety > 0
    assert v.class_disruption > 0


def test_peer_conflict_inactive_zero():
    v = peer_conflict_event(active=False)
    assert not v.peer_conflict
    assert v.global_anxiety == 0.0


def test_substitute_teacher_amplifies_odd():
    v = substitute_teacher(active=True)
    assert v.substitute_teacher
    assert v.odd_amplification > 2.0
    assert v.global_compliance < 0


def test_presentation_amplifies_anxiety():
    v = presentation_event(active=True)
    assert v.presentation_active
    assert v.anxiety_amplification > 2.0


# ------------------------------------------------------------------
# SituationalModulator integration
# ------------------------------------------------------------------


def test_modulator_compose_basic():
    modulator = SituationalModulator()
    mod = modulator.compute_modulation(turn=1, period=1)
    # Should have new-semester effect
    assert mod.global_anxiety > 0


def test_modulator_add_event_active_in_range():
    modulator = SituationalModulator()
    modulator.add_event(SituationalEvent("peer_conflict", start_turn=100, duration=20, severity=1.0))
    # Inside range
    mod = modulator.compute_modulation(turn=110, period=1)
    assert mod.peer_conflict
    # Outside range
    mod2 = modulator.compute_modulation(turn=130, period=1)
    assert not mod2.peer_conflict


def test_modulator_multiple_events_overlap():
    modulator = SituationalModulator()
    modulator.add_event(SituationalEvent("peer_conflict", start_turn=100, duration=20))
    modulator.add_event(SituationalEvent("substitute_teacher", start_turn=105, duration=5))
    mod = modulator.compute_modulation(turn=107, period=1)
    assert mod.peer_conflict
    assert mod.substitute_teacher


def test_modulator_diurnal_and_academic_combine():
    modulator = SituationalModulator()
    # Exam week + post-lunch period
    mod = modulator.compute_modulation(turn=350, period=5)
    assert mod.exam_week
    assert mod.post_lunch_dip
    # Combined attention effect should be worse than either alone
    exam_only = modulator.compute_modulation(turn=350, period=1)
    assert mod.global_attention < exam_only.global_attention


def test_default_korean_schedule_has_events():
    modulator = default_korean_k6_schedule()
    assert len(modulator.scheduled_events) > 0
    # Should have peer conflicts
    types = [e.event_type for e in modulator.scheduled_events]
    assert "peer_conflict" in types
    assert "substitute_teacher" in types
    assert "presentation" in types


def test_default_schedule_runs_full_year():
    modulator = default_korean_k6_schedule(total_turns=950)
    # Sample several turns, ensure no crashes
    for turn in [1, 100, 200, 350, 500, 700, 850, 950]:
        for period in range(1, 7):
            mod = modulator.compute_modulation(turn=turn, period=period)
            # Just verify it returns a valid ModulationVector
            assert isinstance(mod, ModulationVector)
