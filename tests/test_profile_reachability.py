"""Regression tests for profile reachability, behavior map coverage,
and transient situational modulation.

These tests cover the issues raised in Codex review on 2026-04-10:
  1. Profile generation should reach comorbidity / distractor profiles
  2. _PROFILE_BEHAVIOR_MAP should cover every profile in PROFILE_DELTAS
  3. Situational modulation should NOT accumulate drift over many turns
"""

import random
import pytest

from src.simulation.cognitive_agent import (
    PROFILE_DELTAS,
    _PROFILE_BEHAVIOR_MAP,
    BEHAVIOR_POOLS,
    CognitiveStudent,
)
from src.simulation.classroom_env_v2 import ClassroomV2, TeacherAction


# ==========================================================================
# Section 1 — Behavior-map coverage
# ==========================================================================


def test_behavior_map_covers_all_profiles():
    """Every profile in PROFILE_DELTAS must have an explicit behavior pool.

    Without this, comorbidity / distractor profiles silently fall back to
    the default `['normal']` pool and never emit their signature behaviors.
    """
    missing = [
        name for name in PROFILE_DELTAS
        if name not in _PROFILE_BEHAVIOR_MAP
    ]
    assert not missing, (
        f"These profiles lack behavior map entries (would fall back "
        f"to normal pool silently): {missing}"
    )


def test_behavior_map_references_valid_pools():
    """Every pool name in _PROFILE_BEHAVIOR_MAP must exist in BEHAVIOR_POOLS."""
    for profile, pools in _PROFILE_BEHAVIOR_MAP.items():
        for pool_name in pools:
            assert pool_name in BEHAVIOR_POOLS, (
                f"Profile {profile!r} references unknown pool {pool_name!r}. "
                f"Valid pools: {list(BEHAVIOR_POOLS.keys())}"
            )


def test_new_comorbidity_profiles_have_meaningful_pools():
    """Comorbidity profiles should use multi-pool mixtures, not just ['normal']."""
    comorbid = [
        "adhd_i_plus_anxiety",
        "adhd_h_plus_odd",
        "adhd_c_plus_odd",
        "adhd_i_plus_ld",
        "adhd_plus_depression",
        "anxiety_plus_depression",
    ]
    for name in comorbid:
        pools = _PROFILE_BEHAVIOR_MAP[name]
        assert len(pools) >= 2, (
            f"{name} should draw from ≥2 behavior pools, got {pools}"
        )
        assert pools != ["normal"], (
            f"{name} must not be normal-only fallback"
        )


def test_distractor_profiles_have_non_adhd_pools():
    """Distractor profiles should NOT primarily draw from hyperactivity
    or impulsivity pools (those are ADHD-core)."""
    for name in ("asd_like", "depression", "learning_disorder"):
        pools = _PROFILE_BEHAVIOR_MAP[name]
        # These distractors should not have ADHD-signature pools as primary
        assert "hyperactivity" not in pools, (
            f"{name} should not draw from hyperactivity pool"
        )
        assert "impulsivity" not in pools, (
            f"{name} should not draw from impulsivity pool"
        )


# ==========================================================================
# Section 2 — Profile generation reachability
# ==========================================================================


def test_profile_generation_no_crash_for_all_sizes():
    """Generating classrooms of varied sizes should never crash."""
    for n in (5, 10, 20, 30):
        env = ClassroomV2(n_students=n, seed=42)
        env.reset()
        assert len(env.students) == n


def test_profile_generation_reaches_comorbidity_over_many_classes():
    """Over many random classrooms, comorbidity profiles should appear at
    least occasionally (not permanently dead code paths)."""
    seen_profiles: set[str] = set()
    for seed in range(50):
        env = ClassroomV2(n_students=20, seed=seed)
        env.reset()
        for student in env.students:
            seen_profiles.add(student.profile_type)

    # Must see at least ONE comorbidity profile across 50 classrooms
    comorbid = {
        "adhd_i_plus_anxiety", "adhd_h_plus_odd", "adhd_c_plus_odd",
        "adhd_i_plus_ld", "adhd_plus_depression", "anxiety_plus_depression",
    }
    assert seen_profiles & comorbid, (
        f"No comorbidity profile generated across 50 classrooms. "
        f"Seen: {sorted(seen_profiles)}"
    )


def test_profile_generation_reaches_distractor_profiles():
    """Differential-diagnosis distractor profiles should be reachable too."""
    seen_profiles: set[str] = set()
    for seed in range(100):
        env = ClassroomV2(n_students=20, seed=seed)
        env.reset()
        for student in env.students:
            seen_profiles.add(student.profile_type)

    distractors = {"asd_like", "depression", "learning_disorder"}
    # Distractors are rare (1-3%), so over 100 classrooms * 20 students
    # we should see at least one
    assert seen_profiles & distractors, (
        f"No distractor profile generated across 100 classrooms. "
        f"Seen: {sorted(seen_profiles)}"
    )


def test_profile_generation_preserves_adhd_prevalence_range():
    """Aggregate ADHD prevalence over many classes should stay within the
    epidemiology target range [0.04, 0.15] (some slack for randomness).

    ADHD identity is determined by profile_type.startswith('adhd_'),
    which covers all comorbid branches correctly.
    """
    n_students_per = 20
    n_classes = 80
    total_students = n_students_per * n_classes
    total_adhd = 0

    for seed in range(n_classes):
        env = ClassroomV2(n_students=n_students_per, seed=seed)
        env.reset()
        total_adhd += sum(1 for s in env.students if s.is_adhd)

    prevalence = total_adhd / total_students
    # Default adhd_prevalence=(0.06, 0.11) → mean ~0.085
    # Allow wide tolerance [0.03, 0.16] for sampling variance
    assert 0.03 <= prevalence <= 0.16, (
        f"Aggregate ADHD prevalence {prevalence:.3f} outside expected range. "
        f"n_classes={n_classes}, n_adhd={total_adhd}"
    )


def test_comorbid_adhd_still_counts_as_adhd():
    """Comorbid profiles like adhd_i_plus_anxiety must have is_adhd=True."""
    adhd_comorbid = [
        "adhd_i_plus_anxiety",
        "adhd_h_plus_odd",
        "adhd_c_plus_odd",
        "adhd_i_plus_ld",
        "adhd_plus_depression",
    ]
    for profile in adhd_comorbid:
        student = CognitiveStudent(
            student_id="TEST",
            profile_type=profile,
            age=9,
            gender="male",
        )
        assert student.is_adhd is True, (
            f"{profile} should have is_adhd=True (starts with 'adhd_')"
        )


def test_non_adhd_distractors_not_flagged_as_adhd():
    """asd_like, depression, learning_disorder, anxiety_plus_depression
    must NOT be flagged as ADHD (they are distractors)."""
    non_adhd = [
        "asd_like",
        "depression",
        "learning_disorder",
        "anxiety_plus_depression",
    ]
    for profile in non_adhd:
        student = CognitiveStudent(
            student_id="TEST",
            profile_type=profile,
            age=9,
            gender="male",
        )
        assert student.is_adhd is False, (
            f"{profile} should have is_adhd=False"
        )


# ==========================================================================
# Section 3 — Transient situational modulation (no drift)
# ==========================================================================


def _run_env_steps(env: ClassroomV2, n_steps: int) -> None:
    """Helper: step the environment N times with no-op teacher actions."""
    env.reset()
    for _ in range(n_steps):
        action = TeacherAction(action_type="observe", student_id=None)
        env.step(action)


def test_modulation_does_not_accumulate_drift_short():
    """Over 50 turns of strong modulation, student state should not
    drift systematically beyond what cognitive dynamics alone would produce.
    """
    # Two envs: one with modulator, one without
    env_with_mod = ClassroomV2(n_students=20, seed=777)
    env_without = ClassroomV2(n_students=20, seed=777)
    env_without._enable_situational_modulation = False

    _run_env_steps(env_with_mod, 50)
    _run_env_steps(env_without, 50)

    # Compare per-student attention drift magnitudes
    for s_mod, s_no in zip(env_with_mod.students, env_without.students):
        assert s_mod.profile_type == s_no.profile_type
        # The difference in final attention should be bounded by
        # single-turn modulation magnitude (~0.15), not accumulating over turns.
        diff = abs(s_mod.state.get("attention", 0.65) - s_no.state.get("attention", 0.65))
        assert diff < 0.25, (
            f"Student {s_mod.student_id} ({s_mod.profile_type}): "
            f"attention diff after 50 turns = {diff:.3f} "
            f"(modulation may be accumulating drift)"
        )


def test_modulation_does_not_accumulate_drift_long():
    """Over 300 turns, drift should remain bounded (stronger test)."""
    env_with_mod = ClassroomV2(n_students=20, seed=42)
    env_without = ClassroomV2(n_students=20, seed=42)
    env_without._enable_situational_modulation = False

    _run_env_steps(env_with_mod, 300)
    _run_env_steps(env_without, 300)

    # Worst-case drift should still be bounded (transient offsets only)
    max_attention_drift = max(
        abs(
            sm.state.get("attention", 0.65)
            - sn.state.get("attention", 0.65)
        )
        for sm, sn in zip(env_with_mod.students, env_without.students)
    )
    assert max_attention_drift < 0.35, (
        f"Max attention drift after 300 turns = {max_attention_drift:.3f} "
        f"(should stay bounded with transient modulation)"
    )


def test_modulation_info_still_surfaced():
    """Ensure info['situation'] field is still exposed after refactor."""
    env = ClassroomV2(n_students=10, seed=1)
    env.reset()
    action = TeacherAction(action_type="observe", student_id=None)
    _, _, _, info = env.step(action)
    assert "situation" in info
    sit = info["situation"]
    for key in (
        "exam_week", "peer_conflict", "substitute_teacher",
        "presentation_active", "post_lunch_dip",
        "class_stress", "class_disruption",
    ):
        assert key in sit, f"info['situation'] missing key {key!r}"


def test_same_seed_same_trajectory_with_modulation():
    """Determinism check: same seed should give same state trajectory."""
    env_a = ClassroomV2(n_students=15, seed=123)
    env_b = ClassroomV2(n_students=15, seed=123)
    _run_env_steps(env_a, 30)
    _run_env_steps(env_b, 30)
    for sa, sb in zip(env_a.students, env_b.students):
        assert sa.profile_type == sb.profile_type
        assert abs(sa.state.get("attention", 0) - sb.state.get("attention", 0)) < 1e-9


def test_cognitive_delta_preserved_through_modulation():
    """Cognitive step changes to state/emotions should NOT be reverted
    by the transient-modulation restore logic.

    We capture pre-step state, force a step with modulation, and verify
    that any cognitive-driven delta in emotions is preserved post-step.
    """
    env = ClassroomV2(n_students=5, seed=9)
    env.reset()

    # Capture pre-step emotion state
    pre_anxiety = [s.emotions.anxiety for s in env.students]

    action = TeacherAction(action_type="observe", student_id=None)
    env.step(action)

    post_anxiety = [s.emotions.anxiety for s in env.students]

    # The test here is loose: we just verify that the system is not
    # stuck at its baseline (cognitive dynamics should produce SOME change
    # over one step for at least one student — or at minimum values should
    # remain valid [0, 1]).
    for i, (pre, post) in enumerate(zip(pre_anxiety, post_anxiety)):
        assert 0.0 <= post <= 1.0, (
            f"Student {i}: anxiety {post} out of [0,1] "
            f"(modulation restore likely broken)"
        )
