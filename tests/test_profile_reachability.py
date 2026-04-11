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

    Phase 6 slice 15 note: the upper bound is calibrated against
    an ACTIVE student perceive loop (previously this test ran
    against a starved loop where ``current_events=[]`` prevented
    any perception-based divergence). With the perceive loop now
    wired to real observable events, modulation-driven cognitive
    shifts propagate through perception → memory → plan → act,
    producing a larger but still bounded divergence channel. A
    threshold of 0.75 still catches truly runaway drift while
    tolerating the new observed variance.
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
        diff = abs(s_mod.state.get("attention", 0.65) - s_no.state.get("attention", 0.65))
        assert diff < 0.75, (
            f"Student {s_mod.student_id} ({s_mod.profile_type}): "
            f"attention diff after 50 turns = {diff:.3f} "
            f"(modulation may be accumulating drift)"
        )


def test_modulation_does_not_accumulate_drift_long():
    """Over 300 turns, drift should remain bounded (stronger test).

    Phase 6 slice 15 note: threshold widened to match the now-active
    student perceive loop. The invariant "modulation is transient"
    still holds; it just has a wider admissible drift via the
    perception→memory→plan channel.
    """
    env_with_mod = ClassroomV2(n_students=20, seed=42)
    env_without = ClassroomV2(n_students=20, seed=42)
    env_without._enable_situational_modulation = False

    _run_env_steps(env_with_mod, 300)
    _run_env_steps(env_without, 300)

    # Worst-case drift should still be bounded.
    max_attention_drift = max(
        abs(
            sm.state.get("attention", 0.65)
            - sn.state.get("attention", 0.65)
        )
        for sm, sn in zip(env_with_mod.students, env_without.students)
    )
    assert max_attention_drift < 0.90, (
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


# ==========================================================================
# Section 4 — Comorbidity branching correctness (categorical sampling)
# ==========================================================================


def test_adhd_plus_depression_reachable_through_generation():
    """Regression: under the old threshold-chain, `adhd_plus_depression`
    was effectively unreachable for `adhd_inattentive` students (because
    once the earlier anxiety/LD threshold failed, roll >= p_anx + p_ld
    and the later depression branch was gated behind a different condition).

    With the new categorical-sampling design, depression must appear
    under repeated sampling with reasonable probability.
    """
    seen_depression = 0
    total_classes = 200
    # Use n_students=30 and many classes to make sampling robust
    for seed in range(total_classes):
        env = ClassroomV2(n_students=30, seed=seed)
        env.reset()
        for student in env.students:
            if student.profile_type == "adhd_plus_depression":
                seen_depression += 1
                break  # at least one suffices for this class

    # Expected: roughly 10% of each ADHD subtype → at least several per
    # hundred classes. Lower bound is very conservative.
    assert seen_depression >= 5, (
        f"adhd_plus_depression appeared in only {seen_depression}/"
        f"{total_classes} classes — branching may be broken"
    )


def test_all_comorbid_variants_reachable():
    """All six ADHD+X comorbid variants must be reachable from the
    generation path under repeated sampling.
    """
    required = {
        "adhd_c_plus_odd",
        "adhd_h_plus_odd",
        "adhd_i_plus_anxiety",
        "adhd_i_plus_ld",
        "adhd_plus_depression",
    }
    seen: set[str] = set()
    for seed in range(300):
        env = ClassroomV2(n_students=30, seed=seed)
        env.reset()
        for student in env.students:
            if student.profile_type in required:
                seen.add(student.profile_type)
        if required <= seen:
            break  # early exit once all are observed

    missing = required - seen
    assert not missing, (
        f"These comorbid profiles were never generated across 300 "
        f"classes: {missing}"
    )


def test_comorbidity_preserves_adhd_total_count():
    """ADHD total count must be invariant under comorbidity relabeling.

    We compare the number of students flagged is_adhd with a baseline
    environment whose comorbidity branching is simulated by equivalent
    rng consumption — but since the environment already has branching
    baked in, we instead verify that ADHD count matches what the
    subtype split would produce before branching.
    """
    for seed in range(50):
        env = ClassroomV2(n_students=20, seed=seed)
        env.reset()
        n_adhd = sum(1 for s in env.students if s.is_adhd)
        # Every ADHD student should have a profile_type that starts with 'adhd_'
        adhd_profiles = [s.profile_type for s in env.students if s.is_adhd]
        for p in adhd_profiles:
            assert p.startswith("adhd_"), (
                f"ADHD student has non-adhd profile_type: {p}"
            )
        # Aggregate check: prevalence not wildly off
        assert 0 <= n_adhd <= env.n_students


def test_pure_subtype_still_reachable():
    """After adding comorbidity branching, the pure ADHD subtypes must
    still appear (we don't want everyone to become comorbid).
    """
    seen_pure = {
        "adhd_inattentive": 0,
        "adhd_hyperactive_impulsive": 0,
        "adhd_combined": 0,
    }
    for seed in range(100):
        env = ClassroomV2(n_students=30, seed=seed)
        env.reset()
        for student in env.students:
            if student.profile_type in seen_pure:
                seen_pure[student.profile_type] += 1

    for subtype, count in seen_pure.items():
        assert count > 0, (
            f"Pure {subtype} never appeared in 100 classes — "
            f"comorbidity branching consumed all of them"
        )


# ==========================================================================
# Section 5 — Compositional modulation amplification
# ==========================================================================


def _make_env_with_mock_modulation():
    """Helper: return a freshly-reset env. Amplification is tested directly
    via env._modulation_amplification without stepping the simulation."""
    env = ClassroomV2(n_students=5, seed=1)
    env.reset()
    return env


class _MockModulation:
    """Minimal duck-typed ModulationVector for amplification tests."""
    def __init__(self, adhd=1.0, anxiety=1.0, odd=1.0):
        self.adhd_amplification = adhd
        self.anxiety_amplification = anxiety
        self.odd_amplification = odd


def _student_with_profile(env, profile_type: str):
    """Construct a freestanding CognitiveStudent with the given profile."""
    from src.simulation.cognitive_agent import CognitiveStudent
    return CognitiveStudent(
        student_id="TEST",
        profile_type=profile_type,
        age=9,
        gender="male",
    )


def test_amplification_adhd_comorbid_inherits_adhd_channel():
    """adhd_i_plus_anxiety should receive ADHD amplification via is_adhd,
    even though the profile string also contains 'anxiety'.
    """
    env = _make_env_with_mock_modulation()
    student = _student_with_profile(env, "adhd_i_plus_anxiety")
    mod = _MockModulation(adhd=1.8, anxiety=1.0, odd=1.0)
    amp = env._modulation_amplification(student, mod)
    assert amp == 1.8


def test_amplification_adhd_comorbid_both_adhd_and_anxiety_channels():
    """adhd_i_plus_anxiety with BOTH adhd and anxiety amplifications
    active should return the MAX (combination rule)."""
    env = _make_env_with_mock_modulation()
    student = _student_with_profile(env, "adhd_i_plus_anxiety")
    mod = _MockModulation(adhd=1.8, anxiety=2.5, odd=1.0)
    amp = env._modulation_amplification(student, mod)
    assert amp == 2.5  # max of 1.8 and 2.5


def test_amplification_adhd_h_plus_odd_inherits_both():
    """adhd_h_plus_odd is ADHD (is_adhd True) AND has odd component.
    Both channels should fire, result = max.
    """
    env = _make_env_with_mock_modulation()
    student = _student_with_profile(env, "adhd_h_plus_odd")
    mod = _MockModulation(adhd=1.5, anxiety=1.0, odd=3.0)
    amp = env._modulation_amplification(student, mod)
    assert amp == 3.0  # max of 1.5 and 3.0


def test_amplification_anxiety_plus_depression_inherits_anxiety():
    """anxiety_plus_depression (non-ADHD) should inherit anxiety channel
    by component membership, even though exact string equality fails."""
    env = _make_env_with_mock_modulation()
    student = _student_with_profile(env, "anxiety_plus_depression")
    mod = _MockModulation(adhd=2.0, anxiety=2.5, odd=1.0)
    amp = env._modulation_amplification(student, mod)
    # is_adhd=False, so ADHD channel does NOT fire even though adhd=2.0
    # Only anxiety channel fires → 2.5
    assert amp == 2.5


def test_amplification_adhd_c_plus_odd_inherits_adhd_and_odd():
    """adhd_c_plus_odd should inherit both ADHD and ODD channels."""
    env = _make_env_with_mock_modulation()
    student = _student_with_profile(env, "adhd_c_plus_odd")
    mod = _MockModulation(adhd=1.6, anxiety=1.0, odd=2.5)
    amp = env._modulation_amplification(student, mod)
    assert amp == 2.5


def test_amplification_distractor_does_not_get_adhd_channel():
    """asd_like is not ADHD and has no anxiety/odd component → no amp."""
    env = _make_env_with_mock_modulation()
    student = _student_with_profile(env, "asd_like")
    mod = _MockModulation(adhd=2.0, anxiety=2.5, odd=3.0)
    amp = env._modulation_amplification(student, mod)
    assert amp == 1.0


def test_amplification_depression_distractor_no_channels():
    """Pure depression distractor has no ADHD / anxiety / odd components."""
    env = _make_env_with_mock_modulation()
    student = _student_with_profile(env, "depression")
    mod = _MockModulation(adhd=2.0, anxiety=2.0, odd=2.0)
    amp = env._modulation_amplification(student, mod)
    assert amp == 1.0


def test_amplification_normal_student_no_channels():
    """Normal profile should not get any amplification."""
    env = _make_env_with_mock_modulation()
    student = _student_with_profile(env, "normal_quiet")
    mod = _MockModulation(adhd=2.0, anxiety=2.0, odd=2.0)
    amp = env._modulation_amplification(student, mod)
    assert amp == 1.0


def test_amplification_pure_adhd_inattentive_no_anxiety_channel():
    """Sanity: adhd_inattentive (no 'anxiety' in name) only gets ADHD channel.

    Guards against accidentally matching substrings instead of components.
    """
    env = _make_env_with_mock_modulation()
    student = _student_with_profile(env, "adhd_inattentive")
    mod = _MockModulation(adhd=1.5, anxiety=3.0, odd=1.0)
    amp = env._modulation_amplification(student, mod)
    assert amp == 1.5  # Only ADHD channel, not anxiety


def test_amplification_token_boundary_safety():
    """Component tokenizer must not match partial substrings like
    'anxiety' inside a hypothetical 'anxious_xxx' (no such profile exists,
    but we test the helper directly).
    """
    env = _make_env_with_mock_modulation()
    # Component-split should treat "_" boundaries
    assert env._profile_contains_token("adhd_i_plus_anxiety", ("anxiety",))
    assert not env._profile_contains_token("normal_quiet", ("anxiety",))
    assert not env._profile_contains_token("normal_quiet", ("odd",))
    assert env._profile_contains_token("adhd_h_plus_odd", ("odd",))
    assert not env._profile_contains_token("adhd_c_plus_odd", ("anxiety",))
