"""Tests for Phase 6 slice 19: observable-only StudentSummary.

Covers:
  - `_make_observation` no longer derives `profile_hint` from
    latent `escalation_risk` / `attention` reads
  - `_visible_behaviors` no longer emits latent fallback sentinels
    (`seems_inattentive`, `on_task`, `quiet`)
  - `StudentSummary.profile_hint` responds only to observable
    behavior + public `is_identified` flag
  - Vocabulary for the hint locked to
    {identified_adhd, disruptive, unknown}
  - Downstream reader compatibility:
    `build_observations_from_classroom` still consumes the field
    and produces sensible values
  - Fixed-seed deterministic runs
  - Full suite regression
"""

import inspect

import pytest

from src.simulation.classroom_env_v2 import (
    ClassroomV2,
    TeacherAction,
    _STUDENT_VISIBLE_DISRUPTIVE_BEHAVIORS,
)


# ---------------------------------------------------------------------------
# Static guards
# ---------------------------------------------------------------------------


def test_make_observation_no_longer_reads_latent_for_profile_hint():
    """The old block branched on `student.state.get("escalation_risk")`
    and `student.state.get("attention")`. Those reads must be
    gone from `_make_observation` now that the hint comes from
    `_derive_observation_profile_hint`."""
    src = inspect.getsource(ClassroomV2._make_observation)
    for latent in (
        "escalation_risk",
        "attention",
        "distress_level",
    ):
        assert latent not in src, (
            f"_make_observation still reads latent field {latent!r}"
        )
    assert "_derive_observation_profile_hint" in src


def test_derive_observation_profile_hint_is_latent_free():
    src = inspect.getsource(ClassroomV2._derive_observation_profile_hint)
    for latent in (
        "escalation_risk",
        "attention",
        "distress_level",
        "compliance",
        ".state.get(",
        "state_snapshot",
    ):
        assert latent not in src
    assert "_STUDENT_VISIBLE_DISRUPTIVE_BEHAVIORS" in src


def test_visible_behaviors_no_longer_emits_latent_fallback():
    """`_visible_behaviors` must not contain the legacy
    sentinel strings — the helper now returns `[]` when no
    high-visibility behavior is present."""
    src = inspect.getsource(ClassroomV2._visible_behaviors)
    for sentinel in ("seems_inattentive", "on_task", "quiet"):
        assert sentinel not in src, (
            f"_visible_behaviors still emits {sentinel!r}"
        )
    # No latent state reads inside the helper either.
    for latent in ("escalation_risk", "attention", "distress_level", "compliance"):
        assert latent not in src


# ---------------------------------------------------------------------------
# Behavior / invariants
# ---------------------------------------------------------------------------


def _clear_behaviors(env: ClassroomV2) -> None:
    for s in env.students:
        s.exhibited_behaviors = []
        s.identified = False


def test_visible_behaviors_empty_under_no_visible_disruption():
    env = ClassroomV2(n_students=5, seed=1)
    env.reset()
    _clear_behaviors(env)
    for s in env.students:
        assert env._visible_behaviors(s) == []


def test_visible_behaviors_empty_even_under_worst_latent_state():
    env = ClassroomV2(n_students=5, seed=1)
    env.reset()
    _clear_behaviors(env)
    # Slam latent scalars. The helper must not branch on these.
    for s in env.students:
        s.state["attention"] = 0.05
        s.state["compliance"] = 0.05
        s.state["distress_level"] = 0.95
    for s in env.students:
        assert env._visible_behaviors(s) == []


def test_visible_behaviors_passes_real_high_vis_through():
    env = ClassroomV2(n_students=3, seed=1)
    env.reset()
    _clear_behaviors(env)
    env.students[0].exhibited_behaviors = ["out_of_seat", "calling_out", "random_noise_token"]
    visible = env._visible_behaviors(env.students[0])
    assert visible == ["out_of_seat", "calling_out"]


def test_profile_hint_unknown_when_no_visible_disruption():
    env = ClassroomV2(n_students=3, seed=1)
    env.reset()
    _clear_behaviors(env)
    # Slam latent state — hint must still be "unknown".
    for s in env.students:
        s.state["escalation_risk"] = 0.99
        s.state["attention"] = 0.01
    obs = env._make_observation()
    for sm in obs.student_summaries:
        assert sm.profile_hint == "unknown"
        assert sm.behaviors == []


def test_profile_hint_disruptive_on_visible_disruption():
    env = ClassroomV2(n_students=3, seed=1)
    env.reset()
    _clear_behaviors(env)
    env.students[0].exhibited_behaviors = ["out_of_seat"]
    obs = env._make_observation()
    first = {sm.student_id: sm for sm in obs.student_summaries}[env.students[0].student_id]
    assert first.profile_hint == "disruptive"
    assert "out_of_seat" in first.behaviors


def test_profile_hint_identified_adhd_wins_over_disruption():
    env = ClassroomV2(n_students=3, seed=1)
    env.reset()
    _clear_behaviors(env)
    env.students[0].exhibited_behaviors = ["out_of_seat"]
    env.students[0].identified = True
    obs = env._make_observation()
    first = {sm.student_id: sm for sm in obs.student_summaries}[env.students[0].student_id]
    assert first.profile_hint == "identified_adhd"


def test_profile_hint_vocabulary_is_three_values_only():
    allowed = {"identified_adhd", "disruptive", "unknown"}
    env = ClassroomV2(n_students=5, seed=1)
    env.reset()
    _clear_behaviors(env)
    # Exercise a mix of identified / disruptive / idle.
    env.students[0].identified = True
    env.students[1].exhibited_behaviors = ["calling_out"]
    env.students[2].exhibited_behaviors = ["on_task"]  # not disruptive
    obs = env._make_observation()
    emitted = {sm.profile_hint for sm in obs.student_summaries}
    assert emitted <= allowed
    # Legacy latent labels must never appear.
    assert "inattentive" not in emitted
    assert "typical" not in emitted


# ---------------------------------------------------------------------------
# Downstream reader compatibility
# ---------------------------------------------------------------------------


def test_build_observations_still_consumes_student_summaries():
    """The teacher-observation builder should still produce a
    batch from the new observable-only StudentSummary. It already
    scrubs latent fallbacks defensively; this test confirms the
    new raw path reaches it cleanly."""
    from src.simulation.teacher_observation import (
        build_observations_from_classroom,
    )
    env = ClassroomV2(n_students=5, seed=1)
    env.reset()
    _clear_behaviors(env)
    env.students[0].exhibited_behaviors = ["out_of_seat"]
    env.students[1].identified = True
    obs = env._make_observation()
    batch = build_observations_from_classroom(obs)
    by_id = batch.by_student_id()
    assert by_id[env.students[0].student_id].profile_hint == "disruptive"
    assert by_id[env.students[1].student_id].profile_hint == "identified_adhd"
    # Other students have no visible disruption + not identified → unknown.
    for sid in [s.student_id for s in env.students[2:]]:
        assert by_id[sid].profile_hint == "unknown"
        assert by_id[sid].visible_behaviors == ()


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def test_fixed_seed_run_still_deterministic():
    def run_one() -> list[tuple]:
        env = ClassroomV2(n_students=5, seed=42)
        env.reset()
        for _ in range(30):
            action = TeacherAction(action_type="observe", student_id=None)
            env.step(action)
        return [
            (s.student_id, round(s.state.get("attention", 0.5), 4))
            for s in env.students
        ]

    a = run_one()
    b = run_one()
    assert a == b
    assert a
