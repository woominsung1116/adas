"""Tests for Phase 6 slice 18: ClassroomObservation.class_mood is
now behavior-derived.

Covers:
  - `_make_observation` no longer computes class_mood from
    latent distress_level / attention averages
  - The observation object's class_mood field responds to
    peer visible disruption the same way the student
    ClassroomContext does (via _derive_student_class_mood)
  - Slamming latent state to worst values with no visible
    disruption leaves the mood at "calm"
  - Remaining reader paths (`build_observations_from_classroom`
    and tests that tamper with the field) still work
  - Fixed-seed deterministic simulator runs
  - No regression across the full suite
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


def test_make_observation_no_longer_reads_latent_averages_for_class_mood():
    """The old block computed class_mood from
    `mean(s.state['distress_level'])` / `mean(s.state['attention'])`.
    Those reads must be gone from `_make_observation` now that
    class_mood is derived via `_derive_student_class_mood`.
    """
    src = inspect.getsource(ClassroomV2._make_observation)
    for latent in (
        "distress_level",
        "avg_distress",
        "avg_attention",
    ):
        assert latent not in src, f"_make_observation still reads {latent}"
    assert "_derive_student_class_mood" in src


# ---------------------------------------------------------------------------
# Behavior-derived response
# ---------------------------------------------------------------------------


def _clear_behaviors(env: ClassroomV2) -> None:
    for s in env.students:
        s.exhibited_behaviors = []


def test_class_mood_on_observation_is_calm_without_visible_disruption():
    env = ClassroomV2(n_students=10, seed=1)
    env.reset()
    _clear_behaviors(env)
    obs = env._make_observation()
    assert obs.class_mood == "calm"


def test_class_mood_on_observation_is_tense_on_moderate_disruption():
    env = ClassroomV2(n_students=10, seed=1)
    env.reset()
    _clear_behaviors(env)
    for i in range(3):
        env.students[i].exhibited_behaviors = ["calling_out"]
    obs = env._make_observation()
    assert obs.class_mood == "tense"


def test_class_mood_on_observation_is_chaotic_on_heavy_disruption():
    env = ClassroomV2(n_students=10, seed=1)
    env.reset()
    _clear_behaviors(env)
    for i in range(6):
        env.students[i].exhibited_behaviors = ["out_of_seat"]
    obs = env._make_observation()
    assert obs.class_mood == "chaotic"


def test_class_mood_invariant_under_latent_state_alone():
    """Phase 6 slice 18: no matter how bad latent state gets,
    class_mood stays calm when no peer is visibly disruptive."""
    env = ClassroomV2(n_students=10, seed=1)
    env.reset()
    _clear_behaviors(env)
    for s in env.students:
        s.state["distress_level"] = 0.99
        s.state["attention"] = 0.01
        s.state["escalation_risk"] = 0.99
        s.state["compliance"] = 0.01
    obs = env._make_observation()
    assert obs.class_mood == "calm"


def test_class_mood_on_observation_matches_context():
    """Both _make_context and _make_observation now derive
    class_mood via the same helper, so they should produce
    identical labels for the same classroom state."""
    env = ClassroomV2(n_students=10, seed=1)
    env.reset()
    _clear_behaviors(env)
    for i in range(5):
        env.students[i].exhibited_behaviors = ["out_of_seat"]
    action = TeacherAction(action_type="observe", student_id=None)
    ctx = env._make_context(action)
    obs = env._make_observation()
    assert ctx.class_mood == obs.class_mood


def test_class_mood_vocabulary_is_behavior_ladder_labels():
    """The new vocabulary is strictly {calm, tense, chaotic}.
    Legacy latent labels ('focused', 'distracted', 'neutral') are
    never emitted."""
    env = ClassroomV2(n_students=10, seed=1)
    env.reset()
    # Try several disruption counts and verify each label is from
    # the new ladder vocabulary.
    allowed = {"calm", "tense", "chaotic"}
    for n_disruptive in range(0, 11):
        _clear_behaviors(env)
        for i in range(n_disruptive):
            env.students[i].exhibited_behaviors = ["out_of_seat"]
        obs = env._make_observation()
        assert obs.class_mood in allowed, (
            f"{n_disruptive}/10 disruptive → {obs.class_mood!r} not in ladder"
        )


# ---------------------------------------------------------------------------
# Downstream reader compatibility
# ---------------------------------------------------------------------------


def test_teacher_observation_builder_still_reads_class_mood_field():
    """`build_observations_from_classroom` forwards
    `ClassroomObservation.class_mood` as a pass-through onto
    `TeacherObservationBatch.class_mood`. The new behavior-derived
    value must still flow through this path unchanged."""
    from src.simulation.teacher_observation import (
        build_observations_from_classroom,
    )
    env = ClassroomV2(n_students=10, seed=1)
    env.reset()
    _clear_behaviors(env)
    for i in range(6):
        env.students[i].exhibited_behaviors = ["out_of_seat"]
    obs = env._make_observation()
    assert obs.class_mood == "chaotic"
    batch = build_observations_from_classroom(obs)
    assert batch.class_mood == "chaotic"


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
