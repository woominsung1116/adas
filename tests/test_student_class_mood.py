"""Tests for Phase 6 slice 17: behavior-derived student class mood.

Covers:
  - `_make_context` no longer reads latent distress/attention averages
    for the student class_mood field
  - `_derive_student_class_mood` maps disruption fraction through a
    calm/tense/chaotic ladder using only observable peer behaviors
  - Empty classroom edge case
  - Changing latent state alone does NOT change the mood (invariant)
  - Changing observable exhibited_behaviors DOES change the mood
  - CognitiveStudent still updates deterministically under fixed seeds
  - Existing end-to-end simulator runs still complete cleanly
"""

import inspect

import pytest

from src.simulation.classroom_env_v2 import (
    ClassroomV2,
    TeacherAction,
    _STUDENT_MOOD_CALM_MAX,
    _STUDENT_MOOD_TENSE_MAX,
    _STUDENT_VISIBLE_DISRUPTIVE_BEHAVIORS,
)


# ---------------------------------------------------------------------------
# Static guards
# ---------------------------------------------------------------------------


def test_make_context_does_not_read_latent_averages_for_class_mood():
    """The prior derivation used `_mean([s.state.get("distress_level") ...])`
    etc. Those reads must be gone from `_make_context` now that the
    student class_mood comes from `_derive_student_class_mood`."""
    src = inspect.getsource(ClassroomV2._make_context)
    assert "distress_level" not in src
    assert "avg_distress" not in src
    assert "avg_attention" not in src
    # The new helper must be called.
    assert "_derive_student_class_mood" in src


def test_derive_student_class_mood_source_has_no_latent_reads():
    """The new helper body must read only exhibited_behaviors."""
    src = inspect.getsource(ClassroomV2._derive_student_class_mood)
    for latent in (
        "distress_level",
        "compliance",
        "attention",
        "escalation_risk",
        ".state.get(",
        "state_snapshot",
    ):
        assert latent not in src
    assert "exhibited_behaviors" in src


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------


def _set_behaviors(env: ClassroomV2, indices: list[int], behaviors: list[str]):
    for i in indices:
        env.students[i].exhibited_behaviors = list(behaviors)


def _clear_behaviors(env: ClassroomV2):
    for s in env.students:
        s.exhibited_behaviors = []


def test_fresh_classroom_is_calm():
    env = ClassroomV2(n_students=10, seed=1)
    env.reset()
    _clear_behaviors(env)
    assert env._derive_student_class_mood() == "calm"


def test_1_out_of_10_disruptive_is_calm():
    env = ClassroomV2(n_students=10, seed=1)
    env.reset()
    _clear_behaviors(env)
    _set_behaviors(env, [0], ["out_of_seat"])
    assert env._derive_student_class_mood() == "calm"


def test_3_out_of_10_disruptive_is_tense():
    env = ClassroomV2(n_students=10, seed=1)
    env.reset()
    _clear_behaviors(env)
    _set_behaviors(env, [0, 1, 2], ["calling_out"])
    assert env._derive_student_class_mood() == "tense"


def test_5_out_of_10_disruptive_is_chaotic():
    env = ClassroomV2(n_students=10, seed=1)
    env.reset()
    _clear_behaviors(env)
    _set_behaviors(env, [0, 1, 2, 3, 4], ["out_of_seat"])
    assert env._derive_student_class_mood() == "chaotic"


def test_non_disruptive_behaviors_do_not_count():
    env = ClassroomV2(n_students=10, seed=1)
    env.reset()
    _clear_behaviors(env)
    for s in env.students:
        s.exhibited_behaviors = ["on_task", "listening"]
    assert env._derive_student_class_mood() == "calm"


def test_empty_classroom_is_calm():
    env = ClassroomV2(n_students=1, seed=1)
    env.reset()
    env.students.clear()
    assert env._derive_student_class_mood() == "calm"


def test_mood_ladder_uses_published_thresholds():
    """The module constants are the single source of truth."""
    assert _STUDENT_MOOD_CALM_MAX == 0.10
    assert _STUDENT_MOOD_TENSE_MAX == 0.40
    # Disruptive vocabulary matches the teacher-side boundary.
    assert "out_of_seat" in _STUDENT_VISIBLE_DISRUPTIVE_BEHAVIORS
    assert "on_task" not in _STUDENT_VISIBLE_DISRUPTIVE_BEHAVIORS


# ---------------------------------------------------------------------------
# Invariance: latent state alone does NOT change mood
# ---------------------------------------------------------------------------


def test_changing_latent_state_alone_does_not_change_mood():
    env = ClassroomV2(n_students=10, seed=1)
    env.reset()
    _clear_behaviors(env)
    before = env._derive_student_class_mood()
    # Slam latent scalars to their worst values across every student.
    for s in env.students:
        s.state["distress_level"] = 0.99
        s.state["attention"] = 0.01
        s.state["escalation_risk"] = 0.99
        s.state["compliance"] = 0.01
    after = env._derive_student_class_mood()
    assert before == after == "calm", (
        "latent state changes must not move the student class mood"
    )


def test_make_context_mood_follows_exhibited_behaviors():
    """Full _make_context path: mood must be derived from observable
    peer behavior, independent of latent averages."""
    env = ClassroomV2(n_students=10, seed=1)
    env.reset()
    _clear_behaviors(env)
    # Plant distress on everyone — should not push mood away from calm.
    for s in env.students:
        s.state["distress_level"] = 0.99
    action = TeacherAction(action_type="observe", student_id=None)
    ctx = env._make_context(action)
    assert ctx.class_mood == "calm"

    # Now plant visible disruption on half — mood flips to chaotic.
    for i in range(5):
        env.students[i].exhibited_behaviors = ["out_of_seat"]
    ctx2 = env._make_context(action)
    assert ctx2.class_mood == "chaotic"


# ---------------------------------------------------------------------------
# End-to-end determinism / no regression
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
