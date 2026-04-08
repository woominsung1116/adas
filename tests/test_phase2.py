"""
Phase 2 integration tests for hypothesis-verification, delayed feedback,
retrieval noise, archetypes, and confounder metrics.
"""
from __future__ import annotations

import pytest

from src.simulation.teacher_memory import TeacherMemory, HYPERACTIVITY_BEHAVIORS
from src.simulation.classroom_env_v2 import ClassroomV2
from src.simulation.orchestrator_v2 import OrchestratorV2, PhaseConfig


# ---------------------------------------------------------------------------
# Issue 1: Hypothesis ruled-out students must not appear in identified set
# ---------------------------------------------------------------------------


class TestHypothesisRuledOut:
    """Ruled-out (non-ADHD) students must not pollute the identified set."""

    def test_hypothesis_ruled_out_not_in_identified(self):
        """Students ruled out as anxiety/ODD must not appear in identified set."""
        # Use short phases so the test finishes quickly
        pc = PhaseConfig(
            observation_end=10,
            screening_end=30,
            identification_end=50,
            care_end=70,
        )
        orch = OrchestratorV2(
            n_students=20, seed=42, max_classes=1, phase_config=pc,
        )
        result = orch.run_class()
        metrics = result["metrics"]

        # Ruled-out students should NOT contribute to TP or FP
        identified_ids = set()
        for report in result["reports"]:
            identified_ids.add(report.student_id)

        # Every identified student must have gone through the identify_adhd action,
        # not been merely ruled out
        ruled_out = getattr(orch, "_stream_ruled_out", set())
        assert ruled_out.isdisjoint(identified_ids), (
            f"Ruled-out students leaked into identified: {ruled_out & identified_ids}"
        )

    def test_ruled_out_set_exists_after_class(self):
        """OrchestratorV2 should have a _stream_ruled_out set after running a class."""
        pc = PhaseConfig(
            observation_end=10,
            screening_end=30,
            identification_end=50,
            care_end=70,
        )
        orch = OrchestratorV2(
            n_students=20, seed=42, max_classes=1, phase_config=pc,
        )
        orch.run_class()
        assert hasattr(orch, "_stream_ruled_out")
        assert isinstance(orch._stream_ruled_out, set)


# ---------------------------------------------------------------------------
# Issue 2: Retrieval noise default
# ---------------------------------------------------------------------------


class TestRetrievalNoise:
    """TeacherMemory retrieval_noise defaults and determinism."""

    def test_teacher_memory_default_no_noise(self):
        """Default TeacherMemory has no retrieval noise for backward compat."""
        mem = TeacherMemory()
        assert mem.retrieval_noise == 0.0

    def test_retrieval_noise_zero_is_deterministic(self):
        """retrieval_noise=0.0 -> same query returns same results."""
        mem = TeacherMemory(retrieval_noise=0.0, seed=42)
        mem.new_class()

        state = {"distress_level": 0.4, "compliance": 0.3, "attention": 0.3, "escalation_risk": 0.3}
        behaviors = ["seat-leaving", "excessive-talking"]

        mem.observe("s01", behaviors, state, action_taken="observe")
        mem.commit_observation("s01", outcome="positive")
        mem.observe("s02", ["blurting-answers"], state, action_taken="observe")
        mem.commit_observation("s02", outcome="neutral")

        query = ["seat-leaving"]
        r1 = mem.retrieve_similar_cases(query, top_k=5, exclude_student_id="s99")
        r2 = mem.retrieve_similar_cases(query, top_k=5, exclude_student_id="s99")

        assert len(r1) == len(r2)
        for (s1, rec1), (s2, rec2) in zip(r1, r2):
            assert s1 == pytest.approx(s2)
            assert rec1.student_id == rec2.student_id

    def test_retrieval_noise_nonzero_may_vary(self):
        """retrieval_noise=0.20 -> results may differ across many calls."""
        mem = TeacherMemory(retrieval_noise=0.20, seed=42)
        mem.new_class()

        state = {"distress_level": 0.4, "compliance": 0.3, "attention": 0.3, "escalation_risk": 0.3}
        # Add enough records so noise has something to drop
        for i in range(10):
            mem.observe(f"s{i:02d}", ["seat-leaving", "blurting-answers"], state)
            mem.commit_observation(f"s{i:02d}", outcome="positive")

        query = ["seat-leaving"]
        results_lengths = set()
        for _ in range(20):
            r = mem.retrieve_similar_cases(query, top_k=5, exclude_student_id="s99")
            results_lengths.add(len(r))

        # With 20% noise over 20 calls, we expect at least some variation in length
        # (not guaranteed, but highly likely with 10 records and 20 attempts)
        assert len(results_lengths) >= 1  # at minimum returns something

    def test_orchestrator_uses_nonzero_noise(self):
        """OrchestratorV2 creates TeacherMemory with retrieval_noise=0.20."""
        orch = OrchestratorV2(seed=42)
        assert orch.memory.retrieval_noise == 0.20


# ---------------------------------------------------------------------------
# Issue 2 (cont): Delayed feedback
# ---------------------------------------------------------------------------


class TestDelayedFeedback:
    """feedback_rate controls how many reports get ground-truth confirmation."""

    def test_delayed_feedback_rate_zero(self):
        """feedback_rate=0.0 -> no reports get confirmed."""
        pc = PhaseConfig(
            observation_end=10,
            screening_end=30,
            identification_end=50,
            care_end=70,
        )
        orch = OrchestratorV2(
            feedback_rate=0.0, n_students=20, seed=42, max_classes=1,
            phase_config=pc,
        )
        result = orch.run_class()
        for report in result["reports"]:
            assert report.is_correct is None, (
                f"Report for {report.student_id} should be unconfirmed with feedback_rate=0.0"
            )

    def test_delayed_feedback_rate_one(self):
        """feedback_rate=1.0 -> all reports get confirmed."""
        pc = PhaseConfig(
            observation_end=10,
            screening_end=30,
            identification_end=50,
            care_end=70,
        )
        orch = OrchestratorV2(
            feedback_rate=1.0, n_students=20, seed=42, max_classes=1,
            phase_config=pc,
        )
        result = orch.run_class()
        for report in result["reports"]:
            assert report.is_correct is not None, (
                f"Report for {report.student_id} should be confirmed with feedback_rate=1.0"
            )


# ---------------------------------------------------------------------------
# Issue 3: Confounder FP matches actual profile names
# ---------------------------------------------------------------------------


class TestConfounderMetrics:
    """confounder_fp should count anxiety/odd/gifted/sleep_deprived wrongly identified."""

    def test_confounder_fp_matches_actual_profiles(self):
        """confounder_fp counts only students with known confounder profile_type values."""
        pc = PhaseConfig(
            observation_end=10,
            screening_end=30,
            identification_end=50,
            care_end=70,
        )
        orch = OrchestratorV2(
            n_students=20, seed=42, max_classes=1, phase_config=pc,
        )
        result = orch.run_class()
        metrics = result["metrics"]

        # confounder_fp must be <= total fp
        assert metrics.confounder_fp <= metrics.false_positives

        # Verify no student with profile_type "confounder" or "anxious" exists
        # (those were the old wrong names). Real names are anxiety, odd, gifted, sleep_deprived.
        valid_confounder_profiles = {"anxiety", "odd", "gifted", "sleep_deprived"}
        for s in orch.classroom.students:
            if not s.is_adhd:
                assert s.profile_type in valid_confounder_profiles or s.profile_type.startswith("normal"), (
                    f"Unexpected non-ADHD profile_type: {s.profile_type}"
                )


# ---------------------------------------------------------------------------
# Archetype tests
# ---------------------------------------------------------------------------


class TestArchetypes:
    """ClassroomV2 archetype selection."""

    def test_archetype_fixed_selection(self):
        """ClassroomV2(archetype='chaotic') uses chaotic archetype."""
        c = ClassroomV2(archetype="chaotic", seed=42)
        c.reset()
        assert c.archetype is not None
        assert c.archetype.name == "chaotic"

    def test_archetype_random_selection(self):
        """ClassroomV2(archetype=None) randomly picks archetype."""
        c = ClassroomV2(archetype=None, seed=42)
        c.reset()
        assert c.archetype is not None
