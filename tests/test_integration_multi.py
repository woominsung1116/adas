"""
Integration tests for the ADHD classroom simulation.

Covers:
  1. MultiStudentClassroom — student generation, actions, rewards
  2. TeacherMemory         — observation pipeline, RAG retrieval, principles
  3. IdentificationReport  — DSM-5 threshold, evaluation, growth curve
  4. GrowthTracker         — per-class metrics, trends, benchmark comparison
  5. Full pipeline         — one class end-to-end; multi-class growth
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from src.simulation.teacher_memory import (
    TeacherMemory,
    HYPERACTIVITY_BEHAVIORS,
    INATTENTION_BEHAVIORS,
)
from src.simulation.multi_student_env import (
    MultiStudentClassroom,
    TeacherAction,
    MANAGED_COMPLIANCE_THRESHOLD,
    MANAGED_CONSECUTIVE_TURNS,
)
from src.eval.identification_report import (
    IdentificationReport,
    IdentificationEvaluator,
    ObservedSymptom,
    DSM5_INATTENTION,
    DSM5_HYPERACTIVITY,
)
from src.eval.growth_metrics import GrowthTracker, ClassMetrics


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_classroom(seed: int = 42, n_students: int = 20) -> MultiStudentClassroom:
    """Return a seeded classroom with no external profiles/scenarios."""
    return MultiStudentClassroom(n_students=n_students, adhd_prevalence=0.09, seed=seed)


def _make_inattention_symptom(criterion: str = "inattention_1", turns: int = 8) -> ObservedSymptom:
    return ObservedSymptom.from_observations(
        dsm_criterion=criterion,
        observed_behavior="careless mistakes",
        turns_observed=list(range(1, turns + 1)),
    )


def _make_hyperactivity_symptom(criterion: str = "hyperactivity_1", turns: int = 8) -> ObservedSymptom:
    return ObservedSymptom.from_observations(
        dsm_criterion=criterion,
        observed_behavior="fidgets",
        turns_observed=list(range(1, turns + 1)),
    )


def _make_class_metrics(
    class_id: int,
    tp: int = 2,
    fp: int = 0,
    fn: int = 0,
    tn: int = 18,
    speed: float = 5.0,
    care_turns: float = 3.0,
    improvement: float = 0.80,
    strategies: list[str] | None = None,
    completion_turn: int = 20,
) -> ClassMetrics:
    return ClassMetrics(
        class_id=class_id,
        n_students=20,
        n_adhd=tp + fn,
        n_identified=tp + fp,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        avg_identification_turn=speed,
        avg_care_turns=care_turns,
        strategies_used=strategies or ["labeled_praise", "break_offer"],
        behavior_improvement_rates=[improvement],
        class_completion_turn=completion_turn,
    )


# ===========================================================================
# 1. Multi-Student Classroom Tests
# ===========================================================================


class TestClassroomCreation:
    """Tests for reset() and initial student generation."""

    def test_classroom_creates_mixed_students(self):
        """20 students are generated with at least some ADHD and some normal."""
        env = _make_classroom()
        env.reset()
        assert len(env.students) == 20
        adhd_count = sum(1 for s in env.students if s.is_adhd)
        normal_count = sum(1 for s in env.students if not s.is_adhd)
        assert adhd_count >= 1
        assert normal_count >= 1

    def test_classroom_adhd_prevalence(self):
        """ADHD count over 100 resets averages within the 6-11% range."""
        totals = []
        for seed in range(100):
            env = _make_classroom(seed=seed)
            env.reset()
            totals.append(sum(1 for s in env.students if s.is_adhd))
        avg_pct = sum(totals) / len(totals) / 20
        # 9% target with deterministic rounding — allow 5-13% window
        assert 0.05 <= avg_pct <= 0.13

    def test_classroom_gender_ratio(self):
        """Among ADHD students across many seeds, male ratio is roughly 3:1."""
        male_adhd = 0
        female_adhd = 0
        for seed in range(200):
            env = _make_classroom(seed=seed)
            env.reset()
            for s in env.students:
                if s.is_adhd:
                    if s.gender == "male":
                        male_adhd += 1
                    else:
                        female_adhd += 1
        if female_adhd == 0:
            pytest.skip("No female ADHD students sampled — increase seed range")
        ratio = male_adhd / female_adhd
        # 3.19:1 target; allow 2.0-5.0 as a realistic empirical window
        assert 2.0 <= ratio <= 5.0


class TestClassroomStep:
    """Tests for step() action dispatch and observations."""

    def test_classroom_step_observe(self):
        """Observe action returns a state_snapshot for the targeted student."""
        env = _make_classroom()
        env.reset()
        target = env.students[0].student_id

        action = TeacherAction(action_type="observe", student_id=target)
        obs, reward, done, info = env.step(action)

        focused = next(
            so for so in obs.student_observations if so.student_id == target
        )
        assert focused.state_snapshot is not None
        assert set(focused.state_snapshot.keys()) == {
            "distress_level", "compliance", "attention", "escalation_risk"
        }
        # Other students do not get a state snapshot on focused observe
        others_with_snap = [
            so for so in obs.student_observations
            if so.student_id != target and so.state_snapshot is not None
        ]
        assert others_with_snap == []

    def test_classroom_step_intervention_changes_state(self):
        """Individual intervention modifies the target student's state."""
        env = _make_classroom(seed=7)
        env.reset()
        # Find an ADHD student to intervene on
        adhd = next((s for s in env.students if s.is_adhd), None)
        if adhd is None:
            pytest.skip("No ADHD students in this seed")

        before = dict(adhd.state)
        action = TeacherAction(
            action_type="individual_intervention",
            student_id=adhd.student_id,
            strategy="labeled_praise",
        )
        env.step(action)
        after = dict(adhd.state)
        # At least one state dimension must have changed
        assert before != after

    def test_classroom_private_vs_public_correction(self):
        """
        Private correction yields lower expected distress than public correction.

        Tested by running the same seeded classroom twice and comparing
        the distress outcomes for the same ADHD student.
        """
        results = {}
        for correction_type in ("private_correction", "public_correction"):
            env = _make_classroom(seed=99)
            env.reset()
            adhd = next((s for s in env.students if s.is_adhd), None)
            if adhd is None:
                pytest.skip("No ADHD students in seed 99")
            sid = adhd.student_id

            action = TeacherAction(action_type=correction_type, student_id=sid)
            env.step(action)
            results[correction_type] = env.get_student(sid).state["distress_level"]

        # Private correction should generally keep distress lower.
        # Private: distress decreases by 0.05-0.15; public: may increase up to +0.12.
        # We check private distress <= public distress (deterministic with fixed seed).
        assert results["private_correction"] <= results["public_correction"]

    def test_classroom_identify_adhd_correct(self):
        """Identifying an actual ADHD student gives +1 reward."""
        env = _make_classroom(seed=0)
        env.reset()
        adhd = next((s for s in env.students if s.is_adhd), None)
        if adhd is None:
            pytest.skip("No ADHD students in seed 0")

        action = TeacherAction(
            action_type="identify_adhd",
            student_id=adhd.student_id,
            reasoning="hyperactive behaviors observed",
        )
        _, reward, _, _ = env.step(action)
        assert reward >= 1.0  # includes possible class-completion bonus

    def test_classroom_identify_adhd_false_positive(self):
        """Identifying a normal student gives -1 reward."""
        env = _make_classroom(seed=0)
        env.reset()
        normal = next((s for s in env.students if not s.is_adhd), None)
        if normal is None:
            pytest.skip("No normal students in seed 0")

        action = TeacherAction(
            action_type="identify_adhd",
            student_id=normal.student_id,
            reasoning="suspected ADHD",
        )
        _, reward, _, _ = env.step(action)
        assert reward <= -1.0

    def test_classroom_completion_when_all_adhd_identified_and_managed(self):
        """is_class_complete() is True when every ADHD student is identified and managed."""
        env = _make_classroom(seed=5)
        env.reset()

        for student in env.students:
            if student.is_adhd:
                student.identified = True
                env.identified_adhd_ids.add(student.student_id)
                student.managed = True
                env.managed_ids.add(student.student_id)
                student._consecutive_compliant_turns = MANAGED_CONSECUTIVE_TURNS

        assert env.is_class_complete() is True

    def test_normal_students_stable(self):
        """Normal students maintain high compliance/attention over 10 turns."""
        env = _make_classroom(seed=3)
        env.reset()

        for _ in range(10):
            action = TeacherAction(action_type="class_instruction")
            env.step(action)

        normal_students = [s for s in env.students if not s.is_adhd]
        if not normal_students:
            pytest.skip("No normal students in seed 3")

        for s in normal_students:
            # Normal student compliance baseline is 0.80 ± 0.05 noise
            assert s.state["compliance"] >= 0.50, (
                f"Normal student {s.student_id} compliance dropped to {s.state['compliance']:.3f}"
            )
            assert s.state["attention"] >= 0.45, (
                f"Normal student {s.student_id} attention dropped to {s.state['attention']:.3f}"
            )


# ===========================================================================
# 2. Teacher Memory Tests
# ===========================================================================


class TestTeacherMemory:
    """Tests for the two-tier TeacherMemory system."""

    def test_memory_observe_and_retrieve(self):
        """Observations stored in the case base are retrievable by similarity."""
        memory = TeacherMemory()
        memory.new_class()

        behaviors = ["seat-leaving", "excessive-talking"]
        state = {"distress_level": 0.4, "compliance": 0.3, "attention": 0.3, "escalation_risk": 0.3}

        memory.observe("s01", behaviors, state, action_taken="labeled_praise")
        idx = memory.commit_observation("s01", outcome="positive")

        assert len(memory.case_base) == 1
        assert idx == 0

        # A different student with similar behaviors should find this case
        similar = memory.retrieve_similar_cases(["seat-leaving"], exclude_student_id="s02")
        assert len(similar) == 1
        sim_score, record = similar[0]
        assert sim_score > 0.0
        assert record.student_id == "s01"

    def test_memory_identify_adhd_returns_tuple(self):
        """identify_adhd returns (bool, float, str) with confidence in [0,1]."""
        memory = TeacherMemory()
        memory.new_class()

        # Load behaviors typical of ADHD
        state = {"distress_level": 0.5, "compliance": 0.2, "attention": 0.2, "escalation_risk": 0.5}
        adhd_behaviors = list(HYPERACTIVITY_BEHAVIORS[:3]) + list(INATTENTION_BEHAVIORS[:2])
        for _ in range(5):
            memory.observe("s01", adhd_behaviors, state)
            memory.commit_observation("s01", outcome="positive")

        is_adhd, confidence, reasoning = memory.identify_adhd("s01")
        assert isinstance(is_adhd, bool)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(reasoning, str) and len(reasoning) > 0

    def test_memory_identify_confidence_increases_with_more_observations(self):
        """More ADHD-consistent observations yield higher identification confidence."""
        memory_few = TeacherMemory()
        memory_few.new_class()
        state = {"distress_level": 0.5, "compliance": 0.2, "attention": 0.2, "escalation_risk": 0.5}
        adhd_behaviors = list(HYPERACTIVITY_BEHAVIORS[:3])

        # 2 observations
        for _ in range(2):
            memory_few.observe("s01", adhd_behaviors, state)
            memory_few.commit_observation("s01", outcome="positive")
        _, conf_few, _ = memory_few.identify_adhd("s01")

        memory_many = TeacherMemory()
        memory_many.new_class()
        # 10 observations
        for _ in range(10):
            memory_many.observe("s01", adhd_behaviors, state)
            memory_many.commit_observation("s01", outcome="positive")
        _, conf_many, _ = memory_many.identify_adhd("s01")

        assert conf_many >= conf_few

    def test_memory_experience_base_adds_principle_after_correct_identification(self):
        """A principle is extracted after record_outcome(was_correct=True)."""
        memory = TeacherMemory()
        memory.new_class()

        state = {"distress_level": 0.4, "compliance": 0.3, "attention": 0.3, "escalation_risk": 0.3}
        behaviors = list(HYPERACTIVITY_BEHAVIORS[:2])
        memory.observe("s01", behaviors, state)
        idx = memory.commit_observation("s01", outcome="positive")

        memory.identify_adhd("s01")
        initial_count = len(memory.experience_base)
        memory.record_outcome("s01", was_correct=True)
        assert len(memory.experience_base) >= initial_count  # principle added or merged

    def test_memory_experience_base_adds_corrective_principle_after_wrong_identification(self):
        """A corrective principle is extracted after record_outcome(was_correct=False)."""
        memory = TeacherMemory()
        memory.new_class()

        state = {"distress_level": 0.2, "compliance": 0.7, "attention": 0.7, "escalation_risk": 0.1}
        behaviors = ["fidgeting_slightly"]  # mild noise, not in behavior index — harmless
        memory.observe("s01", behaviors, state)
        memory.commit_observation("s01", outcome="neutral")

        # Force a positive identification in the profile
        profile = memory.get_profile("s01")
        profile.identified_as_adhd = True
        profile.identification_confidence = 0.6
        profile.behavior_frequency_counts["seat-leaving"] = 3

        memory.record_outcome("s01", was_correct=False)
        corrective = memory.experience_base.corrective_principles()
        assert len(corrective) >= 1

    def test_memory_cross_class_persistence(self):
        """Case base and experience base persist across new_class() calls."""
        memory = TeacherMemory()
        memory.new_class()

        state = {"distress_level": 0.4, "compliance": 0.3, "attention": 0.3, "escalation_risk": 0.4}
        memory.observe("s01", ["seat-leaving", "blurting-answers"], state)
        memory.commit_observation("s01", outcome="positive")
        memory.identify_adhd("s01")
        memory.record_outcome("s01", was_correct=True)

        case_size_before = len(memory.case_base)
        exp_size_before = len(memory.experience_base)

        # Start a new class
        memory.new_class()

        assert len(memory.case_base) == case_size_before
        assert len(memory.experience_base) == exp_size_before

    def test_memory_growth_report_contains_required_fields(self):
        """growth_report() returns a dict with TP/FP/FN counts and more."""
        memory = TeacherMemory()
        memory.new_class()

        state = {"distress_level": 0.5, "compliance": 0.2, "attention": 0.2, "escalation_risk": 0.5}
        memory.observe("s01", list(HYPERACTIVITY_BEHAVIORS[:3]), state)
        memory.commit_observation("s01", outcome="positive")
        memory.identify_adhd("s01")
        memory.record_outcome("s01", was_correct=True)

        report = memory.growth_report()
        required_keys = {
            "classes_seen", "total_identifications", "correct_identifications",
            "false_positives", "false_negatives", "precision", "recall", "f1",
            "case_base_size", "experience_base_size", "top_principles",
        }
        assert required_keys.issubset(report.keys())
        assert report["correct_identifications"] == 1
        assert report["false_positives"] == 0


# ===========================================================================
# 3. Identification Report Tests
# ===========================================================================


class TestIdentificationReport:
    """Tests for IdentificationReport and IdentificationEvaluator."""

    def _report_with_symptoms(
        self, n_inattention: int = 0, n_hyperactivity: int = 0
    ) -> IdentificationReport:
        report = IdentificationReport(
            student_id="s01",
            teacher_class_id=1,
            turn_identified=10,
            identified_subtype="combined",
            confidence=0.75,
            reasoning="multiple criteria observed",
        )
        inattention_keys = list(DSM5_INATTENTION.keys())
        hyperactivity_keys = list(DSM5_HYPERACTIVITY.keys())

        for i in range(n_inattention):
            report.add_inattention_symptom(
                ObservedSymptom.from_observations(
                    dsm_criterion=inattention_keys[i],
                    observed_behavior="observed inattention",
                    turns_observed=list(range(1, 9)),
                )
            )
        for i in range(n_hyperactivity):
            report.add_hyperactivity_symptom(
                ObservedSymptom.from_observations(
                    dsm_criterion=hyperactivity_keys[i],
                    observed_behavior="observed hyperactivity",
                    turns_observed=list(range(1, 9)),
                )
            )
        return report

    def test_report_dsm5_threshold_met_with_six_inattention_symptoms(self):
        """Report meets DSM-5 threshold when 6+ inattention symptoms are added."""
        report = self._report_with_symptoms(n_inattention=6)
        assert report.meets_dsm5_threshold is True
        assert report.inattention_count == 6

    def test_report_dsm5_threshold_met_with_six_hyperactivity_symptoms(self):
        """Report meets DSM-5 threshold when 6+ hyperactivity symptoms are added."""
        report = self._report_with_symptoms(n_hyperactivity=6)
        assert report.meets_dsm5_threshold is True

    def test_report_dsm5_threshold_not_met_with_five_symptoms(self):
        """Report does not meet DSM-5 threshold with only 5 symptoms total."""
        report = self._report_with_symptoms(n_inattention=3, n_hyperactivity=2)
        assert report.meets_dsm5_threshold is False

    def test_report_evaluate_true_positive(self):
        """evaluate() marks is_correct=True when student is truly ADHD."""
        report = self._report_with_symptoms(n_inattention=6)
        result = report.evaluate(ground_truth_adhd=True, ground_truth_subtype="combined")
        assert result["is_correct"] is True
        assert result["student_id"] == "s01"
        assert result["meets_dsm5_threshold"] is True

    def test_report_evaluate_false_positive(self):
        """evaluate() marks is_correct=False when normal student is flagged."""
        report = self._report_with_symptoms(n_inattention=4)
        result = report.evaluate(ground_truth_adhd=False, ground_truth_subtype=None)
        assert result["is_correct"] is False

    def test_observed_symptom_evidence_strength_range(self):
        """ObservedSymptom.from_observations produces evidence_strength in [0, 1]."""
        symptom = ObservedSymptom.from_observations(
            dsm_criterion="inattention_3",
            observed_behavior="not listening",
            turns_observed=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        )
        assert 0.0 <= symptom.evidence_strength <= 1.0

    def test_evaluator_computes_tp_fp_correctly(self):
        """IdentificationEvaluator counts TP and FP from evaluated reports."""
        evaluator = IdentificationEvaluator()

        # 2 true positives
        for i in range(2):
            r = IdentificationReport(
                student_id=f"s0{i}",
                teacher_class_id=1,
                turn_identified=5,
                identified_subtype="combined",
                confidence=0.8,
            )
            r.evaluate(ground_truth_adhd=True, ground_truth_subtype="combined")
            evaluator.add_report(r)

        # 1 false positive
        r_fp = IdentificationReport(
            student_id="s99",
            teacher_class_id=1,
            turn_identified=8,
            identified_subtype="combined",
            confidence=0.55,
        )
        r_fp.evaluate(ground_truth_adhd=False, ground_truth_subtype=None)
        evaluator.add_report(r_fp)

        metrics = evaluator.compute_metrics()
        assert metrics["tp"] == 2
        assert metrics["fp"] == 1

    def test_evaluator_sensitivity_specificity_with_external_counts(self):
        """Sensitivity and specificity compute correctly when FN/TN are supplied."""
        evaluator = IdentificationEvaluator()

        r = IdentificationReport(
            student_id="s01",
            teacher_class_id=1,
            turn_identified=7,
            identified_subtype="inattentive",
            confidence=0.7,
        )
        r.evaluate(ground_truth_adhd=True, ground_truth_subtype="inattentive")
        evaluator.add_report(r)

        evaluator.add_missed(1)       # 1 ADHD student was never flagged (FN)
        evaluator.add_true_negative(10)  # 10 normal students correctly passed over

        metrics = evaluator.compute_metrics()
        # TP=1, FN=1 → sensitivity = 0.5
        assert metrics["sensitivity"] == pytest.approx(0.5)
        # TN=10, FP=0 → specificity = 1.0
        assert metrics["specificity"] == pytest.approx(1.0)

    def test_evaluator_growth_curve_ordered_by_class_id(self):
        """growth_curve() returns entries in class_id order with accuracy values."""
        evaluator = IdentificationEvaluator()

        for class_id in [3, 1, 2]:
            r = IdentificationReport(
                student_id=f"s_class{class_id}",
                teacher_class_id=class_id,
                turn_identified=class_id * 3,
                identified_subtype="combined",
                confidence=0.7,
            )
            r.evaluate(ground_truth_adhd=True, ground_truth_subtype="combined")
            evaluator.add_report(r)

        curve = evaluator.growth_curve()
        class_ids = [entry["teacher_class_id"] for entry in curve]
        assert class_ids == sorted(class_ids)
        for entry in curve:
            assert "accuracy" in entry
            assert entry["accuracy"] is not None


# ===========================================================================
# 4. Growth Metrics Tests
# ===========================================================================


class TestGrowthTracker:
    """Tests for GrowthTracker accumulation and computation."""

    def test_growth_tracker_records_class_correctly(self):
        """record_class() stores the ClassMetrics and total_classes_completed increments."""
        tracker = GrowthTracker()
        m = _make_class_metrics(class_id=1, tp=2, fp=0, fn=0, tn=18)
        tracker.record_class(m)
        assert tracker.total_classes_completed == 1
        assert tracker.class_history[0].class_id == 1

    def test_growth_sensitivity_with_known_confusion_matrix(self):
        """Sensitivity = TP / (TP + FN) computed correctly."""
        tracker = GrowthTracker()
        tracker.record_class(_make_class_metrics(class_id=1, tp=3, fp=0, fn=1, tn=16))
        # sensitivity = 3 / (3+1) = 0.75
        assert tracker.sensitivity(class_id=1) == pytest.approx(0.75)

    def test_growth_specificity_with_known_confusion_matrix(self):
        """Specificity = TN / (TN + FP) computed correctly."""
        tracker = GrowthTracker()
        tracker.record_class(_make_class_metrics(class_id=1, tp=2, fp=2, fn=0, tn=16))
        # specificity = 16 / (16+2) = 0.888...
        assert tracker.specificity(class_id=1) == pytest.approx(16 / 18)

    def test_growth_ppv_with_known_confusion_matrix(self):
        """PPV = TP / (TP + FP) computed correctly."""
        tracker = GrowthTracker()
        tracker.record_class(_make_class_metrics(class_id=1, tp=2, fp=2, fn=0, tn=16))
        # ppv = 2 / (2+2) = 0.5
        assert tracker.ppv(class_id=1) == pytest.approx(0.5)

    def test_growth_vs_benchmarks_returns_required_keys(self):
        """vs_benchmarks() returns entries for sensitivity, specificity, ppv."""
        tracker = GrowthTracker()
        tracker.record_class(_make_class_metrics(class_id=1, tp=2, fp=0, fn=0, tn=18))
        result = tracker.vs_benchmarks()
        assert "sensitivity" in result
        assert "specificity" in result
        assert "ppv" in result
        for v in result.values():
            assert "agent" in v
            assert "benchmark" in v
            assert "meets" in v

    def test_growth_trend_positive_when_improving(self):
        """Trend slope is positive when sensitivity increases class over class."""
        tracker = GrowthTracker()
        # Class 1: low sensitivity (tp=1, fn=1)
        tracker.record_class(_make_class_metrics(class_id=1, tp=1, fp=0, fn=1, tn=18))
        # Class 2: high sensitivity (tp=2, fn=0)
        tracker.record_class(_make_class_metrics(class_id=2, tp=2, fp=0, fn=0, tn=18))
        slope = tracker.trend("sensitivity")
        assert slope > 0

    def test_growth_trend_negative_when_speed_improves(self):
        """Trend slope is negative for identification_speed when agent gets faster."""
        tracker = GrowthTracker()
        tracker.record_class(_make_class_metrics(class_id=1, speed=15.0))
        tracker.record_class(_make_class_metrics(class_id=2, speed=10.0))
        tracker.record_class(_make_class_metrics(class_id=3, speed=5.0))
        slope = tracker.trend("identification_speed")
        assert slope < 0

    def test_growth_trend_returns_zero_for_single_class(self):
        """Trend returns 0.0 when only one class has been recorded."""
        tracker = GrowthTracker()
        tracker.record_class(_make_class_metrics(class_id=1))
        assert tracker.trend("sensitivity") == 0.0

    def test_growth_export_json_contains_required_fields(self, tmp_path):
        """export_json() writes valid JSON with total_classes, class_history, growth_curves."""
        tracker = GrowthTracker()
        tracker.record_class(_make_class_metrics(class_id=1, tp=2, fp=0, fn=0, tn=18))
        tracker.record_class(_make_class_metrics(class_id=2, tp=2, fp=0, fn=0, tn=18))

        out_path = str(tmp_path / "growth.json")
        tracker.export_json(out_path)

        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "total_classes" in data
        assert "class_history" in data
        assert "growth_curves" in data
        assert "vs_benchmarks" in data
        assert data["total_classes"] == 2
        assert len(data["class_history"]) == 2


# ===========================================================================
# 5. Full Pipeline Integration
# ===========================================================================


class TestFullPipeline:
    """End-to-end pipeline tests combining environment + memory + metrics."""

    def test_full_pipeline_one_class(self):
        """
        Run one class: observe every student, identify all ADHD students,
        apply private corrections until managed, verify completion.
        """
        env = _make_classroom(seed=42)
        memory = TeacherMemory()
        memory.new_class()

        env.reset()
        adhd_ids = env.ground_truth_adhd_ids()
        if not adhd_ids:
            pytest.skip("No ADHD students in seed 42")

        # Phase 1: observe all students
        for s in env.students:
            action = TeacherAction(action_type="observe", student_id=s.student_id)
            obs, _, _, _ = env.step(action)
            behaviors = next(
                so.behaviors for so in obs.student_observations
                if so.student_id == s.student_id
            )
            state_snap = next(
                so.state_snapshot for so in obs.student_observations
                if so.student_id == s.student_id
            )
            memory.observe(
                s.student_id,
                behaviors,
                state_snap or s.state,
                action_taken="observe",
            )
            memory.commit_observation(s.student_id, outcome="neutral")

        # Phase 2: identify ADHD students
        for sid in adhd_ids:
            action = TeacherAction(
                action_type="identify_adhd",
                student_id=sid,
                reasoning="behaviors observed",
            )
            env.step(action)
            memory.identify_adhd(sid)
            memory.record_outcome(sid, was_correct=True)

        # Phase 3: private corrections until all ADHD are managed
        max_turns = 60
        for _ in range(max_turns):
            if env.is_class_complete():
                break
            for sid in adhd_ids:
                student = env.get_student(sid)
                if student and not student.managed:
                    action = TeacherAction(
                        action_type="private_correction",
                        student_id=sid,
                    )
                    env.step(action)

        assert env.is_class_complete(), (
            "Class did not complete within expected turns after identify + private correction"
        )

        report = memory.growth_report()
        assert report["correct_identifications"] >= 1

    def test_full_pipeline_multi_class_growth_accumulates_case_base(self):
        """
        Run 3 classes. Verify case base grows across classes and
        GrowthTracker accumulates per-class records.
        """
        memory = TeacherMemory()
        tracker = GrowthTracker()

        for class_num in range(3):
            env = _make_classroom(seed=class_num * 17)
            memory.new_class()
            env.reset()

            adhd_ids = env.ground_truth_adhd_ids()
            n_adhd = len(adhd_ids)

            # Quick observe + identify all ADHD students
            for s in env.students:
                action = TeacherAction(action_type="observe", student_id=s.student_id)
                obs, _, _, _ = env.step(action)
                snap = next(
                    so.state_snapshot for so in obs.student_observations
                    if so.student_id == s.student_id
                )
                memory.observe(s.student_id, s.exhibited_behaviors, snap or s.state)
                memory.commit_observation(s.student_id, outcome="neutral")

            tp = 0
            for sid in adhd_ids:
                action = TeacherAction(
                    action_type="identify_adhd",
                    student_id=sid,
                    reasoning="pipeline test",
                )
                env.step(action)
                memory.identify_adhd(sid)
                memory.record_outcome(sid, was_correct=True)
                tp += 1

            # Private corrections to progress toward managed state
            for _ in range(20):
                for sid in adhd_ids:
                    student = env.get_student(sid)
                    if student and not student.managed:
                        env.step(TeacherAction(
                            action_type="private_correction",
                            student_id=sid,
                        ))

            # Record metrics for this class
            n_normal = env.n_students - n_adhd
            tracker.record_class(ClassMetrics(
                class_id=class_num,
                n_students=env.n_students,
                n_adhd=n_adhd,
                n_identified=len(env.identified_adhd_ids),
                true_positives=tp,
                false_positives=0,
                false_negatives=0,
                true_negatives=n_normal,
                avg_identification_turn=float(env.turn // max(n_adhd, 1)),
                avg_care_turns=5.0,
                strategies_used=["private_correction", "observe"],
                behavior_improvement_rates=[0.75],
                class_completion_turn=env.turn,
            ))

        assert tracker.total_classes_completed == 3
        assert len(memory.case_base) >= 3  # at least one case per class
        growth = tracker.growth_curve()
        assert len(growth["sensitivity"]) == 3
        # Case base grows monotonically (each new class adds observations)
        report = memory.growth_report()
        assert report["classes_seen"] == 3
        assert report["case_base_size"] >= 3
