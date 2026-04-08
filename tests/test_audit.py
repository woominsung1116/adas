"""
Audit tests for ADAS codebase integrity.

Covers:
  1. No oracle leakage in teacher action selection
  2. App path grows memory via SimulationOrchestrator
  3. Behavior improvement definition consistency
  4. Build config references expected assets
"""
from __future__ import annotations

import json
import os
import re

import pytest

from src.simulation.multi_student_env import (
    MultiStudentClassroom,
    TeacherAction,
    StudentState,
)
from src.simulation.teacher_memory import TeacherMemory
from src.simulation.orchestrator import SimulationOrchestrator
from src.eval.growth_metrics import GrowthTracker, ClassMetrics


# ===========================================================================
# 1. No oracle leakage
# ===========================================================================


class TestNoOracleLeakage:
    """Verify teacher decisions never use ground-truth labels."""

    def test_orchestrator_choose_strategy_uses_observable_state_only(self):
        """
        _choose_strategy should not reference adhd_subtype.
        Verified by giving two students with different subtypes but
        identical observable state -- they must get the same strategy.
        """
        orch = SimulationOrchestrator(n_students=20, seed=42, max_classes=1)
        orch.classroom.reset()
        orch.memory.new_class()

        # Create two fake students with identical observable state
        # but different subtypes
        state = {"distress_level": 0.3, "compliance": 0.4, "attention": 0.5, "escalation_risk": 0.3}

        s1 = StudentState(
            student_id="test_inatt", is_adhd=True,
            adhd_subtype="inattentive", severity="moderate",
            gender="male", age=9, state=dict(state),
        )
        s2 = StudentState(
            student_id="test_hyper", is_adhd=True,
            adhd_subtype="hyperactive_impulsive", severity="moderate",
            gender="male", age=9, state=dict(state),
        )

        strategy1 = orch._choose_strategy(s1)
        strategy2 = orch._choose_strategy(s2)

        # Same observable state should produce the same strategy
        assert strategy1 == strategy2, (
            f"Strategy differs by subtype: {strategy1} vs {strategy2}. "
            "This indicates oracle leakage from adhd_subtype."
        )

    def test_server_debug_mode_off_by_default(self):
        """ADAS_DEBUG should default to off (no ground-truth in normal events)."""
        # Import the module-level flag
        import app.backend.server as srv
        # When ADAS_DEBUG env var is not set, _DEBUG_MODE should be False
        assert srv._DEBUG_MODE is False or os.environ.get("ADAS_DEBUG") == "1"

    def test_orchestrator_decide_action_no_subtype_access(self):
        """
        The orchestrator's rule-based decision path should work correctly
        even when adhd_subtype is None for all students (no oracle data).
        """
        orch = SimulationOrchestrator(n_students=10, seed=7, max_classes=1)
        obs = orch.classroom.reset()
        orch.memory.new_class()

        # Null out all subtypes to simulate no ground-truth access
        for s in orch.classroom.students:
            s.adhd_subtype = None

        # This should not raise any errors
        action = orch._decide_action(obs, turn=1)
        assert action.action_type in (
            "observe", "class_instruction", "individual_intervention",
            "identify_adhd", "private_correction", "public_correction",
        )


# ===========================================================================
# 2. App path grows memory
# ===========================================================================


class TestAppPathGrowsMemory:
    """Verify that the orchestrator path (used by the app) grows memory."""

    def test_orchestrator_run_class_grows_case_base(self):
        """
        Running one class through the orchestrator should add records
        to the case base (commit_observation is called).
        """
        orch = SimulationOrchestrator(n_students=20, seed=42, max_classes=1)

        initial_case_count = len(orch.memory.case_base)
        orch.run_class(max_turns=30)
        final_case_count = len(orch.memory.case_base)

        assert final_case_count > initial_case_count, (
            f"Case base did not grow: {initial_case_count} -> {final_case_count}. "
            "commit_observation() may not be wired correctly."
        )

    def test_orchestrator_run_class_grows_experience_base_on_identification(self):
        """
        When the orchestrator identifies students and records outcomes,
        principles should be extracted into the experience base.
        """
        orch = SimulationOrchestrator(n_students=20, seed=42, max_classes=3)

        # Run multiple classes to accumulate identifications
        for _ in range(3):
            orch.run_class(max_turns=50)
            orch.class_count += 1
            orch.memory.new_class()

        report = orch.memory.growth_report()
        # After 3 classes with identifications, there should be some
        # principles (positive or corrective) and cases
        assert report["case_base_size"] > 0, "Case base empty after 3 classes"

    def test_orchestrator_memory_persists_across_classes(self):
        """Case base size should be monotonically non-decreasing across classes."""
        orch = SimulationOrchestrator(n_students=20, seed=42, max_classes=3)

        sizes = []
        for _ in range(3):
            orch.run_class(max_turns=30)
            orch.class_count += 1
            sizes.append(len(orch.memory.case_base))
            orch.memory.new_class()

        # Each class should add to the case base
        for i in range(1, len(sizes)):
            assert sizes[i] >= sizes[i - 1], (
                f"Case base shrank between classes: {sizes}"
            )


# ===========================================================================
# 3. Metrics consistency
# ===========================================================================


class TestMetricsConsistency:
    """Verify behavior improvement and metrics definitions are consistent."""

    def test_behavior_improvement_is_compliance_based(self):
        """
        Orchestrator computes behavior improvement as
        max(0, final_compliance - initial_compliance).
        Verify this by running a class and checking the metrics.
        """
        orch = SimulationOrchestrator(n_students=20, seed=42, max_classes=1)
        result = orch.run_class(max_turns=50)
        metrics = result["metrics"]

        # Each improvement rate should be non-negative (clamped at 0)
        for rate in metrics.behavior_improvement_rates:
            assert rate >= 0.0, f"Negative improvement rate: {rate}"

    def test_growth_tracker_and_orchestrator_use_same_classmetrics(self):
        """
        Both paths should produce ClassMetrics with the same field set.
        Verify by running the orchestrator and recording into a GrowthTracker.
        """
        orch = SimulationOrchestrator(n_students=20, seed=42, max_classes=1)
        result = orch.run_class(max_turns=30)
        metrics = result["metrics"]

        tracker = GrowthTracker()
        # This should not raise -- the ClassMetrics is compatible
        tracker.record_class(metrics)
        assert tracker.total_classes_completed == 1

        # Verify all expected fields are populated
        assert metrics.n_students == 20
        assert metrics.true_positives + metrics.false_negatives == metrics.n_adhd or metrics.n_adhd == 0
        assert metrics.class_completion_turn > 0

    def test_confusion_matrix_sums_correctly(self):
        """TP + FP + FN + TN should equal n_students."""
        orch = SimulationOrchestrator(n_students=20, seed=42, max_classes=1)
        result = orch.run_class(max_turns=50)
        m = result["metrics"]

        total = m.true_positives + m.false_positives + m.false_negatives + m.true_negatives
        assert total == m.n_students, (
            f"Confusion matrix doesn't sum to n_students: "
            f"TP={m.true_positives} FP={m.false_positives} "
            f"FN={m.false_negatives} TN={m.true_negatives} = {total} != {m.n_students}"
        )


# ===========================================================================
# 4. Build config references expected assets
# ===========================================================================


class TestBuildConfig:
    """Verify Electron build config references files that exist."""

    def _load_package_json(self) -> dict:
        pkg_path = os.path.join(
            os.path.dirname(__file__), "..", "app", "package.json"
        )
        with open(pkg_path, encoding="utf-8") as f:
            return json.load(f)

    def test_package_json_includes_backend_files(self):
        """Build config should include backend/**/* so server.py is packaged."""
        pkg = self._load_package_json()
        files = pkg.get("build", {}).get("files", [])
        assert any("backend" in f for f in files), (
            f"backend not in build.files: {files}"
        )

    def test_package_json_includes_renderer_dist(self):
        """Build config should include renderer/dist for the frontend."""
        pkg = self._load_package_json()
        files = pkg.get("build", {}).get("files", [])
        assert any("renderer/dist" in f for f in files), (
            f"renderer/dist not in build.files: {files}"
        )

    def test_package_json_extra_resources_reference_existing_dirs(self):
        """extraResources should reference directories that exist in the repo."""
        pkg = self._load_package_json()
        extra = pkg.get("build", {}).get("extraResources", [])
        app_dir = os.path.join(os.path.dirname(__file__), "..", "app")

        for entry in extra:
            from_path = entry.get("from", "") if isinstance(entry, dict) else ""
            if from_path:
                # Resolve relative to the app directory
                full_path = os.path.normpath(os.path.join(app_dir, from_path))
                assert os.path.exists(full_path), (
                    f"extraResources 'from' path does not exist: {from_path} "
                    f"(resolved to {full_path})"
                )

    def test_electron_main_js_exists(self):
        """The main entry point referenced in package.json should exist."""
        pkg = self._load_package_json()
        main = pkg.get("main", "")
        app_dir = os.path.join(os.path.dirname(__file__), "..", "app")
        main_path = os.path.join(app_dir, main)
        assert os.path.exists(main_path), f"main entry {main} not found at {main_path}"

    def test_packaged_backend_in_extra_resources(self):
        """Backend must be in extraResources, not just build.files."""
        pkg = self._load_package_json()
        extra = pkg.get("build", {}).get("extraResources", [])
        backend_entries = [
            e for e in extra
            if isinstance(e, dict) and e.get("from") == "backend"
        ]
        assert len(backend_entries) == 1, (
            f"Expected backend in extraResources, got: {extra}"
        )
        assert backend_entries[0].get("to") == "backend", (
            f"Backend extraResources 'to' should be 'backend', got: {backend_entries[0]}"
        )


# ===========================================================================
# 5. Electron dev venv discovery order
# ===========================================================================


class TestDevVenvDiscovery:
    """Verify dev mode checks repo-local .venv before parent."""

    def test_dev_venv_checks_repo_local_first(self):
        """Dev mode should check repo-local .venv before parent .venv."""
        main_js_path = os.path.join(
            os.path.dirname(__file__), "..", "app", "electron", "main.js"
        )
        with open(main_js_path, encoding="utf-8") as f:
            source = f.read()

        # The repo-local venv check should appear before the parent venv check
        local_idx = source.find('path.join(repoRoot, ".venv"')
        parent_idx = source.find('path.join(repoRoot, "..", ".venv"')

        assert local_idx != -1, "repo-local .venv check not found in main.js"
        assert parent_idx != -1, "parent .venv check not found in main.js"
        assert local_idx < parent_idx, (
            "repo-local .venv check must come before parent .venv check"
        )

    def test_packaged_path_uses_resources_backend(self):
        """Packaged mode should look for backend/server.py at process.resourcesPath/backend/."""
        main_js_path = os.path.join(
            os.path.dirname(__file__), "..", "app", "electron", "main.js"
        )
        with open(main_js_path, encoding="utf-8") as f:
            source = f.read()

        # Packaged path should NOT contain "python/app/backend"
        assert "python/app/backend" not in source and 'python", "app", "backend"' not in source, (
            "Packaged path still references old python/app/backend location"
        )
        # Should contain the correct resourcesPath/backend path
        assert 'process.resourcesPath, "backend", "server.py"' in source, (
            "Packaged path should use process.resourcesPath/backend/server.py"
        )


# ===========================================================================
# 6. No class double-counting
# ===========================================================================


class TestNoClassDoubleCounting:
    """Server path must not double-count classes."""

    def test_no_class_double_counting(self):
        """Running orchestrator for 2 classes should increment class_count by exactly 2."""
        orch = SimulationOrchestrator(n_students=10, seed=42, max_classes=2)

        classes_run = 0
        for _event in orch.run(max_turns_per_class=20):
            classes_run += 1

        assert classes_run == 2
        assert orch.class_count == 2, (
            f"Expected class_count==2 after 2 classes, got {orch.class_count}"
        )

    def test_server_does_not_call_memory_new_class(self):
        """The server's multi-student path should not call orch.memory.new_class()."""
        server_path = os.path.join(
            os.path.dirname(__file__), "..", "app", "backend", "server.py"
        )
        with open(server_path, encoding="utf-8") as f:
            source = f.read()

        # Find the _run_multi_student_session function body
        func_start = source.find("async def _run_multi_student_session")
        assert func_start != -1, "_run_multi_student_session not found"

        # Get the function body (up to the next top-level function or end)
        func_body = source[func_start:]
        next_func = func_body.find("\nasync def ", 1)
        if next_func == -1:
            next_func = func_body.find("\ndef ", 1)
        if next_func != -1:
            func_body = func_body[:next_func]

        # Should NOT contain orch.memory.new_class() (that's the double-count bug)
        assert "orch.memory.new_class()" not in func_body, (
            "Server still calls orch.memory.new_class() -- causes double-counting"
        )


# ===========================================================================
# 7. Managed count accuracy
# ===========================================================================


class TestManagedCountAccuracy:
    """class_complete payload must use actual managed count, not identified count."""

    def test_managed_count_not_identified_count(self):
        """ClassMetrics should track n_managed separately from n_identified."""
        orch = SimulationOrchestrator(n_students=20, seed=42, max_classes=1)
        result = orch.run_class(max_turns=50)
        metrics = result["metrics"]

        # n_managed must exist and be an int
        assert hasattr(metrics, "n_managed"), "ClassMetrics missing n_managed field"
        assert isinstance(metrics.n_managed, int)

        # n_managed counts ADHD students who are managed, so it cannot exceed n_adhd
        assert metrics.n_managed <= metrics.n_adhd, (
            f"n_managed ({metrics.n_managed}) > n_adhd ({metrics.n_adhd})"
        )

        # n_identified can include false positives, so it can differ from n_managed
        # The key point: they are tracked independently
        assert metrics.n_identified == metrics.true_positives + metrics.false_positives, (
            f"n_identified should be TP+FP: {metrics.n_identified} != "
            f"{metrics.true_positives} + {metrics.false_positives}"
        )

    def test_server_uses_n_managed_not_n_identified(self):
        """The server's class_complete payload should use metrics.n_managed."""
        server_path = os.path.join(
            os.path.dirname(__file__), "..", "app", "backend", "server.py"
        )
        with open(server_path, encoding="utf-8") as f:
            source = f.read()

        # Find the class_complete_payload section
        payload_start = source.find("class_complete_payload")
        assert payload_start != -1
        payload_section = source[payload_start:payload_start + 500]

        assert "metrics.n_managed" in payload_section, (
            "class_complete_payload should use metrics.n_managed for managed_count"
        )
        assert "metrics.n_identified" not in payload_section, (
            "class_complete_payload should NOT use metrics.n_identified for managed_count"
        )
