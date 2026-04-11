"""Tests for Phase 6 slice 10: retrieval-noise plumbing through
OrchestratorV2 and the default calibration setup path.

Covers:
  - OrchestratorV2(..., retrieval_noise_config=cfg) installs the
    config on the underlying TeacherMemory instance
  - Default OrchestratorV2(...) construction leaves memory in the
    default (no-op) retrieval-noise state
  - DefaultEvaluator.retrieval_noise_config propagates into every
    orchestrator it constructs
  - build_default_evaluator forwards retrieval_noise_config
  - build_default_autoresearch_setup forwards retrieval_noise_config
    into the evaluator it builds
  - Fixed-seed determinism of a real run_class() under the
    plumbed retrieval-noise path
  - No regression: default path still produces a usable run
"""

import pytest

from src.simulation.teacher_memory import (
    RetrievalNoiseConfig,
    TeacherMemory,
)
from src.simulation.orchestrator_v2 import OrchestratorV2


# ---------------------------------------------------------------------------
# OrchestratorV2 constructor wiring
# ---------------------------------------------------------------------------


def test_orchestrator_constructor_installs_retrieval_noise_config():
    cfg = RetrievalNoiseConfig(dropout_prob=0.5, similarity_jitter=0.1)
    orch = OrchestratorV2(
        n_students=5, seed=7, retrieval_noise_config=cfg,
    )
    assert orch.memory.retrieval_noise_config is cfg
    assert not orch.memory.retrieval_noise_config.is_disabled


def test_orchestrator_default_retrieval_noise_is_disabled():
    orch = OrchestratorV2(n_students=5, seed=7)
    assert orch.memory.retrieval_noise_config.is_disabled


def test_orchestrator_default_construction_still_works():
    """Default construction with no retrieval_noise_config must
    not crash and must produce a memory with default no-op
    config."""
    orch = OrchestratorV2()
    assert isinstance(orch.memory.retrieval_noise_config, RetrievalNoiseConfig)
    assert orch.memory.retrieval_noise_config.is_disabled


def test_orchestrator_retrieval_noise_propagates_to_case_base_queries():
    """Full-dropout config installed via the constructor must
    actually reach TeacherMemory.retrieve_similar_cases."""
    from src.simulation.teacher_memory import HYPERACTIVITY_BEHAVIORS
    cfg = RetrievalNoiseConfig(dropout_prob=1.0)
    orch = OrchestratorV2(
        n_students=5, seed=1, retrieval_noise_config=cfg,
    )
    # Seed a few records so retrieval has something to drop.
    for i in range(3):
        orch.memory.observe(f"S{i:02d}", list(HYPERACTIVITY_BEHAVIORS[:3]))
        orch.memory.commit_observation(f"S{i:02d}", outcome="positive")
    results = orch.memory.retrieve_similar_cases(
        list(HYPERACTIVITY_BEHAVIORS[:3]), top_k=5,
    )
    # Full dropout must empty the result.
    assert results == []


# ---------------------------------------------------------------------------
# Default setup path plumbing
# ---------------------------------------------------------------------------


def test_build_default_evaluator_forwards_retrieval_noise_config():
    from src.calibration.applier import build_default_evaluator
    cfg = RetrievalNoiseConfig(dropout_prob=0.3, similarity_jitter=0.1)
    ev = build_default_evaluator(
        n_classes=1, max_turns=10, seed=1,
        retrieval_noise_config=cfg,
    )
    assert ev.retrieval_noise_config is cfg


def test_build_default_evaluator_default_retrieval_noise_is_none():
    from src.calibration.applier import build_default_evaluator
    ev = build_default_evaluator(n_classes=1, max_turns=10, seed=1)
    assert ev.retrieval_noise_config is None


def test_build_default_autoresearch_setup_forwards_retrieval_noise_config():
    from src.calibration import build_default_autoresearch_setup
    cfg = RetrievalNoiseConfig(dropout_prob=0.2, similarity_jitter=0.15)
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=20,
        retrieval_noise_config=cfg,
    )
    assert setup.evaluator.retrieval_noise_config is cfg


def test_build_default_autoresearch_setup_default_path_unchanged():
    from src.calibration import build_default_autoresearch_setup
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=1, n_classes=1, max_turns=20,
    )
    assert setup.evaluator.retrieval_noise_config is None


# ---------------------------------------------------------------------------
# End-to-end: run_class() with plumbed retrieval noise is deterministic
# ---------------------------------------------------------------------------


def test_run_class_with_plumbed_retrieval_noise_is_deterministic():
    def run_one() -> list:
        cfg = RetrievalNoiseConfig(
            dropout_prob=0.3, similarity_jitter=0.1,
        )
        orch = OrchestratorV2(
            n_students=5, max_classes=1, seed=42,
            retrieval_noise_config=cfg,
        )
        orch.classroom.MAX_TURNS = 60
        orch.run_class()
        return [
            (r.student_id, r.turn, r.outcome, tuple(r.observed_behaviors))
            for r in orch.memory.case_base._records
        ]

    a = run_one()
    b = run_one()
    assert a == b
    assert a  # non-empty


def test_default_evaluator_evaluate_runs_under_plumbed_retrieval_noise():
    """Real DefaultEvaluator.evaluate path must execute end-to-end
    when the plumbed retrieval-noise config is active."""
    from src.calibration.applier import build_default_evaluator
    cfg = RetrievalNoiseConfig(dropout_prob=0.25, similarity_jitter=0.1)
    ev = build_default_evaluator(
        n_classes=1, max_turns=15, seed=7,
        retrieval_noise_config=cfg,
    )
    bundle, loss = ev.evaluate(config={})
    assert loss.total != float("inf")
    assert len(bundle.histories) == 1
