"""Tests for Phase 6 slice 14: checkpoint/resume handling for
the persisted noise configs.

Covers:
  - Static restorers reconstruct RetrievalNoiseConfig / TeacherNoiseConfig
  - Restorers handle null / missing / malformed blobs gracefully
  - run(resume=True) adopts checkpoint configs when evaluator is None
  - Explicit constructor-supplied configs win over checkpoint
  - resume → save → resume preserves config fields across cycles
  - summary/report on the resumed setup reflect the effective config
  - Full pytest still passes
"""

import json

import pytest

from src.simulation.teacher_memory import RetrievalNoiseConfig
from src.simulation.teacher_noise import TeacherNoiseConfig
from src.calibration import build_default_autoresearch_setup
from src.calibration.orchestrator import AutoresearchOrchestrator


# ---------------------------------------------------------------------------
# Static restorers
# ---------------------------------------------------------------------------


def test_restore_retrieval_noise_from_dict():
    cfg = AutoresearchOrchestrator._restore_retrieval_noise_from_checkpoint(
        {"dropout_prob": 0.3, "similarity_jitter": 0.1}
    )
    assert isinstance(cfg, RetrievalNoiseConfig)
    assert cfg.dropout_prob == 0.3
    assert cfg.similarity_jitter == 0.1


def test_restore_retrieval_noise_handles_null():
    assert (
        AutoresearchOrchestrator._restore_retrieval_noise_from_checkpoint(None)
        is None
    )


def test_restore_retrieval_noise_handles_missing_keys():
    cfg = AutoresearchOrchestrator._restore_retrieval_noise_from_checkpoint({})
    assert isinstance(cfg, RetrievalNoiseConfig)
    assert cfg.dropout_prob == 0.0
    assert cfg.similarity_jitter == 0.0


def test_restore_retrieval_noise_handles_malformed_scalar():
    """Non-numeric input should NOT raise; return None."""
    cfg = AutoresearchOrchestrator._restore_retrieval_noise_from_checkpoint(
        {"dropout_prob": "nope", "similarity_jitter": 0.1}
    )
    assert cfg is None


def test_restore_teacher_noise_from_dict():
    cfg = AutoresearchOrchestrator._restore_teacher_noise_from_checkpoint(
        {
            "observation_dropout_prob": 0.25,
            "observation_confusion_prob": 0.1,
        }
    )
    assert isinstance(cfg, TeacherNoiseConfig)
    assert cfg.observation_dropout_prob == 0.25
    assert cfg.observation_confusion_prob == 0.1


def test_restore_teacher_noise_handles_null_and_missing():
    assert (
        AutoresearchOrchestrator._restore_teacher_noise_from_checkpoint(None)
        is None
    )
    cfg = AutoresearchOrchestrator._restore_teacher_noise_from_checkpoint({})
    assert isinstance(cfg, TeacherNoiseConfig)
    assert cfg.observation_dropout_prob == 0.0
    assert cfg.observation_confusion_prob == 0.0


# ---------------------------------------------------------------------------
# Resume precedence
# ---------------------------------------------------------------------------


def _write_initial_checkpoint(tmp_path, *, rn=None, tn=None, n_starts=2):
    setup = build_default_autoresearch_setup(
        n_iterations=1, n_starts=n_starts, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
        retrieval_noise_config=rn,
        teacher_noise_config=tn,
    )
    setup.orchestrator.run()
    return setup


def test_resume_inherits_configs_from_checkpoint(tmp_path):
    """Original run has both configs; resume run supplies neither
    and inherits from checkpoint."""
    rn = RetrievalNoiseConfig(dropout_prob=0.2, similarity_jitter=0.05)
    tn = TeacherNoiseConfig(
        observation_dropout_prob=0.15,
        observation_confusion_prob=0.05,
    )
    _write_initial_checkpoint(tmp_path, rn=rn, tn=tn)

    setup2 = build_default_autoresearch_setup(
        n_iterations=1, n_starts=3, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
    )
    assert setup2.evaluator.retrieval_noise_config is None
    assert setup2.evaluator.teacher_noise_config is None
    setup2.orchestrator.run(resume=True)
    # After resume, the evaluator should carry the configs
    # reconstructed from the checkpoint.
    assert isinstance(setup2.evaluator.retrieval_noise_config, RetrievalNoiseConfig)
    assert setup2.evaluator.retrieval_noise_config.dropout_prob == 0.2
    assert setup2.evaluator.retrieval_noise_config.similarity_jitter == 0.05
    assert isinstance(setup2.evaluator.teacher_noise_config, TeacherNoiseConfig)
    assert setup2.evaluator.teacher_noise_config.observation_dropout_prob == 0.15
    assert setup2.evaluator.teacher_noise_config.observation_confusion_prob == 0.05


def test_resume_explicit_retrieval_override_wins(tmp_path):
    rn_original = RetrievalNoiseConfig(dropout_prob=0.2)
    _write_initial_checkpoint(tmp_path, rn=rn_original)

    override = RetrievalNoiseConfig(dropout_prob=0.5, similarity_jitter=0.2)
    setup2 = build_default_autoresearch_setup(
        n_iterations=1, n_starts=3, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
        retrieval_noise_config=override,
    )
    setup2.orchestrator.run(resume=True)
    # Explicit override wins.
    assert setup2.evaluator.retrieval_noise_config is override


def test_resume_explicit_teacher_override_wins(tmp_path):
    tn_original = TeacherNoiseConfig(observation_dropout_prob=0.1)
    _write_initial_checkpoint(tmp_path, tn=tn_original)

    override = TeacherNoiseConfig(
        observation_dropout_prob=0.4,
        observation_confusion_prob=0.2,
    )
    setup2 = build_default_autoresearch_setup(
        n_iterations=1, n_starts=3, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
        teacher_noise_config=override,
    )
    setup2.orchestrator.run(resume=True)
    assert setup2.evaluator.teacher_noise_config is override


def test_resume_partial_override(tmp_path):
    """Evaluator provides only retrieval override — teacher noise
    should still inherit from checkpoint."""
    rn_original = RetrievalNoiseConfig(dropout_prob=0.2)
    tn_original = TeacherNoiseConfig(observation_dropout_prob=0.1)
    _write_initial_checkpoint(tmp_path, rn=rn_original, tn=tn_original)

    override_rn = RetrievalNoiseConfig(dropout_prob=0.5)
    setup2 = build_default_autoresearch_setup(
        n_iterations=1, n_starts=3, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
        retrieval_noise_config=override_rn,
    )
    setup2.orchestrator.run(resume=True)
    assert setup2.evaluator.retrieval_noise_config is override_rn
    # Teacher noise inherited from checkpoint.
    assert isinstance(setup2.evaluator.teacher_noise_config, TeacherNoiseConfig)
    assert setup2.evaluator.teacher_noise_config.observation_dropout_prob == 0.1


# ---------------------------------------------------------------------------
# Round-trip preservation
# ---------------------------------------------------------------------------


def test_resume_save_resume_preserves_configs(tmp_path):
    """A resume → save → resume cycle must not collapse the
    config fields to null or drift their values."""
    rn = RetrievalNoiseConfig(dropout_prob=0.2, similarity_jitter=0.05)
    tn = TeacherNoiseConfig(
        observation_dropout_prob=0.15,
        observation_confusion_prob=0.05,
    )
    # Cycle 1: original run
    _write_initial_checkpoint(tmp_path, rn=rn, tn=tn, n_starts=2)
    payload1 = json.loads(
        (tmp_path / "orchestrator_checkpoint.json").read_text()
    )
    assert payload1["retrieval_noise_config"] == {
        "dropout_prob": 0.2,
        "similarity_jitter": 0.05,
    }
    assert payload1["teacher_noise_config"] == {
        "observation_dropout_prob": 0.15,
        "observation_confusion_prob": 0.05,
    }

    # Cycle 2: resume and save again
    setup2 = build_default_autoresearch_setup(
        n_iterations=1, n_starts=3, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
    )
    setup2.orchestrator.run(resume=True)
    payload2 = json.loads(
        (tmp_path / "orchestrator_checkpoint.json").read_text()
    )
    assert payload2["retrieval_noise_config"] == payload1["retrieval_noise_config"]
    assert payload2["teacher_noise_config"] == payload1["teacher_noise_config"]

    # Cycle 3: resume again from the new checkpoint
    setup3 = build_default_autoresearch_setup(
        n_iterations=1, n_starts=4, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
    )
    setup3.orchestrator.run(resume=True)
    payload3 = json.loads(
        (tmp_path / "orchestrator_checkpoint.json").read_text()
    )
    assert payload3["retrieval_noise_config"] == payload1["retrieval_noise_config"]
    assert payload3["teacher_noise_config"] == payload1["teacher_noise_config"]


# ---------------------------------------------------------------------------
# Summary / report after resume
# ---------------------------------------------------------------------------


def test_resumed_setup_summary_reflects_effective_configs(tmp_path):
    rn = RetrievalNoiseConfig(dropout_prob=0.2, similarity_jitter=0.05)
    tn = TeacherNoiseConfig(
        observation_dropout_prob=0.15,
        observation_confusion_prob=0.05,
    )
    _write_initial_checkpoint(tmp_path, rn=rn, tn=tn)

    setup2 = build_default_autoresearch_setup(
        n_iterations=1, n_starts=3, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
    )
    # Before resume, summary shows disabled (no config on evaluator yet).
    pre_summary = setup2.summary()
    assert "retrieval_noise=disabled" in pre_summary
    assert "teacher_noise=disabled" in pre_summary

    setup2.orchestrator.run(resume=True)

    # After resume, summary reflects the reconstructed configs.
    post_summary = setup2.summary()
    assert "retrieval_noise=dropout=0.200,jitter=0.050" in post_summary
    assert "teacher_noise=dropout=0.150,confusion=0.050" in post_summary

    post_report = setup2.report()
    assert "dropout=0.200,jitter=0.050" in post_report
    assert "dropout=0.150,confusion=0.050" in post_report


# ---------------------------------------------------------------------------
# Malformed checkpoint does not crash resume
# ---------------------------------------------------------------------------


def test_malformed_checkpoint_noise_does_not_crash_resume(tmp_path):
    """If the persisted noise config is garbage, resume should
    proceed with the evaluator's current config (None in this
    case) — not crash."""
    rn = RetrievalNoiseConfig(dropout_prob=0.2)
    _write_initial_checkpoint(tmp_path, rn=rn)
    ckpt_path = tmp_path / "orchestrator_checkpoint.json"
    payload = json.loads(ckpt_path.read_text())
    payload["retrieval_noise_config"] = {"dropout_prob": "bad"}
    payload["teacher_noise_config"] = "not a dict"
    ckpt_path.write_text(json.dumps(payload))

    setup2 = build_default_autoresearch_setup(
        n_iterations=1, n_starts=3, n_classes=1, max_turns=10,
        enforce_constraints=False, results_dir=tmp_path,
    )
    setup2.orchestrator.run(resume=True)  # must not raise
    # Malformed blobs → evaluator stays at None.
    assert setup2.evaluator.retrieval_noise_config is None
    assert setup2.evaluator.teacher_noise_config is None
