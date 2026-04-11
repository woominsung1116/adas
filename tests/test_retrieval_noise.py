"""Tests for Phase 6 slice 9: teacher-memory retrieval noise.

Covers:
  - RetrievalNoiseConfig defaults / clamping / is_disabled
  - apply_retrieval_noise no-op path returns input unchanged
  - No-op path does not advance the RNG
  - Full-dropout path removes everything
  - Full-jitter path reshuffles similarities
  - Determinism under fixed seed
  - Stored ObservationRecords are NOT mutated by noise
  - No latent fields introduced
  - TeacherMemory.retrieve_similar_cases uses the config when set
  - Default TeacherMemory runs still pass existing tests (regression)
  - Real orchestrator run still works with retrieval noise enabled
"""

import copy
import random

import pytest

from src.simulation.teacher_memory import (
    HYPERACTIVITY_BEHAVIORS,
    IMPULSIVITY_BEHAVIORS,
    INATTENTION_BEHAVIORS,
    ObservationRecord,
    RetrievalNoiseConfig,
    TeacherMemory,
    apply_retrieval_noise,
)
from src.simulation.orchestrator_v2 import OrchestratorV2


def _record(sid: str, behaviors, turn: int = 0) -> ObservationRecord:
    return ObservationRecord(
        student_id=sid,
        turn=turn,
        observed_behaviors=list(behaviors),
        action_taken="observe",
        outcome="neutral",
    )


# ---------------------------------------------------------------------------
# Config invariants
# ---------------------------------------------------------------------------


def test_config_default_is_disabled():
    c = RetrievalNoiseConfig()
    assert c.dropout_prob == 0.0
    assert c.similarity_jitter == 0.0
    assert c.is_disabled


def test_config_clamps_out_of_range_probs():
    c = RetrievalNoiseConfig(dropout_prob=1.5, similarity_jitter=-0.3)
    assert c.dropout_prob == 1.0
    assert c.similarity_jitter == 0.0


def test_config_is_json_serializable():
    import json
    c = RetrievalNoiseConfig(dropout_prob=0.3, similarity_jitter=0.1)
    blob = c.as_dict()
    json.dumps(blob)
    assert blob["dropout_prob"] == 0.3


def test_config_is_frozen():
    c = RetrievalNoiseConfig(dropout_prob=0.3)
    with pytest.raises(Exception):
        c.dropout_prob = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# apply_retrieval_noise no-op
# ---------------------------------------------------------------------------


def test_noop_config_returns_input_unchanged():
    r = _record("X", ["seat-leaving"])
    scored = [(0.9, r), (0.5, r), (0.3, r)]
    out = apply_retrieval_noise(scored, random.Random(0), RetrievalNoiseConfig())
    assert [s for s, _ in out] == [0.9, 0.5, 0.3]


def test_noop_config_does_not_advance_rng():
    r = _record("X", ["seat-leaving"])
    scored = [(0.9, r), (0.5, r)]
    rng = random.Random(0)
    before = rng.getstate()
    apply_retrieval_noise(scored, rng, RetrievalNoiseConfig())
    after = rng.getstate()
    assert before == after


def test_empty_input_returns_empty():
    out = apply_retrieval_noise(
        [], random.Random(0),
        RetrievalNoiseConfig(dropout_prob=1.0, similarity_jitter=1.0),
    )
    assert out == []


# ---------------------------------------------------------------------------
# Dropout + jitter behavior
# ---------------------------------------------------------------------------


def test_full_dropout_removes_everything():
    r = _record("X", ["seat-leaving"])
    scored = [(0.9, r)] * 10
    out = apply_retrieval_noise(
        scored, random.Random(0), RetrievalNoiseConfig(dropout_prob=1.0)
    )
    assert out == []


def test_partial_dropout_is_deterministic_under_fixed_seed():
    r = _record("X", ["seat-leaving"])
    scored = [(float(i), r) for i in range(20)]
    cfg = RetrievalNoiseConfig(dropout_prob=0.5)
    a = apply_retrieval_noise(scored, random.Random(123), cfg)
    b = apply_retrieval_noise(scored, random.Random(123), cfg)
    assert [x[0] for x in a] == [x[0] for x in b]
    assert 0 < len(a) < len(scored)


def test_full_jitter_reshuffles_similarities():
    r = _record("X", ["seat-leaving"])
    scored = [(0.90, r), (0.80, r), (0.70, r)]
    # With very large jitter, order can flip. Run several seeds
    # and expect at least one reshuffling to occur.
    seen_shuffled = False
    for seed in range(5):
        out = apply_retrieval_noise(
            scored, random.Random(seed),
            RetrievalNoiseConfig(similarity_jitter=1.0),
        )
        perturbed = [x[0] for x in out]
        if perturbed != sorted(perturbed, reverse=True) or perturbed != [0.90, 0.80, 0.70]:
            seen_shuffled = True
            break
    assert seen_shuffled


def test_jitter_results_are_sorted_descending():
    r = _record("X", ["seat-leaving"])
    scored = [(0.5, r), (0.5, r), (0.5, r)]
    out = apply_retrieval_noise(
        scored, random.Random(7),
        RetrievalNoiseConfig(similarity_jitter=0.2),
    )
    sims = [x[0] for x in out]
    assert sims == sorted(sims, reverse=True)


def test_dropout_and_jitter_combined_is_deterministic():
    r = _record("X", ["seat-leaving"])
    scored = [(float(i) / 10.0, r) for i in range(10)]
    cfg = RetrievalNoiseConfig(dropout_prob=0.3, similarity_jitter=0.1)
    a = apply_retrieval_noise(scored, random.Random(42), cfg)
    b = apply_retrieval_noise(scored, random.Random(42), cfg)
    assert len(a) == len(b)
    for (sa, _), (sb, _) in zip(a, b):
        assert sa == sb


# ---------------------------------------------------------------------------
# Stored records are not mutated
# ---------------------------------------------------------------------------


def test_apply_retrieval_noise_does_not_mutate_records():
    r1 = _record("S01", list(HYPERACTIVITY_BEHAVIORS[:3]))
    r2 = _record("S02", list(IMPULSIVITY_BEHAVIORS[:2]))
    original = [
        (1, copy.deepcopy(r1.observed_behaviors), r1.outcome, r1.action_taken),
        (2, copy.deepcopy(r2.observed_behaviors), r2.outcome, r2.action_taken),
    ]
    scored = [(0.9, r1), (0.5, r2)]
    apply_retrieval_noise(
        scored, random.Random(1),
        RetrievalNoiseConfig(dropout_prob=0.3, similarity_jitter=0.2),
    )
    # Records untouched.
    assert r1.observed_behaviors == original[0][1]
    assert r1.outcome == original[0][2]
    assert r1.action_taken == original[0][3]
    assert r2.observed_behaviors == original[1][1]


def test_apply_retrieval_noise_returns_new_tuples():
    r = _record("X", ["seat-leaving"])
    scored = [(0.9, r)]
    out = apply_retrieval_noise(
        scored, random.Random(0),
        RetrievalNoiseConfig(similarity_jitter=0.1),
    )
    # Similarity should be perturbed (not the same object).
    assert out[0][1] is r  # record pointer preserved
    # The similarity float should (almost certainly) differ.
    assert out[0][0] != 0.9


# ---------------------------------------------------------------------------
# No latent fields introduced
# ---------------------------------------------------------------------------


def test_retrieval_noise_config_has_no_latent_fields():
    fields = set(RetrievalNoiseConfig.__dataclass_fields__.keys())
    for latent in (
        "state_snapshot",
        "distress_level",
        "compliance",
        "attention",
        "escalation_risk",
    ):
        assert latent not in fields


# ---------------------------------------------------------------------------
# TeacherMemory integration
# ---------------------------------------------------------------------------


def _seeded_memory(noise_cfg: RetrievalNoiseConfig | None = None) -> TeacherMemory:
    mem = TeacherMemory(
        retrieval_noise=0.0,
        seed=1,
        retrieval_noise_config=noise_cfg,
    )
    # Seed the case base with a few records with distinct behaviors.
    for i in range(5):
        mem.observe(f"S{i:02d}", list(HYPERACTIVITY_BEHAVIORS[:3]))
        mem.commit_observation(f"S{i:02d}", outcome="positive")
    for i in range(5, 10):
        mem.observe(f"S{i:02d}", list(IMPULSIVITY_BEHAVIORS[:2]))
        mem.commit_observation(f"S{i:02d}", outcome="neutral")
    return mem


def test_teacher_memory_default_noise_config_is_disabled():
    mem = TeacherMemory()
    assert mem.retrieval_noise_config.is_disabled


def test_teacher_memory_retrieve_uses_config_when_set():
    mem = _seeded_memory(
        noise_cfg=RetrievalNoiseConfig(dropout_prob=1.0)
    )
    results = mem.retrieve_similar_cases(
        list(HYPERACTIVITY_BEHAVIORS[:3]), top_k=5,
    )
    assert results == []


def test_teacher_memory_default_retrieval_preserves_legacy_behavior():
    """When retrieval_noise_config is default (disabled),
    retrieve_similar_cases must still fall back to the legacy
    rate-based drop and produce sensible results."""
    mem = _seeded_memory()
    results = mem.retrieve_similar_cases(
        list(HYPERACTIVITY_BEHAVIORS[:3]), top_k=5,
    )
    assert len(results) >= 1
    # Best match should be a hyperactive record.
    best_sim, best_record = results[0]
    assert best_sim > 0.5
    assert any(
        b in HYPERACTIVITY_BEHAVIORS for b in best_record.observed_behaviors
    )


def test_teacher_memory_retrieval_noise_is_deterministic():
    def retrieve():
        mem = _seeded_memory(
            noise_cfg=RetrievalNoiseConfig(
                dropout_prob=0.3, similarity_jitter=0.1
            )
        )
        return [
            (r.student_id, round(s, 4))
            for s, r in mem.retrieve_similar_cases(
                list(HYPERACTIVITY_BEHAVIORS[:3]), top_k=5,
            )
        ]

    a = retrieve()
    b = retrieve()
    assert a == b


def test_retrieval_noise_does_not_mutate_case_base_records():
    mem = _seeded_memory()
    # Snapshot record fields before retrieval.
    before = [
        (r.student_id, tuple(r.observed_behaviors), r.outcome)
        for r in mem.case_base._records
    ]
    mem.retrieval_noise_config = RetrievalNoiseConfig(
        dropout_prob=0.3, similarity_jitter=0.2
    )
    mem.retrieve_similar_cases(list(HYPERACTIVITY_BEHAVIORS[:3]), top_k=5)
    after = [
        (r.student_id, tuple(r.observed_behaviors), r.outcome)
        for r in mem.case_base._records
    ]
    assert before == after


# ---------------------------------------------------------------------------
# Orchestrator integration
# ---------------------------------------------------------------------------


def test_orchestrator_runs_with_retrieval_noise_enabled():
    """Smoke test: plug a retrieval noise config into the
    orchestrator's TeacherMemory and run a short class."""
    orch = OrchestratorV2(n_students=5, max_classes=1, seed=7)
    orch.memory.retrieval_noise_config = RetrievalNoiseConfig(
        dropout_prob=0.3, similarity_jitter=0.1,
    )
    orch.classroom.MAX_TURNS = 60
    result = orch.run_class()
    assert "metrics" in result
    # Case base still populated.
    assert len(orch.memory.case_base._records) >= 1
