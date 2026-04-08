import pytest

pytest.importorskip("gymnasium")

import numpy as np
from unittest.mock import MagicMock
from train import load_config, build_env, run_training
from evaluate import evaluate_agent
from src.agent.baselines import RandomAgent


@pytest.fixture
def mock_backend():
    """Backend that simulates a child who gradually complies."""
    call_count = {"n": 0}

    def generate(prompt):
        call_count["n"] += 1
        turn = call_count["n"] % 10
        compliance = min(0.1 + turn * 0.12, 0.95)
        distress = max(0.6 - turn * 0.08, 0.1)
        return (
            f'{{"state": {{"distress_level": {distress:.2f}, "compliance": {compliance:.2f}, '
            f'"attention": {0.3 + turn * 0.05:.2f}, "escalation_risk": {max(0.3 - turn * 0.03, 0.05):.2f}}}, '
            f'"narrative": "Turn {turn}: child gradually settles."}}'
        )

    backend = MagicMock()
    backend.generate.side_effect = generate
    backend.cache = MagicMock()
    backend.cache.stats.return_value = {"hits": 0, "misses": 0}
    return backend


def test_full_training_loop(mock_backend):
    config = load_config("configs/default.yaml")
    config["agent"]["total_timesteps"] = 256
    config["agent"]["n_steps"] = 64
    config["agent"]["batch_size"] = 32

    env = build_env(config, backend=mock_backend)
    agent = run_training(config, env)
    assert agent is not None


def test_trained_agent_runs_evaluation(mock_backend):
    config = load_config("configs/default.yaml")
    config["agent"]["total_timesteps"] = 256
    config["agent"]["n_steps"] = 64
    config["agent"]["batch_size"] = 32

    env = build_env(config, backend=mock_backend)
    agent = run_training(config, env)

    eval_env = build_env(config, backend=mock_backend)
    summary = evaluate_agent(agent, eval_env, n_episodes=5)
    assert summary["total_episodes"] == 5
