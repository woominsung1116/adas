import pytest

pytest.importorskip("gymnasium")

import numpy as np
from unittest.mock import MagicMock
from evaluate import evaluate_agent


def test_evaluate_agent_returns_summary():
    mock_env = MagicMock()
    mock_env.reset.return_value = (
        np.zeros(9, dtype="float32"),
        {"narrative": "start"},
    )
    mock_env.step.return_value = (
        np.zeros(9, dtype="float32"),
        5.0,
        True,
        False,
        {"narrative": "done", "action": "test", "turn": 1},
    )
    mock_env.distress_history = [0.3, 0.2]
    mock_env.history = [{"turn": 1, "action": "test", "state": {"distress_level": 0.2}, "narrative": "done"}]
    mock_env.current_state = {"compliance": 0.9}
    mock_env.success_threshold = 0.8

    mock_agent = MagicMock()
    mock_agent.predict.return_value = 0

    summary = evaluate_agent(mock_agent, mock_env, n_episodes=5)
    assert "success_rate" in summary
    assert summary["total_episodes"] == 5
