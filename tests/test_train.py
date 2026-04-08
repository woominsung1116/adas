import pytest

pytest.importorskip("gymnasium")

from unittest.mock import MagicMock
from train import load_config, build_env


def test_load_config():
    config = load_config("configs/default.yaml")
    assert "environment" in config
    assert "agent" in config
    assert "reward" in config
    assert "llm" in config


def test_build_env_returns_gymnasium_env():
    config = load_config("configs/default.yaml")
    mock_backend = MagicMock()
    mock_backend.generate.return_value = '{"state": {"distress_level": 0.4, "compliance": 0.4, "attention": 0.5, "escalation_risk": 0.2}, "narrative": "test"}'
    env = build_env(config, backend=mock_backend)
    obs, info = env.reset()
    assert obs is not None
