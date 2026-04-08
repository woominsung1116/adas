import pytest

gymnasium = pytest.importorskip("gymnasium")
gym = gymnasium

import numpy as np
from unittest.mock import MagicMock
from gymnasium import spaces


def test_ppo_agent_wrapper_constructs():
    from src.agent.ppo_agent import PPOAgent
    env = MagicMock(spec=gym.Env)
    env.observation_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)
    env.action_space = spaces.Discrete(12)
    agent = PPOAgent(env=env, hidden_sizes=[64, 64], learning_rate=0.0003)
    assert agent is not None


def test_ppo_agent_predict_returns_valid_action():
    from src.agent.ppo_agent import PPOAgent
    env = MagicMock(spec=gym.Env)
    env.observation_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)
    env.action_space = spaces.Discrete(12)
    agent = PPOAgent(env=env)
    obs = np.zeros(9, dtype=np.float32)
    action = agent.predict(obs)
    assert 0 <= action < 12


def test_ppo_agent_save_and_load(tmp_path):
    from src.agent.ppo_agent import PPOAgent
    env = MagicMock(spec=gym.Env)
    env.observation_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)
    env.action_space = spaces.Discrete(12)
    agent = PPOAgent(env=env)
    save_path = str(tmp_path / "test_model")
    agent.save(save_path)
    loaded = PPOAgent.load(save_path, env=env)
    obs = np.zeros(9, dtype=np.float32)
    action = loaded.predict(obs)
    assert 0 <= action < 12
