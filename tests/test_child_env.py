import pytest

gymnasium = pytest.importorskip("gymnasium")
gym = gymnasium

import numpy as np
from unittest.mock import MagicMock
from src.environment.child_env import ADHDChildEnv
from src.environment.child_profiles import ChildProfile
from src.environment.scenarios import Scenario


@pytest.fixture
def mock_backend():
    backend = MagicMock()
    backend.generate.return_value = '{"state": {"distress_level": 0.4, "compliance": 0.4, "attention": 0.5, "escalation_risk": 0.2}, "narrative": "The child looks at the timer."}'
    return backend


@pytest.fixture
def profile():
    return ChildProfile(
        name="test", age=9, severity="moderate",
        traits={"impulsivity": 0.6, "inattention": 0.5, "emotional_reactivity": 0.5},
        description="Test child",
    )


@pytest.fixture
def scenario():
    return Scenario(
        name="test_scenario", type="preferred_to_nonpreferred",
        description="Recess to math",
        initial_state={"distress_level": 0.3, "compliance": 0.2, "attention": 0.4, "escalation_risk": 0.2},
    )


@pytest.fixture
def env(mock_backend, profile, scenario):
    return ADHDChildEnv(
        backend=mock_backend,
        profiles=[profile],
        scenarios=[scenario],
    )


def test_env_is_gymnasium_compatible(env):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Box)


def test_env_action_space_size(env):
    assert env.action_space.n == 12


def test_env_observation_space_shape(env):
    assert env.observation_space.shape[0] >= 4


def test_env_reset_returns_observation(env):
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert "narrative" in info


def test_env_step_returns_correct_tuple(env):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "narrative" in info


def test_env_terminates_on_success(mock_backend):
    mock_backend.generate.return_value = '{"state": {"distress_level": 0.1, "compliance": 0.9, "attention": 0.8, "escalation_risk": 0.1}, "narrative": "Child complies happily."}'
    profile = ChildProfile(name="t", age=9, severity="mild", traits={}, description="")
    scenario = Scenario(name="s", type="test", description="", initial_state={"distress_level": 0.2, "compliance": 0.2, "attention": 0.4, "escalation_risk": 0.1})
    env = ADHDChildEnv(backend=mock_backend, profiles=[profile], scenarios=[scenario])
    env.reset()
    _, _, terminated, _, _ = env.step(0)
    assert terminated is True


def test_env_truncates_at_max_turns(mock_backend):
    mock_backend.generate.return_value = '{"state": {"distress_level": 0.5, "compliance": 0.3, "attention": 0.4, "escalation_risk": 0.3}, "narrative": "Still resisting."}'
    profile = ChildProfile(name="t", age=9, severity="moderate", traits={}, description="")
    scenario = Scenario(name="s", type="test", description="", initial_state={"distress_level": 0.3, "compliance": 0.2, "attention": 0.4, "escalation_risk": 0.2})
    env = ADHDChildEnv(backend=mock_backend, profiles=[profile], scenarios=[scenario], max_turns=3)
    env.reset()
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(0)
        assert not truncated
    _, _, terminated, truncated, _ = env.step(0)
    assert truncated is True


def test_env_records_episode_history(env):
    env.reset()
    env.step(0)
    env.step(1)
    assert len(env.history) == 2
