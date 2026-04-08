from __future__ import annotations
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO


class PPOAgent:
    def __init__(
        self,
        env: gym.Env,
        hidden_sizes: list[int] | None = None,
        learning_rate: float = 0.0003,
        n_steps: int = 128,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        seed: int = 42,
    ):
        hidden_sizes = hidden_sizes or [64, 64]
        policy_kwargs = {"net_arch": hidden_sizes}
        self.model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            seed=seed,
            verbose=0,
        )

    def train(self, total_timesteps: int) -> None:
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str, env: gym.Env) -> "PPOAgent":
        agent = cls.__new__(cls)
        agent.model = PPO.load(path, env=env)
        return agent
