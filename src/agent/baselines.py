from __future__ import annotations

import numpy as np


class RandomAgent:
    def __init__(self, n_actions: int = 12, seed: int = 42):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(self.n_actions))

    def reset(self):
        pass


class RuleBasedAgent:
    def __init__(self, sequence: list[int] | None = None):
        self.sequence = sequence or [0, 2, 10]
        self.step_idx = 0

    def predict(self, obs: np.ndarray) -> int:
        action = self.sequence[self.step_idx % len(self.sequence)]
        self.step_idx += 1
        return action

    def reset(self):
        self.step_idx = 0


class SingleActionAgent:
    def __init__(self, action: int = 0):
        self.action = action

    def predict(self, obs: np.ndarray) -> int:
        return self.action

    def reset(self):
        pass
