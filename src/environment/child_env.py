from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from src.llm.backend import LLMBackend
from src.environment.child_profiles import ChildProfile
from src.environment.scenarios import Scenario
from src.environment.state_parser import StateParser
from src.environment.transition_constraints import (
    DEFAULT_ACTION_CONSTRAINTS,
    constrain_transition,
)
from src.reward.reward_function import RewardFunction

ACTIONS = [
    "transition_warning",
    "offer_choice",
    "labeled_praise",
    "visual_schedule_cue",
    "break_offer",
    "empathic_acknowledgment",
    "redirect_attention",
    "countdown_timer",
    "collaborative_problem_solving",
    "ignore_wait",
    "firm_boundary",
    "sensory_support",
]


class ADHDChildEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        backend: LLMBackend,
        profiles: list[ChildProfile],
        scenarios: list[Scenario],
        max_turns: int = 10,
        success_threshold: float = 0.8,
        failure_distress: float = 0.9,
        failure_consecutive: int = 3,
        reward_fn: RewardFunction | None = None,
        memory_store=None,
        action_constraints=None,
        use_constrained_transitions: bool = True,
        use_memory_adjusted_reset: bool = True,
    ):
        super().__init__()
        self.backend = backend
        self.profiles = profiles
        self.scenarios = scenarios
        self.max_turns = max_turns
        self.success_threshold = success_threshold
        self.failure_distress = failure_distress
        self.failure_consecutive = failure_consecutive
        self.reward_fn = reward_fn or RewardFunction()
        self.parser = StateParser()
        self.memory_store = memory_store
        self.action_constraints = action_constraints or DEFAULT_ACTION_CONSTRAINTS
        self.use_constrained_transitions = use_constrained_transitions
        self.use_memory_adjusted_reset = use_memory_adjusted_reset

        self.action_space = spaces.Discrete(len(ACTIONS))

        # observation: 4 state dims + 4 scenario one-hot + 1 normalized turn
        obs_dim = 4 + len(scenarios) + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.current_profile = None
        self.current_scenario = None
        self.current_state = None
        self.turn = 0
        self.history = []
        self.distress_history = []
        self.consecutive_high_distress = 0
        self.current_memory = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        self.current_profile = self.profiles[rng.integers(len(self.profiles))]
        self.current_scenario = self.scenarios[rng.integers(len(self.scenarios))]
        self.current_memory = (
            self.memory_store.get(self.current_profile.name)
            if self.memory_store is not None
            else None
        )
        self.current_state = self._build_initial_state()
        self.turn = 0
        self.history = []
        self.distress_history = [self.current_state["distress_level"]]
        self.consecutive_high_distress = 0

        obs = self._make_observation()
        info = {
            "narrative": self.current_scenario.description,
            "profile": self.current_profile.name,
            "scenario": self.current_scenario.name,
            "memory": self.current_memory.summary() if self.current_memory else None,
        }
        return obs, info

    def step(self, action: int):
        self.turn += 1
        action_name = ACTIONS[action]

        prompt = self._build_prompt(action_name)
        response = self.backend.generate(prompt)
        new_state, narrative = self.parser.parse(response)
        if self.use_constrained_transitions and self._has_literature_grounding():
            new_state = constrain_transition(
                action_name,
                self.current_state,
                new_state,
                constraints=self.action_constraints,
            )

        reward = self.reward_fn.compute(self.current_state, new_state, self.turn)
        prev_state = dict(self.current_state)

        self.history.append({
            "turn": self.turn,
            "action": action_name,
            "state": dict(new_state),
            "narrative": narrative,
        })
        self.distress_history.append(new_state["distress_level"])

        self.current_state = new_state

        # Check termination conditions
        terminated = False
        truncated = False

        if new_state["compliance"] > self.success_threshold:
            terminated = True
            reward += self.reward_fn.episode_bonus(
                success=True, turns=self.turn, distress_history=self.distress_history
            )

        if new_state["distress_level"] > self.failure_distress:
            self.consecutive_high_distress += 1
        else:
            self.consecutive_high_distress = 0

        if self.consecutive_high_distress >= self.failure_consecutive:
            terminated = True
            reward += self.reward_fn.episode_bonus(
                success=False, turns=self.turn, distress_history=self.distress_history
            )

        if self.turn >= self.max_turns and not terminated:
            truncated = True

        if self.current_memory is not None:
            self.current_memory.update(
                action_name=action_name,
                prev_state=prev_state,
                next_state=new_state,
                reward=float(reward),
                scenario_type=self.current_scenario.type,
                terminated_successfully=(
                    terminated and new_state["compliance"] > self.success_threshold
                ),
            )
            if terminated or truncated:
                self.current_memory.mark_session_complete()

        obs = self._make_observation()
        info = {
            "narrative": narrative,
            "action": action_name,
            "turn": self.turn,
            "memory": self.current_memory.summary() if self.current_memory else None,
        }
        return obs, float(reward), terminated, truncated, info

    def _build_initial_state(self) -> dict[str, float]:
        state = dict(self.current_scenario.initial_state)

        if not self.use_memory_adjusted_reset:
            return {
                key: max(0.0, min(1.0, float(value)))
                for key, value in state.items()
            }

        profile_priors = getattr(self.current_profile, "state_priors", {}) or {}
        scenario_priors = getattr(self.current_scenario, "state_priors", {}) or {}
        profile_sensitivity = (
            getattr(self.current_profile, "expected_transition_sensitivity", {}) or {}
        )
        scenario_sensitivity = (
            getattr(self.current_scenario, "expected_transition_sensitivity", {}) or {}
        )
        memory_adjustment = (
            self.current_memory.initial_state_adjustment(
                self.current_profile,
                self.current_scenario,
            )
            if self.current_memory is not None
            else {}
        )

        for key in state:
            blended_prior = 0.6 * float(state.get(key, 0.5))
            blended_prior += 0.2 * float(scenario_priors.get(key, state.get(key, 0.5)))
            blended_prior += 0.2 * float(profile_priors.get(key, state.get(key, 0.5)))
            blended_prior += 0.35 * float(scenario_sensitivity.get(key, 0.0))
            blended_prior += 0.25 * float(profile_sensitivity.get(key, 0.0))
            blended_prior += float(memory_adjustment.get(key, 0.0))
            state[key] = max(0.0, min(1.0, blended_prior))

        return state

    def _has_literature_grounding(self) -> bool:
        return bool(getattr(self.current_profile, "evidence", [])) and bool(
            getattr(self.current_scenario, "evidence", [])
        )

    def _make_observation(self) -> np.ndarray:
        state_arr = self.parser.state_to_array(self.current_state)

        scenario_onehot = np.zeros(len(self.scenarios), dtype=np.float32)
        idx = self.scenarios.index(self.current_scenario)
        scenario_onehot[idx] = 1.0

        turn_norm = np.array([self.turn / self.max_turns], dtype=np.float32)

        return np.concatenate([state_arr, scenario_onehot, turn_norm])

    def _build_prompt(self, action_name: str) -> str:
        history_text = ""
        for h in self.history[-3:]:
            history_text += f"Turn {h['turn']}: Clinician used '{h['action']}'. Child: {h['narrative']}\n"

        return f"""You are simulating a {self.current_profile.age}-year-old child with {self.current_profile.severity} ADHD.
Profile: {self.current_profile.description}
Traits: impulsivity={self.current_profile.traits.get('impulsivity', 0.5)}, inattention={self.current_profile.traits.get('inattention', 0.5)}, emotional_reactivity={self.current_profile.traits.get('emotional_reactivity', 0.5)}

Scenario: {self.current_scenario.description}
Scenario rationale: {self.current_scenario.behavioral_rationale or "Not specified."}
Scenario priors: {self.current_scenario.state_priors or self.current_scenario.initial_state}
Current observed state: distress_level={self.current_state["distress_level"]:.3f}, compliance={self.current_state["compliance"]:.3f}, attention={self.current_state["attention"]:.3f}, escalation_risk={self.current_state["escalation_risk"]:.3f}

Previous interactions:
{history_text if history_text else "None yet."}

Long-term student memory:
{self.current_memory.summary() if self.current_memory else "No prior classroom memory available."}

The clinician now uses: {action_name}

Respond as this child would. Return ONLY a JSON object with this exact format:
{{"state": {{"distress_level": <0.0-1.0>, "compliance": <0.0-1.0>, "attention": <0.0-1.0>, "escalation_risk": <0.0-1.0>}}, "narrative": "<1-2 sentence description of child's behavioral response>"}}"""


