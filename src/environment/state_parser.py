from __future__ import annotations
import json
import re
import numpy as np

STATE_KEYS = ["distress_level", "compliance", "attention", "escalation_risk"]
DEFAULT_STATE = {k: 0.5 for k in STATE_KEYS}


class StateParser:
    def parse(self, response: str) -> tuple[dict[str, float], str]:
        data = self._extract_json(response)
        if data is None:
            return dict(DEFAULT_STATE), "Failed to parse response."

        state_raw = data.get("state", DEFAULT_STATE)
        state = {}
        for key in STATE_KEYS:
            val = float(state_raw.get(key, 0.5))
            state[key] = max(0.0, min(1.0, val))

        narrative = data.get("narrative", "No narrative provided.")
        return state, narrative

    def _extract_json(self, text: str) -> dict | None:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        match = re.search(r"\{[^{}]*\"state\"[^{}]*\{.*?\}.*?\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def state_to_array(self, state: dict[str, float]) -> np.ndarray:
        return np.array([state[k] for k in STATE_KEYS], dtype=np.float32)
