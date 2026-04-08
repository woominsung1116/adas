from __future__ import annotations
import hashlib
import json
import os


class ResponseCache:
    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        self._hits = 0
        self._misses = 0
        if enabled:
            os.makedirs(cache_dir, exist_ok=True)

    def _make_key(self, prompt: str, context: str) -> str:
        combined = json.dumps({"prompt": prompt, "context": context}, sort_keys=True)
        return hashlib.sha256(combined.encode()).hexdigest()

    def get(self, prompt: str, context: str) -> str | None:
        if not self.enabled:
            self._misses += 1
            return None
        key = self._make_key(prompt, context)
        path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(path):
            self._hits += 1
            with open(path, "r") as f:
                return json.load(f)["response"]
        self._misses += 1
        return None

    def set(self, prompt: str, context: str, response: str) -> None:
        if not self.enabled:
            return
        key = self._make_key(prompt, context)
        path = os.path.join(self.cache_dir, f"{key}.json")
        with open(path, "w") as f:
            json.dump({"prompt": prompt, "context": context, "response": response}, f)

    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses}
