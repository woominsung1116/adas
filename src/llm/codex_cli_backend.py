from __future__ import annotations
import json
import os
import subprocess
import tempfile
import time

from src.cache.response_cache import ResponseCache
from src.environment.state_parser import STATE_KEYS, StateParser
from src.llm.backend import LLMBackend


OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["state", "narrative"],
    "properties": {
        "state": {
            "type": "object",
            "additionalProperties": False,
            "required": list(STATE_KEYS),
            "properties": {
                key: {"type": "number", "minimum": 0.0, "maximum": 1.0}
                for key in STATE_KEYS
            },
        },
        "narrative": {"type": "string"},
    },
}


class CodexCLIBackend(LLMBackend):
    def __init__(
        self,
        cache_dir: str = ".cache/responses",
        cache_enabled: bool = True,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
        model: str | None = None,
        command: str = "codex",
        timeout: int = 180,
        cwd: str | None = None,
    ):
        self.cache = ResponseCache(cache_dir=cache_dir, enabled=cache_enabled)
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.model = model
        self.command = command
        self.timeout = timeout
        self.cwd = cwd
        self.parser = StateParser()

    def generate(self, prompt: str) -> str:
        cached = self.cache.get(prompt, self._cache_context())
        if cached is not None:
            return cached

        response = self._call_codex(prompt)
        self.cache.set(prompt, self._cache_context(), response)
        return response

    def _cache_context(self) -> str:
        return json.dumps(
            {
                "backend": "codex_cli",
                "model": self.model or "default",
                "command": self.command,
            },
            sort_keys=True,
        )

    def _call_codex(self, prompt: str) -> str:
        last_error = "Codex CLI returned an empty or invalid response."
        for attempt in range(self.retry_attempts):
            schema_path = None
            output_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w",
                    suffix=".json",
                    delete=False,
                    encoding="utf-8",
                ) as schema_file:
                    json.dump(OUTPUT_SCHEMA, schema_file)
                    schema_path = schema_file.name

                output_fd, output_path = tempfile.mkstemp(suffix=".txt")
                os.close(output_fd)

                command = [
                    self.command,
                    "exec",
                    "--skip-git-repo-check",
                    "--sandbox",
                    "read-only",
                    "--ephemeral",
                    "--output-schema",
                    schema_path,
                    "--output-last-message",
                    output_path,
                ]
                if self.model:
                    command.extend(["-m", self.model])
                command.append("-")

                result = subprocess.run(
                    command,
                    input=self._wrap_prompt(prompt),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    cwd=self.cwd,
                )

                candidate = self._read_candidate_response(result.stdout, output_path)
                if result.returncode == 0 and self._is_valid_payload(candidate):
                    return candidate

                stderr = (result.stderr or "").strip()
                if stderr:
                    last_error = stderr
            except subprocess.TimeoutExpired:
                last_error = f"Codex CLI timed out after {self.timeout} seconds"
            finally:
                for temp_path in (schema_path, output_path):
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

            if attempt < self.retry_attempts - 1:
                time.sleep(self.retry_delay)

        raise RuntimeError(f"Codex CLI failed after {self.retry_attempts} attempts: {last_error}")

    def _wrap_prompt(self, prompt: str) -> str:
        return (
            "Return only a single JSON object with keys 'state' and 'narrative'.\n"
            "The 'state' object must contain distress_level, compliance, attention, "
            "and escalation_risk as numbers between 0 and 1.\n\n"
            f"{prompt}"
        )

    def _read_candidate_response(self, stdout: str, output_path: str | None) -> str:
        if output_path and os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                file_text = f.read().strip()
            if file_text:
                return file_text
        return (stdout or "").strip()

    def _is_valid_payload(self, candidate: str) -> bool:
        if not candidate:
            return False
        state, narrative = self.parser.parse(candidate)
        if narrative == "Failed to parse response.":
            return False
        return all(key in state for key in STATE_KEYS)
