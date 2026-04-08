"""
mock_backend.py

Deterministic mock LLM backend for end-to-end testing without external API calls.

Implements the LLMBackend interface with rule-based JSON responses:
  - Early turns (<=5):  observe action
  - Middle turns (6-15): identify_adhd action on first suspicious student found
  - Later turns (16+):  individual_intervention
  - Report prompts: valid DSM-5 structured JSON
"""

from __future__ import annotations

import json
import re

from src.llm.backend import LLMBackend


class MockTeacherBackend(LLMBackend):
    """
    Stateless mock backend that returns valid TeacherAction JSON or report JSON
    based on simple pattern matching on the prompt.

    No external calls. No state between calls. Deterministic output.
    """

    def generate(self, prompt: str) -> str:
        # Detect if this is a report generation prompt vs action prompt
        if self._is_report_prompt(prompt):
            return self._make_report_response(prompt)
        return self._make_action_response(prompt)

    # ------------------------------------------------------------------
    # Prompt classification
    # ------------------------------------------------------------------

    def _is_report_prompt(self, prompt: str) -> bool:
        """Report prompts contain DSM-5 criterion lists and JSON schema template."""
        return (
            "inattention_symptoms" in prompt
            and "hyperactivity_symptoms" in prompt
            and "identified_subtype" in prompt
        )

    # ------------------------------------------------------------------
    # Action response
    # ------------------------------------------------------------------

    def _make_action_response(self, prompt: str) -> str:
        """
        Parse turn number and student list from prompt, return appropriate action.

        Phase 1 (turn <=5):  observe
        Phase 2 (turn 6-15): identify_adhd on a visible student
        Phase 3 (turn 16+):  individual_intervention
        """
        turn = self._extract_turn(prompt)
        student_ids = self._extract_student_ids(prompt)
        first_student = student_ids[0] if student_ids else "S01"

        if turn <= 5:
            action = {
                "action_type": "observe",
                "student_id": first_student,
                "strategy": None,
                "reasoning": f"Phase 1 mock: observing {first_student} at turn {turn}",
            }
        elif turn <= 15:
            action = {
                "action_type": "identify_adhd",
                "student_id": first_student,
                "strategy": None,
                "reasoning": (
                    f"Phase 2 mock: behavioral score high for {first_student} "
                    f"at turn {turn} — daydreaming, off_task observed repeatedly"
                ),
            }
        else:
            action = {
                "action_type": "individual_intervention",
                "student_id": first_student,
                "strategy": "redirect_attention",
                "reasoning": f"Phase 3 mock: redirecting {first_student} at turn {turn}",
            }

        return json.dumps(action)

    # ------------------------------------------------------------------
    # Report response
    # ------------------------------------------------------------------

    def _make_report_response(self, prompt: str) -> str:
        """Return a minimal valid DSM-5 report JSON."""
        # Extract student_id if present in prompt
        sid_match = re.search(r"학생\s+(\w+)", prompt)
        student_id = sid_match.group(1) if sid_match else "S01"

        report = {
            "student_id": student_id,
            "identified_subtype": "combined",
            "confidence": 0.75,
            "reasoning": (
                "Mock report: repeated daydreaming and out-of-seat behavior "
                "observed across multiple turns. Meets DSM-5 combined subtype criteria."
            ),
            "inattention_symptoms": [
                {
                    "criterion": "inattention_2",
                    "observed_behavior": "daydreaming during instruction",
                    "frequency": 4,
                },
                {
                    "criterion": "inattention_8",
                    "observed_behavior": "staring out window",
                    "frequency": 3,
                },
            ],
            "hyperactivity_symptoms": [
                {
                    "criterion": "hyperactivity_2",
                    "observed_behavior": "leaving seat without permission",
                    "frequency": 3,
                },
                {
                    "criterion": "hyperactivity_1",
                    "observed_behavior": "fidgeting in seat",
                    "frequency": 5,
                },
            ],
        }
        return json.dumps(report)

    # ------------------------------------------------------------------
    # Prompt parsers
    # ------------------------------------------------------------------

    def _extract_turn(self, prompt: str) -> int:
        """Extract turn number from 'Turn N' or '(Turn N)' pattern."""
        match = re.search(r"[Tt]urn\s+(\d+)", prompt)
        if match:
            return int(match.group(1))
        return 1

    def _extract_student_ids(self, prompt: str) -> list[str]:
        """Extract student IDs like S01, S02 etc. from prompt."""
        matches = re.findall(r"\bS\d{2}\b", prompt)
        # Deduplicate while preserving order
        seen: set[str] = set()
        result: list[str] = []
        for m in matches:
            if m not in seen:
                seen.add(m)
                result.append(m)
        return result
