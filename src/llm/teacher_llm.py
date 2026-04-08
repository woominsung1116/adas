"""
teacher_llm.py

LLM-backed decision layer for the teacher agent in the ADHD classroom simulation.

Wraps any LLMBackend to:
  - Build Korean-language prompts from ClassroomObservation + TeacherMemory
  - Parse JSON responses into TeacherAction / IdentificationReport
  - Cache action-decision calls; skip cache for identify/report (context-unique)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.cache.response_cache import ResponseCache
from src.llm.backend import LLMBackend
from src.simulation.multi_student_env import (
    ClassroomObservation,
    StudentObservation,
    TeacherAction,
)
from src.simulation.teacher_memory import TeacherMemory

# ---------------------------------------------------------------------------
# Valid action types and intervention strategies
# ---------------------------------------------------------------------------

VALID_ACTION_TYPES = {
    "observe",
    "class_instruction",
    "individual_intervention",
    "private_correction",
    "public_correction",
    "identify_adhd",
    "generate_report",
}

VALID_STRATEGIES = {
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
}

# Actions that are highly context-specific — skip cache
_NO_CACHE_ACTIONS = {"identify_adhd", "generate_report"}


# ---------------------------------------------------------------------------
# IdentificationReport dataclass
# ---------------------------------------------------------------------------


@dataclass
class SymptomEntry:
    criterion: str
    observed_behavior: str
    frequency: int = 0


@dataclass
class IdentificationReport:
    student_id: str
    identified_subtype: str  # inattentive / hyperactive-impulsive / combined
    confidence: float        # 0.0 – 1.0
    reasoning: str
    inattention_symptoms: list[SymptomEntry] = field(default_factory=list)
    hyperactivity_symptoms: list[SymptomEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "student_id": self.student_id,
            "identified_subtype": self.identified_subtype,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "inattention_symptoms": [
                {
                    "criterion": s.criterion,
                    "observed_behavior": s.observed_behavior,
                    "frequency": s.frequency,
                }
                for s in self.inattention_symptoms
            ],
            "hyperactivity_symptoms": [
                {
                    "criterion": s.criterion,
                    "observed_behavior": s.observed_behavior,
                    "frequency": s.frequency,
                }
                for s in self.hyperactivity_symptoms
            ],
        }


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_ACTION_PROMPT_TEMPLATE = """\
당신은 한국 초등학교 담임교사입니다. {n}명의 학생이 있는 교실을 관찰하고 있습니다.
목표: ADHD가 의심되는 학생을 관찰을 통해 판별하고, 적절한 케어를 제공하세요.

## 현재 관찰 (Turn {turn})
{student_observations}

## 교사 기억
유사 사례: {similar_cases}
학습된 원칙: {principles}
학생별 누적 관찰: {behavior_summaries}

## 현재까지 ADHD로 판별한 학생
{identified_list}

## 관리 진행 상황
판별된 ADHD: {n_identified}명 / 관리 완료: {n_managed}명

## 사용 가능한 행동
1. observe(student_id) - 특정 학생 집중 관찰
2. class_instruction() - 전체 학급 지도
3. individual_intervention(student_id, strategy) - 개별 개입
   전략: transition_warning, offer_choice, labeled_praise, visual_schedule_cue,
   break_offer, empathic_acknowledgment, redirect_attention, countdown_timer,
   collaborative_problem_solving, ignore_wait, firm_boundary, sensory_support
4. private_correction(student_id) - 교무실 1:1 개별 지도
5. public_correction(student_id) - 교실 내 공개 지적
6. identify_adhd(student_id, reasoning) - ADHD 판별 (근거 필수)
7. generate_report(student_id) - 판별 리포트 생성

하나의 행동을 선택하세요. 반드시 JSON으로 응답:
{{"action_type": "...", "student_id": "...", "strategy": "...", "reasoning": "..."}}
"""

_REPORT_PROMPT_TEMPLATE = """\
당신은 한국 초등학교 담임교사입니다. 학생 {student_id}에 대한 ADHD 판별 리포트를 작성하세요.

## 관찰 기록
{observation_history}

## DSM-5 ADHD 진단 기준
부주의 증상 (9개 중 6개 이상):
1. 부주의한 실수 2. 주의 유지 어려움 3. 경청 어려움 4. 지시 따르기 실패
5. 조직화 어려움 6. 지속적 노력 회피 7. 물건 분실 8. 외부 자극에 산만 9. 일상활동 잊음

과잉행동-충동성 증상 (9개 중 6개 이상):
1. 손발 꼼지락 2. 자리 이탈 3. 부적절한 달리기 4. 조용히 놀기 어려움
5. 끊임없이 움직임 6. 과도한 말하기 7. 질문 전 대답 8. 차례 기다리기 어려움 9. 방해/끼어들기

다음 JSON 형식으로 리포트를 작성하세요:
{{
    "student_id": "{student_id}",
    "identified_subtype": "inattentive|hyperactive-impulsive|combined",
    "confidence": 0.0-1.0,
    "reasoning": "판별 근거 설명",
    "inattention_symptoms": [
        {{"criterion": "inattention_1", "observed_behavior": "관찰된 구체적 행동", "frequency": 횟수}}
    ],
    "hyperactivity_symptoms": [
        {{"criterion": "hyperactivity_1", "observed_behavior": "관찰된 구체적 행동", "frequency": 횟수}}
    ]
}}
"""


# ---------------------------------------------------------------------------
# TeacherLLM
# ---------------------------------------------------------------------------


class TeacherLLM:
    """
    Wraps any LLMBackend for the teacher agent's two core tasks:
      1. decide_action()            — choose the next classroom action
      2. generate_identification_report() — write a DSM-5-structured report
    """

    def __init__(
        self,
        backend: LLMBackend,
        memory: TeacherMemory,
        cache_enabled: bool = True,
    ) -> None:
        self.backend = backend
        self.memory = memory
        self.cache = (
            ResponseCache(".cache/teacher_responses", enabled=True)
            if cache_enabled
            else ResponseCache(".cache/teacher_responses", enabled=False)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide_action(
        self, observation: ClassroomObservation, turn: int
    ) -> TeacherAction:
        """Use LLM to decide the teacher's next action."""
        prompt = self._build_prompt(observation, turn)
        cache_context = self._action_cache_context(observation, turn)
        response = self._call_llm(prompt, cache_context=cache_context, skip_cache=False)
        return self._parse_response(response)

    def generate_identification_report(
        self, student_id: str, observation_history: list[dict[str, Any]]
    ) -> IdentificationReport:
        """Use LLM to generate a detailed DSM-5-aligned identification report."""
        prompt = self._build_report_prompt(student_id, observation_history)
        # Reports are unique per full context; always skip cache
        response = self._call_llm(prompt, cache_context="", skip_cache=True)
        return self._parse_report(response, student_id)

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_prompt(self, observation: ClassroomObservation, turn: int) -> str:
        n = len(observation.student_observations)

        # Student observation lines
        student_lines = self._format_student_observations(observation.student_observations)

        # RAG: gather all unique behaviors from this turn for retrieval
        all_behaviors: list[str] = []
        for obs in observation.student_observations:
            all_behaviors.extend(obs.behaviors)

        similar_cases = self._format_similar_cases(all_behaviors)
        principles = self._format_principles()
        behavior_summaries = self._format_behavior_summaries()

        # Identified ADHD list
        if observation.identified_adhd_ids:
            profiles = self.memory.all_profiles()
            id_parts: list[str] = []
            for sid in observation.identified_adhd_ids:
                p = profiles.get(sid)
                if p:
                    conf = f"{p.identification_confidence:.2f}"
                    reason = p.identification_reasoning or "근거 없음"
                    id_parts.append(f"- {sid}: 신뢰도={conf}, 근거={reason}")
                else:
                    id_parts.append(f"- {sid}")
            identified_list = "\n".join(id_parts)
        else:
            identified_list = "없음"

        return _ACTION_PROMPT_TEMPLATE.format(
            n=n,
            turn=turn,
            student_observations=student_lines,
            similar_cases=similar_cases,
            principles=principles,
            behavior_summaries=behavior_summaries,
            identified_list=identified_list,
            n_identified=len(observation.identified_adhd_ids),
            n_managed=len(observation.managed_ids),
        )

    def _build_report_prompt(
        self, student_id: str, observation_history: list[dict[str, Any]]
    ) -> str:
        if observation_history:
            history_lines: list[str] = []
            for entry in observation_history:
                turn_n = entry.get("turn", "?")
                behaviors = entry.get("behaviors", [])
                state = entry.get("state", {})
                action = entry.get("action_taken", "none")
                outcome = entry.get("outcome", "")
                line = (
                    f"Turn {turn_n}: 행동={behaviors}, "
                    f"상태={state}, 개입={action}"
                )
                if outcome:
                    line += f", 결과={outcome}"
                history_lines.append(line)
            history_text = "\n".join(history_lines)
        else:
            # Fall back to what memory has for this student
            profile = self.memory.get_profile(student_id)
            history_text = (
                f"누적 행동 빈도: {profile.behavior_frequency_counts}\n"
                f"개입 반응: {profile.response_to_interventions}"
            )

        return _REPORT_PROMPT_TEMPLATE.format(
            student_id=student_id,
            observation_history=history_text,
        )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_student_observations(
        self, observations: list[StudentObservation]
    ) -> str:
        lines: list[str] = []
        for obs in observations:
            behaviors_str = ", ".join(obs.behaviors) if obs.behaviors else "행동 없음"
            line = f"- {obs.student_id}: {behaviors_str}"
            if obs.state_snapshot:
                snap = obs.state_snapshot
                line += (
                    f" [주의력={snap.get('attention', 0):.2f}, "
                    f"순응도={snap.get('compliance', 0):.2f}, "
                    f"스트레스={snap.get('distress_level', 0):.2f}, "
                    f"위험도={snap.get('escalation_risk', 0):.2f}]"
                )
            lines.append(line)
        return "\n".join(lines) if lines else "관찰 없음"

    def _format_similar_cases(self, behaviors: list[str]) -> str:
        if not behaviors:
            return "없음"
        similar = self.memory.retrieve_similar_cases(behaviors, top_k=3)
        if not similar:
            return "없음"
        parts: list[str] = []
        for sim, rec in similar:
            if sim < 0.05:
                continue
            parts.append(
                f"유사도={sim:.2f}: 학생={rec.student_id}, "
                f"행동={rec.observed_behaviors}, 결과={rec.outcome}"
            )
        return "\n".join(parts) if parts else "없음"

    def _format_principles(self) -> str:
        principles = self.memory.experience_base.top_principles(top_k=5)
        if not principles:
            return "없음"
        lines = [
            f"{'[교정]' if p.is_corrective else '[긍정]'} {p.text}"
            for p in principles
        ]
        return "\n".join(lines)

    def _format_behavior_summaries(self) -> str:
        profiles = self.memory.all_profiles()
        if not profiles:
            return "없음"
        lines: list[str] = []
        for sid, profile in profiles.items():
            top = profile.dominant_behaviors(top_k=3)
            score = profile.adhd_indicator_score()
            lines.append(
                f"- {sid}: 주요행동={top}, ADHD지표={score:.2f}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM call with optional caching
    # ------------------------------------------------------------------

    def _call_llm(
        self, prompt: str, cache_context: str = "", skip_cache: bool = False
    ) -> str:
        if not skip_cache:
            cached = self.cache.get(prompt, cache_context)
            if cached is not None:
                return cached

        response = self.backend.generate(prompt)

        if not skip_cache:
            self.cache.set(prompt, cache_context, response)

        return response

    # ------------------------------------------------------------------
    # Cache context
    # ------------------------------------------------------------------

    def _action_cache_context(
        self, observation: ClassroomObservation, turn: int
    ) -> str:
        """
        Cache key component: hash of (turn, per-student behavior snapshot,
        memory summary) so identical situations reuse the cached decision.
        """
        student_snapshot = {
            obs.student_id: sorted(obs.behaviors)
            for obs in observation.student_observations
        }
        profiles = self.memory.all_profiles()
        memory_summary = {
            sid: {
                "top_behaviors": p.dominant_behaviors(top_k=3),
                "adhd_score": round(p.adhd_indicator_score(), 3),
            }
            for sid, p in profiles.items()
        }
        payload = {
            "turn": turn,
            "student_snapshot": student_snapshot,
            "memory_summary": memory_summary,
            "identified": sorted(observation.identified_adhd_ids),
            "managed": sorted(observation.managed_ids),
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Response parsers
    # ------------------------------------------------------------------

    def _parse_response(self, response: str) -> TeacherAction:
        """Parse LLM response into TeacherAction with safe fallbacks."""
        data = _extract_json(response)

        action_type = str(data.get("action_type", "class_instruction")).strip()
        if action_type not in VALID_ACTION_TYPES:
            action_type = "class_instruction"

        student_id = data.get("student_id") or None
        if student_id is not None:
            student_id = str(student_id).strip() or None

        strategy = data.get("strategy") or None
        if strategy is not None:
            strategy = str(strategy).strip()
            if strategy not in VALID_STRATEGIES:
                strategy = None

        reasoning = str(data.get("reasoning", "")).strip()

        # Structural validity: actions that need a student_id
        if action_type in {
            "observe",
            "individual_intervention",
            "private_correction",
            "public_correction",
            "identify_adhd",
            "generate_report",
        } and not student_id:
            action_type = "class_instruction"
            student_id = None
            strategy = None

        # individual_intervention needs a valid strategy
        if action_type == "individual_intervention" and not strategy:
            strategy = "redirect_attention"

        return TeacherAction(
            action_type=action_type,
            student_id=student_id,
            strategy=strategy,
            reasoning=reasoning,
        )

    def _parse_report(self, response: str, student_id: str) -> IdentificationReport:
        """Parse LLM response into IdentificationReport with safe fallbacks."""
        data = _extract_json(response)

        sid = str(data.get("student_id", student_id)).strip() or student_id

        subtype = str(data.get("identified_subtype", "combined")).strip()
        if subtype not in {"inattentive", "hyperactive-impulsive", "combined"}:
            subtype = "combined"

        raw_confidence = data.get("confidence", 0.5)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        reasoning = str(data.get("reasoning", "")).strip()

        inattention_symptoms = _parse_symptom_list(
            data.get("inattention_symptoms", [])
        )
        hyperactivity_symptoms = _parse_symptom_list(
            data.get("hyperactivity_symptoms", [])
        )

        return IdentificationReport(
            student_id=sid,
            identified_subtype=subtype,
            confidence=confidence,
            reasoning=reasoning,
            inattention_symptoms=inattention_symptoms,
            hyperactivity_symptoms=hyperactivity_symptoms,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict[str, Any]:
    """
    Extract a JSON object from raw LLM output.
    Handles:
      - Plain JSON string
      - JSON wrapped in ```json ... ``` or ``` ... ``` markdown blocks
      - JSON embedded in surrounding prose
    Falls back to {} on any failure.
    """
    if not text:
        return {}

    # Strip markdown code fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        # Find first { ... } block
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace_match.group(0) if brace_match else text.strip()

    try:
        result = json.loads(candidate)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    return {}


def _parse_symptom_list(raw: Any) -> list[SymptomEntry]:
    """Convert a list of dicts from LLM JSON into SymptomEntry objects."""
    if not isinstance(raw, list):
        return []
    entries: list[SymptomEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        criterion = str(item.get("criterion", "")).strip()
        observed_behavior = str(item.get("observed_behavior", "")).strip()
        try:
            frequency = int(item.get("frequency", 0))
        except (TypeError, ValueError):
            frequency = 0
        if criterion:
            entries.append(
                SymptomEntry(
                    criterion=criterion,
                    observed_behavior=observed_behavior,
                    frequency=frequency,
                )
            )
    return entries
