"""Interaction Log — records all events between students, teacher, and environment.

Every interaction (student↔student, teacher→student, environment→student)
is logged with timestamp, participants, type, content, and emotional context.
Logs persist across turns and classes for post-hoc analysis.

Usage:
    log = InteractionLog()
    log.record(InteractionEvent(...))
    log.save_json("results/logs/class_1.json")
    log.summary()
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any


@dataclass
class InteractionEvent:
    """A single interaction event in the simulation."""

    # When
    class_id: int
    turn: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Who
    actor: str = ""           # Who initiated (student_id, "teacher", "environment")
    target: str = ""          # Who was affected (student_id, "class", "teacher")
    participants: list[str] = field(default_factory=list)  # All involved

    # What
    event_type: str = ""      # see EVENT_TYPES below
    action: str = ""          # specific action taken
    content: str = ""         # natural language description
    inner_thought: str = ""   # actor's inner thought (not visible to others)

    # Context
    location: str = ""        # classroom, playground, office, hallway
    trigger: str = ""         # what caused this event

    # Emotional impact
    actor_emotions_before: dict[str, float] = field(default_factory=dict)
    actor_emotions_after: dict[str, float] = field(default_factory=dict)
    target_emotions_before: dict[str, float] = field(default_factory=dict)
    target_emotions_after: dict[str, float] = field(default_factory=dict)

    # State impact
    actor_state_before: dict[str, float] = field(default_factory=dict)
    actor_state_after: dict[str, float] = field(default_factory=dict)
    target_state_before: dict[str, float] = field(default_factory=dict)
    target_state_after: dict[str, float] = field(default_factory=dict)

    # Outcome
    outcome: str = ""         # positive, negative, neutral, escalated
    reward: float = 0.0

    # Monotonic event ID assigned by InteractionLog.record().
    # Used for stable trimming instead of fragile id()-based matching.
    _event_id: int = -1

    def to_dict(self) -> dict:
        return asdict(self)


# Standard event types
EVENT_TYPES = {
    # Teacher → Student
    "teacher_observe": "교사가 학생을 관찰",
    "teacher_intervene": "교사가 학생에게 개입",
    "teacher_praise": "교사가 학생을 칭찬",
    "teacher_correct_public": "교사가 교실에서 공개 지적",
    "teacher_correct_private": "교사가 교무실에서 1:1 상담",
    "teacher_identify": "교사가 학생을 ADHD로 판별",
    "teacher_class_instruction": "교사가 전체 학급 지도",

    # Student → Student
    "peer_chat": "학생끼리 대화/수다",
    "peer_conflict": "학생 간 충돌/싸움",
    "peer_bullying": "괴롭힘",
    "peer_help": "학생이 다른 학생을 도움",
    "peer_exclusion": "또래 배제",
    "peer_contagion": "행동 전염 (옆 학생 따라함)",
    "peer_friendship": "우정 상호작용",
    "peer_competition": "경쟁/비교",

    # Student → Self
    "emotional_outburst": "감정 폭발",
    "emotional_withdrawal": "감정적 위축/철수",
    "self_regulation": "자기 조절 시도",
    "task_engagement": "과제 참여",
    "task_avoidance": "과제 회피",
    "seat_leaving": "자리 이탈",
    "daydreaming": "공상",

    # Environment → Student
    "scenario_change": "시나리오 전환 (쉬는시간→수업 등)",
    "class_start": "수업 시작",
    "class_end": "수업 종료",
    "unexpected_event": "예상치 못한 사건",
}


class InteractionLog:
    """Accumulates and persists all interaction events.

    Risk: In long-running simulations with many classes, memory usage grows
    unboundedly. The max_events_per_class parameter caps per-class storage
    by dropping the oldest events when the limit is exceeded. The global
    _events list is also trimmed to stay consistent.
    """

    def __init__(self, max_events_per_class: int = 10000):
        self._events: list[InteractionEvent] = []
        self._class_events: dict[int, list[InteractionEvent]] = {}
        self._max_per_class = max_events_per_class
        self._next_event_id = 0

    def record(self, event: InteractionEvent) -> None:
        """Record a single interaction event."""
        event._event_id = self._next_event_id
        self._next_event_id += 1

        self._events.append(event)
        class_id = event.class_id
        if class_id not in self._class_events:
            self._class_events[class_id] = []
        self._class_events[class_id].append(event)

        # Trim oldest events for this class if over limit
        if len(self._class_events[class_id]) > self._max_per_class:
            dropped = self._class_events[class_id][:-self._max_per_class]
            self._class_events[class_id] = self._class_events[class_id][-self._max_per_class:]
            # Remove dropped events from the global list using stable event IDs
            dropped_ids = {e._event_id for e in dropped}
            self._events = [e for e in self._events if e._event_id not in dropped_ids]

    def get_events(
        self,
        class_id: int | None = None,
        turn: int | None = None,
        actor: str | None = None,
        target: str | None = None,
        event_type: str | None = None,
    ) -> list[InteractionEvent]:
        """Query events with optional filters."""
        events = self._events
        if class_id is not None:
            events = [e for e in events if e.class_id == class_id]
        if turn is not None:
            events = [e for e in events if e.turn == turn]
        if actor is not None:
            events = [e for e in events if e.actor == actor]
        if target is not None:
            events = [e for e in events if e.target == target]
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        return events

    def get_student_history(self, student_id: str, class_id: int | None = None) -> list[InteractionEvent]:
        """Get all events involving a specific student (as actor or target)."""
        events = self._class_events.get(class_id, self._events) if class_id else self._events
        return [
            e for e in events
            if e.actor == student_id or e.target == student_id or student_id in e.participants
        ]

    def get_teacher_actions(self, class_id: int | None = None) -> list[InteractionEvent]:
        """Get all teacher-initiated events."""
        events = self._class_events.get(class_id, self._events) if class_id else self._events
        return [e for e in events if e.actor == "teacher"]

    def get_peer_interactions(self, class_id: int | None = None) -> list[InteractionEvent]:
        """Get all student↔student events."""
        events = self._class_events.get(class_id, self._events) if class_id else self._events
        return [e for e in events if e.event_type.startswith("peer_")]

    def get_emotional_events(self, class_id: int | None = None) -> list[InteractionEvent]:
        """Get events with emotional state changes."""
        events = self._class_events.get(class_id, self._events) if class_id else self._events
        return [
            e for e in events
            if e.actor_emotions_before or e.target_emotions_before
        ]

    @property
    def total_events(self) -> int:
        return len(self._events)

    @property
    def classes(self) -> list[int]:
        return sorted(self._class_events.keys())

    def summary(self, class_id: int | None = None) -> dict:
        """Generate summary statistics."""
        events = self._class_events.get(class_id, self._events) if class_id else self._events

        type_counts: dict[str, int] = {}
        for e in events:
            type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1

        actor_counts: dict[str, int] = {}
        for e in events:
            actor_counts[e.actor] = actor_counts.get(e.actor, 0) + 1

        outcomes = {"positive": 0, "negative": 0, "neutral": 0, "escalated": 0}
        for e in events:
            if e.outcome in outcomes:
                outcomes[e.outcome] += 1

        return {
            "total_events": len(events),
            "event_types": type_counts,
            "actor_counts": actor_counts,
            "outcomes": outcomes,
            "classes": list(self._class_events.keys()) if class_id is None else [class_id],
            "peer_interactions": len([e for e in events if e.event_type.startswith("peer_")]),
            "teacher_actions": len([e for e in events if e.actor == "teacher"]),
            "emotional_events": len([e for e in events if e.actor_emotions_before]),
        }

    def save_json(self, path: str) -> None:
        """Save all events to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_events": self.total_events,
                    "classes": self.classes,
                    "summary": self.summary(),
                    "events": [e.to_dict() for e in self._events],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def save_class_json(self, class_id: int, path: str) -> None:
        """Save events for a single class."""
        events = self._class_events.get(class_id, [])
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "class_id": class_id,
                    "total_events": len(events),
                    "summary": self.summary(class_id),
                    "events": [e.to_dict() for e in events],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def export_timeline(self, class_id: int | None = None) -> list[dict]:
        """Export as a flat timeline for UI rendering."""
        events = self._class_events.get(class_id, self._events) if class_id else self._events
        return [
            {
                "turn": e.turn,
                "type": e.event_type,
                "type_label": EVENT_TYPES.get(e.event_type, e.event_type),
                "actor": e.actor,
                "target": e.target,
                "content": e.content,
                "location": e.location,
                "outcome": e.outcome,
            }
            for e in events
        ]

    def clear(self) -> None:
        """Clear all events."""
        self._events.clear()
        self._class_events.clear()
