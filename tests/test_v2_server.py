"""Real async WebSocket server tests for _run_v2_session().

Tests the actual server control flow: WebSocket message protocol,
pause/resume via _ctrl_reader, speed control, and session lifecycle.
Uses a MockWebSocket to capture sent messages and inject client commands.

NOTE: Does NOT require pytest-asyncio. All async tests use asyncio.run()
wrappers to stay compatible with any pytest environment.
"""
from __future__ import annotations

import asyncio
import json
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest

import app.backend.server as server_mod
from app.backend.server import _run_v2_session


# ---------------------------------------------------------------------------
# MockWebSocket
# ---------------------------------------------------------------------------


class MockWebSocket:
    """Fake WebSocket for testing server control flow."""

    def __init__(self):
        self.sent: list[dict] = []
        self._receive_queue: asyncio.Queue | None = None
        self.accepted = False
        self.closed = False

    def _ensure_queue(self):
        if self._receive_queue is None:
            self._receive_queue = asyncio.Queue()

    async def accept(self):
        self.accepted = True

    async def send_json(self, data: dict):
        self.sent.append(data)

    async def receive_text(self) -> str:
        self._ensure_queue()
        return await self._receive_queue.get()

    def inject_message(self, msg: dict):
        """Inject a client message (pause/resume/speed/etc)."""
        self._ensure_queue()
        self._receive_queue.put_nowait(json.dumps(msg))

    async def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ORIG_DEFAULT_DELAY = getattr(server_mod, "DEFAULT_V2_TURN_DELAY", 0.1)
_ORIG_MIN_DELAY = getattr(server_mod, "MIN_V2_TURN_DELAY", 0.0)

# Number of simulated days (x5 periods = total turns).
# 10 days = 50 turns: enough for pause/resume, fast enough for CI.
_TEST_TOTAL_DAYS = 10


def _patch_for_fast_short(monkeypatch):
    """Minimal turn delays + short classroom."""
    if hasattr(server_mod, "DEFAULT_V2_TURN_DELAY"):
        monkeypatch.setattr(server_mod, "DEFAULT_V2_TURN_DELAY", 0.001)
    if hasattr(server_mod, "MIN_V2_TURN_DELAY"):
        monkeypatch.setattr(server_mod, "MIN_V2_TURN_DELAY", 0.0)

    from src.simulation.classroom_env_v2 import ClassroomV2
    _orig_init = ClassroomV2.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.TOTAL_DAYS = _TEST_TOTAL_DAYS
        self.MAX_TURNS = self.TOTAL_DAYS * self.PERIODS_PER_DAY

    monkeypatch.setattr(ClassroomV2, "__init__", _patched_init)


@pytest.fixture(autouse=True)
def _fast_short_env(monkeypatch):
    _patch_for_fast_short(monkeypatch)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_msg(n_students: int = 5, seed: int = 42) -> dict:
    return {"type": "start_session", "mode": "v2", "n_students": n_students, "seed": seed}


def _events_of_type(ws: MockWebSocket, event_type: str) -> list[dict]:
    return [m for m in ws.sent if m.get("type") == event_type]


def _run(coro):
    """Run an async coroutine synchronously. Works without pytest-asyncio."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Tests — all synchronous, using _run() wrapper
# ---------------------------------------------------------------------------


def test_v2_session_sends_new_class_first():
    """Starting a v2 session should send new_class as the first event."""
    ws = MockWebSocket()
    _run(asyncio.wait_for(_run_v2_session(ws, _default_msg()), timeout=15))

    assert len(ws.sent) >= 1
    first = ws.sent[0]
    assert first["type"] == "new_class"
    assert "students" in first
    assert first["n_students"] == 5
    assert len(first["students"]) == 5


def test_v2_session_sends_turn_events():
    """v2 session should send turn events between new_class and class_complete."""
    ws = MockWebSocket()
    _run(asyncio.wait_for(_run_v2_session(ws, _default_msg()), timeout=15))

    types = [m["type"] for m in ws.sent]
    turn_events = _events_of_type(ws, "turn")
    assert len(turn_events) > 0

    new_class_idx = types.index("new_class")
    first_turn_idx = types.index("turn")
    assert first_turn_idx > new_class_idx


def test_v2_session_ends_with_class_complete():
    """Session should end with a class_complete event."""
    ws = MockWebSocket()
    _run(asyncio.wait_for(_run_v2_session(ws, _default_msg()), timeout=15))

    last = ws.sent[-1]
    assert last["type"] == "class_complete"
    assert "true_positives" in last
    assert "false_positives" in last
    assert "false_negatives" in last
    assert "reports" in last


def test_v2_session_turn_payload_structure():
    """Turn events should have the expected fields from the server protocol."""
    ws = MockWebSocket()
    _run(asyncio.wait_for(_run_v2_session(ws, _default_msg()), timeout=15))

    turns = _events_of_type(ws, "turn")
    assert len(turns) > 0

    t = turns[0]
    assert "turn" in t
    assert "class_id" in t
    assert "day" in t
    assert "period" in t
    assert "subject" in t
    assert "students" in t
    assert "teacher_action" in t


def test_v2_session_new_class_student_names():
    """new_class event should have 'name' field per student (Korean names)."""
    ws = MockWebSocket()
    _run(asyncio.wait_for(_run_v2_session(ws, _default_msg()), timeout=15))

    nc = ws.sent[0]
    assert nc["type"] == "new_class"
    for s in nc["students"]:
        assert "id" in s
        assert "name" in s
        assert len(s["name"]) >= 2


def test_v2_pause_stops_emission():
    """Sending pause should stop turn events until resume."""
    async def _test():
        ws = MockWebSocket()
        old_default = getattr(server_mod, "DEFAULT_V2_TURN_DELAY", 0.1)
        server_mod.DEFAULT_V2_TURN_DELAY = 0.02
        try:
            task = asyncio.create_task(_run_v2_session(ws, _default_msg()))

            # Wait until we have at least 5 turn events
            for _ in range(500):
                if len(_events_of_type(ws, "turn")) >= 5:
                    break
                await asyncio.sleep(0.01)

            count_before_pause = len(_events_of_type(ws, "turn"))
            ws.inject_message({"type": "pause"})

            await asyncio.sleep(0.15)
            count_while_paused = len(_events_of_type(ws, "turn"))

            # Should not have advanced much (at most 2 in-flight)
            assert count_while_paused - count_before_pause <= 2

            ws.inject_message({"type": "resume"})
            for _ in range(500):
                if len(_events_of_type(ws, "turn")) > count_while_paused + 2:
                    break
                await asyncio.sleep(0.01)

            count_after_resume = len(_events_of_type(ws, "turn"))
            assert count_after_resume > count_while_paused

            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        finally:
            server_mod.DEFAULT_V2_TURN_DELAY = old_default

    _run(asyncio.wait_for(_test(), timeout=15))


def test_v2_resume_continues_after_pause():
    """Resume after pause should continue and eventually complete."""
    async def _test():
        ws = MockWebSocket()
        task = asyncio.create_task(_run_v2_session(ws, _default_msg()))

        for _ in range(300):
            if len(_events_of_type(ws, "turn")) >= 2:
                break
            await asyncio.sleep(0.01)

        ws.inject_message({"type": "pause"})
        await asyncio.sleep(0.1)
        ws.inject_message({"type": "resume"})

        await asyncio.wait_for(task, timeout=10)
        assert ws.sent[-1]["type"] == "class_complete"

    _run(asyncio.wait_for(_test(), timeout=15))


def test_v2_speed_control():
    """Changing speed should be accepted without breaking the session."""
    async def _test():
        ws = MockWebSocket()
        task = asyncio.create_task(_run_v2_session(ws, _default_msg()))

        for _ in range(300):
            if len(_events_of_type(ws, "turn")) >= 2:
                break
            await asyncio.sleep(0.01)

        ws.inject_message({"type": "speed", "delay": 0.0})

        await asyncio.wait_for(task, timeout=10)
        assert ws.sent[-1]["type"] == "class_complete"

    _run(asyncio.wait_for(_test(), timeout=15))


def test_v2_event_ordering():
    """All events should follow strict ordering: new_class, turn*, class_complete."""
    ws = MockWebSocket()
    _run(asyncio.wait_for(_run_v2_session(ws, _default_msg()), timeout=15))

    types = [m["type"] for m in ws.sent]
    assert types[0] == "new_class"
    assert types[-1] == "class_complete"

    middle = types[1:-1]
    assert len(middle) > 0
    assert all(t == "turn" for t in middle)


def test_v2_session_deterministic_with_seed():
    """Same seed should produce the same event sequence."""
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()

    _run(asyncio.wait_for(_run_v2_session(ws1, _default_msg(seed=123)), timeout=15))
    _run(asyncio.wait_for(_run_v2_session(ws2, _default_msg(seed=123)), timeout=15))

    assert len(ws1.sent) == len(ws2.sent)

    types1 = [m["type"] for m in ws1.sent]
    types2 = [m["type"] for m in ws2.sent]
    assert types1 == types2

    ids1 = [s["id"] for s in ws1.sent[0]["students"]]
    ids2 = [s["id"] for s in ws2.sent[0]["students"]]
    assert ids1 == ids2


def test_v2_turn_includes_structured_interactions():
    """Turn events should have interaction array with actor/target/event_type."""
    ws = MockWebSocket()
    _run(asyncio.wait_for(_run_v2_session(ws, _default_msg()), timeout=15))

    turns = _events_of_type(ws, "turn")
    assert len(turns) > 0

    for t in turns:
        interactions = t.get("interactions")
        assert isinstance(interactions, list), (
            f"interactions should be a list, got {type(interactions)}"
        )
        for ix in interactions:
            assert "actor" in ix
            assert "target" in ix
            assert "event_type" in ix
            assert ix["event_type"].startswith("peer_")


def test_v2_class_complete_includes_reports():
    """class_complete should have reports array."""
    ws = MockWebSocket()
    _run(asyncio.wait_for(_run_v2_session(ws, _default_msg()), timeout=15))

    cc = ws.sent[-1]
    assert cc["type"] == "class_complete"
    assert "reports" in cc
    assert isinstance(cc["reports"], list)


def test_v2_class_complete_includes_auprc_macro_f1():
    """class_complete growth should have auprc and macro_f1 keys."""
    ws = MockWebSocket()
    _run(asyncio.wait_for(_run_v2_session(ws, _default_msg()), timeout=15))

    cc = ws.sent[-1]
    assert cc["type"] == "class_complete"
    assert "growth" in cc
    growth = cc["growth"]
    assert isinstance(growth, dict)
    # growth may be empty if no tracker, but the key must exist
    if growth:
        assert "auprc" in growth
        assert "macro_f1" in growth


def test_v2_private_correction_sets_office_location():
    """When teacher does private_correction, turn location should be 'office'."""
    ws = MockWebSocket()
    _run(asyncio.wait_for(_run_v2_session(ws, _default_msg(n_students=20, seed=7)), timeout=30))

    turns = _events_of_type(ws, "turn")
    # Check that any turn with private_correction action has office location
    for t in turns:
        action = t.get("teacher_action", {})
        if action.get("action_type") == "private_correction":
            assert t["location"] == "office", (
                f"Turn {t['turn']}: private_correction should set location to 'office', "
                f"got '{t['location']}'"
            )
