import pytest

pytest.importorskip("gymnasium")

from unittest.mock import MagicMock

from src.agent.baselines import RuleBasedAgent
from src.simulation.classroom_world import ClassroomWorld
from train import build_env, load_config



def test_classroom_world_run_session_emits_event_stream():
    backend = MagicMock()
    backend.generate.return_value = '{"state": {"distress_level": 0.2, "compliance": 0.85, "attention": 0.7, "escalation_risk": 0.1}, "narrative": "The student nods and starts moving toward the next task."}'
    backend.cache = MagicMock()
    backend.cache.stats.return_value = {"hits": 0, "misses": 0}

    config = load_config("configs/default.yaml")
    env = build_env(config, backend=backend)
    world = ClassroomWorld(env=env, teacher_policy=RuleBasedAgent(sequence=[0]))

    session = world.run_session(session_id=1)

    assert session["events"][0]["action"] == "scene_setup"
    assert session["events"][-1]["termination_status"] in {"success", "failure", "timeout"}
    assert session["summary"]["turns"] >= 1
    assert "memory" in session["summary"]



def test_classroom_world_save_sessions_writes_json(tmp_path):
    backend = MagicMock()
    backend.generate.return_value = '{"state": {"distress_level": 0.2, "compliance": 0.85, "attention": 0.7, "escalation_risk": 0.1}, "narrative": "The student transitions successfully."}'
    backend.cache = MagicMock()
    backend.cache.stats.return_value = {"hits": 0, "misses": 0}

    config = load_config("configs/default.yaml")
    env = build_env(config, backend=backend)
    world = ClassroomWorld(env=env, teacher_policy=RuleBasedAgent(sequence=[0]))

    output_path = tmp_path / "classroom.json"
    log_data = world.save_sessions(str(output_path), n_sessions=2)

    assert output_path.exists()
    assert len(log_data["sessions"]) == 2
    assert log_data["memory_snapshot"]
