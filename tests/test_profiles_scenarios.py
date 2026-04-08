import pytest
from pathlib import Path
from src.environment.child_profiles import ChildProfile, load_profiles
from src.environment.scenarios import Scenario, load_scenarios

_ADAS_ROOT = Path(__file__).parent.parent


def test_child_profile_fields():
    profile = ChildProfile(
        name="test_child",
        age=9,
        severity="moderate",
        traits={"impulsivity": 0.7, "inattention": 0.6, "emotional_reactivity": 0.5},
        description="A 9-year-old with moderate ADHD.",
    )
    assert profile.name == "test_child"
    assert profile.severity == "moderate"
    assert 0.0 <= profile.traits["impulsivity"] <= 1.0


def test_load_profiles_from_yaml(tmp_path):
    yaml_content = """
profiles:
  - name: child_a
    age: 8
    severity: mild
    traits:
      impulsivity: 0.4
      inattention: 0.5
      emotional_reactivity: 0.3
    description: "Mild ADHD, mostly inattentive."
  - name: child_b
    age: 11
    severity: severe
    traits:
      impulsivity: 0.9
      inattention: 0.8
      emotional_reactivity: 0.85
    description: "Severe ADHD, combined type."
"""
    path = tmp_path / "profiles.yaml"
    path.write_text(yaml_content)
    profiles = load_profiles(str(path))
    assert len(profiles) == 2
    assert profiles[0].severity == "mild"
    assert profiles[1].severity == "severe"


def test_scenario_fields():
    scenario = Scenario(
        name="recess_to_math",
        type="preferred_to_nonpreferred",
        description="Teacher announces transition from recess to math class.",
        initial_state={"distress_level": 0.3, "compliance": 0.2, "attention": 0.4, "escalation_risk": 0.2},
    )
    assert scenario.type == "preferred_to_nonpreferred"
    assert scenario.initial_state["distress_level"] == 0.3


def test_load_scenarios_from_yaml(tmp_path):
    yaml_content = """
scenarios:
  - name: recess_to_math
    type: preferred_to_nonpreferred
    description: "Teacher announces transition from recess to math."
    initial_state:
      distress_level: 0.3
      compliance: 0.2
      attention: 0.4
      escalation_risk: 0.2
"""
    path = tmp_path / "scenarios.yaml"
    path.write_text(yaml_content)
    scenarios = load_scenarios(str(path))
    assert len(scenarios) == 1
    assert scenarios[0].name == "recess_to_math"


def test_profile_severity_values():
    for sev in ["mild", "moderate", "severe"]:
        p = ChildProfile(name="x", age=8, severity=sev, traits={}, description="")
        assert p.severity in ("mild", "moderate", "severe")


def test_loaded_profiles_have_literature_metadata():
    profiles = load_profiles(str(_ADAS_ROOT / "data/profiles/adhd_profiles.yaml"))
    assert profiles
    for profile in profiles:
        assert profile.evidence
        assert profile.behavioral_rationale
        assert profile.state_priors
        assert profile.expected_transition_sensitivity



def test_loaded_scenarios_have_literature_metadata():
    scenarios = load_scenarios(str(_ADAS_ROOT / "data/scenarios/task_transitions.yaml"))
    assert scenarios
    for scenario in scenarios:
        assert scenario.evidence
        assert scenario.behavioral_rationale
        assert scenario.state_priors
        assert scenario.expected_transition_sensitivity
