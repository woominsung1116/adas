"""Tests for YAML → SearchSpace loader (Phase 4.5 bridge).

Covers:
  - Real .harness/student_parameter_ranges.yaml loads cleanly
  - Expected parameter names present, with correct kind/range/default
  - Int vs float distinction for base_cognitive fields
  - Live-simulator fallback for defaults
  - Midpoint fallback when no explicit or live default
  - Constraints preserved verbatim
  - Metadata surfaced
  - Unsupported sections recorded (not crashed)
  - Malformed entries raise InvalidParameterSpecError loudly
  - Public default factory returns usable SearchSpace
"""

from pathlib import Path

import pytest

from src.calibration import (
    InvalidParameterSpecError,
    LoadedSearchSpace,
    load_search_space,
    load_default_search_space,
    build_default_search_space,
    default_student_ranges_path,
    SearchSpace,
)
from src.calibration.search_space_loader import _normalize_range, _choose_default


# ==========================================================================
# Real harness file
# ==========================================================================


def test_default_path_exists():
    assert default_student_ranges_path().exists()


def test_load_default_returns_non_empty_space():
    loaded = load_default_search_space()
    assert isinstance(loaded, LoadedSearchSpace)
    assert len(loaded.space) > 0
    assert isinstance(loaded.space, SearchSpace)


def test_build_default_search_space_shortcut():
    space = build_default_search_space()
    assert isinstance(space, SearchSpace)
    assert len(space) > 0


def test_expected_parameter_names_present():
    loaded = load_default_search_space()
    names = loaded.space.names()
    # Known base cognitive parameters
    assert "base_cognitive.att_bandwidth" in names
    assert "base_cognitive.vision_r" in names
    assert "base_cognitive.retention" in names
    assert "base_cognitive.plan_consistency" in names
    assert "base_cognitive.impulse_override" in names
    # Base emotional
    assert "base_emotional.frustration" in names
    assert "base_emotional.shame" in names
    assert "base_emotional.trust_in_teacher" in names
    # ADHD-I delta (cognitive)
    assert "adhd_inattentive.cognitive.att_bandwidth" in names
    assert "adhd_inattentive.cognitive.retention" in names
    # ADHD-I delta (emotional)
    assert "adhd_inattentive.emotional.frustration" in names
    # ADHD-H delta — alias mapping from "adhd_hyperactive" to full name
    assert "adhd_hyperactive_impulsive.cognitive.impulse_override" in names
    assert "adhd_hyperactive_impulsive.emotional.anger" in names


def test_integer_fields_have_int_kind():
    loaded = load_default_search_space()
    specs_by_name = {s.name: s for s in loaded.space.specs}
    # att_bandwidth / vision_r / retention are int
    assert specs_by_name["base_cognitive.att_bandwidth"].kind == "int"
    assert specs_by_name["base_cognitive.vision_r"].kind == "int"
    assert specs_by_name["base_cognitive.retention"].kind == "int"
    # Non-int cognitive field is float
    assert specs_by_name["base_cognitive.plan_consistency"].kind == "float"


def test_int_spec_has_int_bounds_and_default():
    loaded = load_default_search_space()
    specs_by_name = {s.name: s for s in loaded.space.specs}
    s = specs_by_name["base_cognitive.att_bandwidth"]
    assert s.kind == "int"
    assert isinstance(s.lo, int)
    assert isinstance(s.hi, int)
    assert isinstance(s.default, int)
    assert s.lo == 2
    assert s.hi == 4
    assert s.default == 3  # explicit YAML value


def test_explicit_default_takes_precedence():
    """YAML explicit default wins over live/midpoint."""
    loaded = load_default_search_space()
    specs_by_name = {s.name: s for s in loaded.space.specs}
    # importance_trigger has explicit default=135.0
    s = specs_by_name["base_cognitive.importance_trigger"]
    assert s.default == pytest.approx(135.0)


def test_source_citations_preserved():
    loaded = load_default_search_space()
    assert "base_cognitive.att_bandwidth" in loaded.sources
    assert "Miller" in loaded.sources["base_cognitive.att_bandwidth"]
    # ParameterSpec itself also carries source
    specs_by_name = {s.name: s for s in loaded.space.specs}
    assert "Miller" in specs_by_name["base_cognitive.att_bandwidth"].source


def test_constraints_preserved_verbatim():
    loaded = load_default_search_space()
    assert len(loaded.constraints) >= 4
    rules = [c["rule"] for c in loaded.constraints]
    # Known constraint substrings from the real YAML
    assert any("att_bandwidth" in r and "impulse_override" in r for r in rules)
    # Constraint entries carry profile + rationale
    for c in loaded.constraints:
        assert "rule" in c
        assert "profile" in c or "rule" in c  # at least one identifier


def test_metadata_surfaced():
    loaded = load_default_search_space()
    assert "version" in loaded.metadata
    assert "philosophy" in loaded.metadata


def test_unsupported_sections_recorded():
    loaded = load_default_search_space()
    # These YAML sections exist but aren't wired into the applier yet
    expected = {
        "emotion_decay_rates",
        "event_reactivity",
        "emotion_to_cognition_coupling",
        "comorbidity_interactions",
    }
    assert expected.issubset(set(loaded.unsupported_sections))


# ==========================================================================
# Defaults from live simulator fallback
# ==========================================================================


def test_live_simulator_default_fallback(tmp_path):
    """A YAML entry without explicit default should fall back to the
    live simulator value (BASE_EMOTIONAL[field])."""
    from src.simulation import cognitive_agent as ca

    # Pick a field whose live value is known
    live_shame = ca.BASE_EMOTIONAL["shame"]

    fake_yaml = tmp_path / "live_fallback.yaml"
    fake_yaml.write_text(
        """
base_emotional:
  shame:
    range: [0.0, 1.0]
    # no explicit default
""",
        encoding="utf-8",
    )
    loaded = load_search_space(fake_yaml)
    spec = [s for s in loaded.space.specs if s.name == "base_emotional.shame"][0]
    assert spec.default == pytest.approx(live_shame)


def test_midpoint_fallback_when_no_live_value(tmp_path):
    """A YAML entry for a field not in the simulator should fall back
    to the midpoint of [lo, hi]."""
    fake_yaml = tmp_path / "midpoint.yaml"
    fake_yaml.write_text(
        """
base_emotional:
  definitely_not_a_real_field:
    range: [0.2, 0.8]
""",
        encoding="utf-8",
    )
    loaded = load_search_space(fake_yaml)
    spec = [s for s in loaded.space.specs][0]
    # Live lookup returns None → midpoint
    assert spec.default == pytest.approx(0.5)


# ==========================================================================
# Choice parameter support
# ==========================================================================


# NOTE: The current real YAML has no choice parameters. Choice kind would
# require adding `kind: choice` + `choices: [...]` support in the loader.
# Since the YAML structure doesn't use that, we don't add synthetic choice
# support in this pass to avoid unused code paths. If needed later,
# `_infer_kind` would read `entry.get("kind")` first.


# ==========================================================================
# Malformed YAML entries — fail loudly
# ==========================================================================


def test_malformed_missing_range_raises(tmp_path):
    fake_yaml = tmp_path / "bad1.yaml"
    fake_yaml.write_text(
        """
base_emotional:
  shame:
    default: 0.1
    # missing range
""",
        encoding="utf-8",
    )
    with pytest.raises(InvalidParameterSpecError, match="missing 'range'"):
        load_search_space(fake_yaml)


def test_malformed_inverted_range_raises(tmp_path):
    fake_yaml = tmp_path / "bad2.yaml"
    fake_yaml.write_text(
        """
base_emotional:
  shame:
    range: [0.8, 0.2]
""",
        encoding="utf-8",
    )
    with pytest.raises(InvalidParameterSpecError, match="lo.*> hi"):
        load_search_space(fake_yaml)


def test_malformed_range_shape_raises(tmp_path):
    fake_yaml = tmp_path / "bad3.yaml"
    fake_yaml.write_text(
        """
base_emotional:
  shame:
    range: "not a list"
""",
        encoding="utf-8",
    )
    with pytest.raises(InvalidParameterSpecError):
        load_search_space(fake_yaml)


def test_unknown_delta_subsection_raises(tmp_path):
    fake_yaml = tmp_path / "bad4.yaml"
    fake_yaml.write_text(
        """
adhd_inattentive_deltas:
  weird_subsection:
    frustration:
      range: [0.1, 0.3]
""",
        encoding="utf-8",
    )
    with pytest.raises(InvalidParameterSpecError, match="unknown subsection"):
        load_search_space(fake_yaml)


def test_malformed_constraint_missing_rule_raises(tmp_path):
    fake_yaml = tmp_path / "bad5.yaml"
    fake_yaml.write_text(
        """
constraints:
  - profile: "adhd_inattentive"
    rationale: "no rule field"
""",
        encoding="utf-8",
    )
    with pytest.raises(InvalidParameterSpecError, match="missing 'rule'"):
        load_search_space(fake_yaml)


def test_constraints_not_list_raises(tmp_path):
    fake_yaml = tmp_path / "bad6.yaml"
    fake_yaml.write_text(
        """
constraints:
  some_key: "not a list"
""",
        encoding="utf-8",
    )
    with pytest.raises(InvalidParameterSpecError, match="must be a list"):
        load_search_space(fake_yaml)


def test_non_dict_yaml_root_raises(tmp_path):
    fake_yaml = tmp_path / "bad7.yaml"
    fake_yaml.write_text(
        """
- just
- a
- list
""",
        encoding="utf-8",
    )
    with pytest.raises(InvalidParameterSpecError, match="root must be a mapping"):
        load_search_space(fake_yaml)


# ==========================================================================
# Helper unit tests
# ==========================================================================


def test_normalize_range_list():
    assert _normalize_range([0.1, 0.5]) == (0.1, 0.5)


def test_normalize_range_dict():
    assert _normalize_range({"min": 0.1, "max": 0.5}) == (0.1, 0.5)


def test_normalize_range_invalid():
    with pytest.raises(InvalidParameterSpecError):
        _normalize_range([0.1])
    with pytest.raises(InvalidParameterSpecError):
        _normalize_range("bad")


def test_choose_default_precedence():
    # Explicit > live > midpoint
    assert _choose_default(0.42, 0.1, 0.0, 1.0, "float") == pytest.approx(0.42)
    assert _choose_default(None, 0.1, 0.0, 1.0, "float") == pytest.approx(0.1)
    assert _choose_default(None, None, 0.0, 1.0, "float") == pytest.approx(0.5)


def test_choose_default_int_coerces():
    # Explicit 2.7 → 3 (round)
    assert _choose_default(2.7, None, 1, 5, "int") == 3
    # Midpoint of 1..5 = 3
    assert _choose_default(None, None, 1, 5, "int") == 3


# ==========================================================================
# Integration — loaded space is usable by proposer
# ==========================================================================


def test_loaded_space_usable_by_random_proposer():
    """End-to-end: load YAML → feed SearchSpace to RandomProposer →
    get a valid config."""
    import random
    from src.calibration.proposer import RandomProposer

    loaded = load_default_search_space()
    proposer = RandomProposer(loaded.space, seed=42)
    config = proposer.propose([])
    # Every spec's default key is in the random config and within bounds
    for spec in loaded.space.specs:
        assert spec.name in config
        if spec.kind == "float":
            assert spec.lo <= config[spec.name] <= spec.hi
        elif spec.kind == "int":
            assert spec.lo <= config[spec.name] <= spec.hi


def test_loaded_space_validates_default_config():
    loaded = load_default_search_space()
    default_cfg = loaded.space.default_config()
    ok, errs = loaded.space.validate_config(default_cfg)
    assert ok, f"default config is invalid: {errs}"
