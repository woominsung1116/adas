"""YAML → SearchSpace loader for autoresearch calibration.

Bridges `.harness/student_parameter_ranges.yaml` into a runtime
`SearchSpace` + constraint list + metadata dict, so autoresearch runs
can be driven by the declarative harness file rather than by manually
constructed `ParameterSpec(...)` lists.

Only **scalar parameter definitions** that map onto dotted config keys
the applier understands are turned into `ParameterSpec` entries.
Richer dynamics sections (emotion decay rates, event reactivity tables,
comorbidity interactions, emotion→cognition coupling) are recognized
and preserved in `unsupported_sections` metadata, but NOT included in
the search space — downstream applier work would be needed first.

Constraints from the YAML are loaded as a raw list of dicts, preserving
their textual `rule` expressions verbatim. This loader does NOT try to
evaluate or enforce them; it only surfaces them to downstream code.
Proposer/orchestrator enforcement is future work.

Dotted-key convention (matches `applier.parse_key`):
  base_cognitive.<field>
  base_emotional.<field>
  base_observable.<field>
  <profile>.cognitive.<field>
  <profile>.emotional.<field>
  <profile>.observable.<field>

Default-value precedence (for ParameterSpec.default):
  1. explicit `default:` in YAML entry
  2. live simulator value (via `cognitive_agent` module lookup)
  3. midpoint of [lo, hi]

Unknown/malformed entries fail loudly with `InvalidParameterSpecError`
rather than being silently skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .proposer import ParameterSpec, SearchSpace


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class InvalidParameterSpecError(ValueError):
    """Raised when a YAML entry cannot be converted into a valid ParameterSpec."""


# ---------------------------------------------------------------------------
# Loaded result structure
# ---------------------------------------------------------------------------


@dataclass
class LoadedSearchSpace:
    """Bundle returned by `load_search_space`.

    Fields:
      space: runtime `SearchSpace` with ParameterSpec entries
      constraints: raw list of constraint dicts from YAML (not enforced)
      metadata: YAML metadata section (version, philosophy, ...)
      unsupported_sections: names of YAML top-level sections that were
        recognized but not turned into ParameterSpecs in this pass
        (e.g. emotion_decay_rates). Preserved so future passes know what
        is left to wire in.
      sources: map of param_name → literature citation string
    """

    space: SearchSpace
    constraints: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    unsupported_sections: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.space)


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------


# Known top-level YAML keys that hold scalar parameter specs routable
# through the applier's dotted-path convention.
_SCALAR_PARAM_SECTIONS: dict[str, dict[str, Any]] = {
    # YAML key -> {"dotted_prefix": str, "default_kind": "int" | "float"}
    "base_cognitive": {"dotted_prefix": "base_cognitive", "default_kind": "float"},
    "base_emotional": {"dotted_prefix": "base_emotional", "default_kind": "float"},
    "base_observable": {"dotted_prefix": "base_observable", "default_kind": "float"},
}

# Integer-typed cognitive fields (for `base_cognitive.*`)
_INTEGER_COGNITIVE_FIELDS: set[str] = {"att_bandwidth", "vision_r", "retention"}

# YAML sections that are profile delta tables, with their canonical
# applier profile name. The YAML uses `<profile>_deltas` naming so we
# strip the suffix to get the real profile.
_PROFILE_DELTA_SUFFIX = "_deltas"

# Known profile mapping for naming variations between YAML and PROFILE_DELTAS.
# The YAML suffix `_deltas` is stripped, but additional aliases go here.
_PROFILE_ALIAS: dict[str, str] = {
    "adhd_hyperactive": "adhd_hyperactive_impulsive",
}

# Top-level YAML keys that are explicitly metadata / constraints / unsupported.
_METADATA_KEY = "metadata"
_CONSTRAINTS_KEY = "constraints"

# Sections that are recognized but NOT yet convertible into ParameterSpec
# (applier does not route these paths). Kept as metadata for traceability.
_UNSUPPORTED_SECTIONS: set[str] = {
    "emotion_decay_rates",
    "event_reactivity",
    "emotion_to_cognition_coupling",
    "comorbidity_interactions",
}


def _try_import_yaml():
    try:
        import yaml  # type: ignore
        return yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load student_parameter_ranges.yaml. "
            "Install with: uv pip install pyyaml"
        ) from exc


def _normalize_range(raw) -> tuple[float, float]:
    """Accept [lo, hi] list or {min, max} dict and return (lo, hi)."""
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return (float(raw[0]), float(raw[1]))
    if isinstance(raw, dict) and "min" in raw and "max" in raw:
        return (float(raw["min"]), float(raw["max"]))
    raise InvalidParameterSpecError(
        f"cannot parse range {raw!r}; expected [lo, hi] or {{min, max}}"
    )


def _infer_kind(
    lo: float,
    hi: float,
    explicit_default: Any,
    section_key: str,
    field_name: str,
) -> str:
    """Decide int vs float.

    Rules:
      - If bounds are both whole numbers AND section is base_cognitive
        AND field is a known integer cognitive field → int
      - Explicit default being int-valued reinforces but doesn't override
      - Otherwise float
    """
    if section_key == "base_cognitive" and field_name in _INTEGER_COGNITIVE_FIELDS:
        return "int"
    return "float"


def _live_default_for_base(
    section_key: str, field_name: str
) -> float | int | None:
    """Read the live simulator baseline as a fallback default.

    Returns None if the value cannot be read (e.g. unknown field name).
    """
    try:
        from src.simulation import cognitive_agent as ca
    except Exception:
        return None
    if section_key == "base_cognitive":
        if hasattr(ca.BASE_COGNITIVE, field_name):
            return getattr(ca.BASE_COGNITIVE, field_name)
    elif section_key == "base_emotional":
        return ca.BASE_EMOTIONAL.get(field_name)
    elif section_key == "base_observable":
        return ca.BASE_OBSERVABLE.get(field_name)
    return None


def _live_default_for_profile(
    profile: str, section: str, field_name: str
) -> float | int | None:
    """Read the live profile delta value as a fallback default."""
    try:
        from src.simulation import cognitive_agent as ca
    except Exception:
        return None
    spec = ca.PROFILE_DELTAS.get(profile)
    if not isinstance(spec, dict):
        return None
    section_dict = spec.get(section)
    if not isinstance(section_dict, dict):
        return None
    return section_dict.get(field_name)


def _choose_default(
    explicit: Any,
    live_value: Any,
    lo: float,
    hi: float,
    kind: str,
) -> float | int:
    """Precedence:
       1. explicit YAML default
       2. live simulator value
       3. midpoint
    """
    if explicit is not None:
        candidate = explicit
    elif live_value is not None:
        candidate = live_value
    else:
        candidate = (lo + hi) / 2.0

    if kind == "int":
        return int(round(float(candidate)))
    return float(candidate)


def _build_base_param_specs(
    section_key: str,
    section_body: dict[str, Any],
    sources_out: dict[str, str],
) -> list[ParameterSpec]:
    """Convert a `base_*` section dict to ParameterSpec list."""
    out: list[ParameterSpec] = []
    prefix = _SCALAR_PARAM_SECTIONS[section_key]["dotted_prefix"]
    for field_name, entry in (section_body or {}).items():
        if not isinstance(entry, dict) or "range" not in entry:
            raise InvalidParameterSpecError(
                f"{section_key}.{field_name}: missing 'range' or not a dict; got {entry!r}"
            )
        try:
            lo, hi = _normalize_range(entry["range"])
        except InvalidParameterSpecError as exc:
            raise InvalidParameterSpecError(
                f"{section_key}.{field_name}: {exc}"
            )
        if lo > hi:
            raise InvalidParameterSpecError(
                f"{section_key}.{field_name}: lo {lo} > hi {hi}"
            )
        kind = _infer_kind(lo, hi, entry.get("default"), section_key, field_name)
        live = _live_default_for_base(section_key, field_name)
        default = _choose_default(entry.get("default"), live, lo, hi, kind)
        dotted = f"{prefix}.{field_name}"
        source_str = entry.get("source", "")
        if source_str:
            sources_out[dotted] = source_str
        out.append(
            ParameterSpec(
                name=dotted,
                lo=lo if kind == "float" else int(round(lo)),
                hi=hi if kind == "float" else int(round(hi)),
                kind=kind,
                default=default,
                source=source_str,
            )
        )
    return out


def _resolve_profile_name(yaml_section_key: str) -> str:
    """Strip `_deltas` suffix and apply alias mapping."""
    if yaml_section_key.endswith(_PROFILE_DELTA_SUFFIX):
        base = yaml_section_key[: -len(_PROFILE_DELTA_SUFFIX)]
    else:
        base = yaml_section_key
    return _PROFILE_ALIAS.get(base, base)


def _build_profile_delta_specs(
    yaml_key: str,
    body: dict[str, Any],
    sources_out: dict[str, str],
) -> list[ParameterSpec]:
    """Convert a `<profile>_deltas` section into ParameterSpec list.

    Each subsection (cognitive/emotional/observable) becomes
    `<profile>.<section>.<field>` dotted paths.
    """
    out: list[ParameterSpec] = []
    profile = _resolve_profile_name(yaml_key)
    if not isinstance(body, dict):
        raise InvalidParameterSpecError(
            f"{yaml_key}: body is not a dict; got {type(body).__name__}"
        )
    for section_name, section_body in body.items():
        if section_name not in ("cognitive", "emotional", "observable"):
            # Unknown subsection inside a delta table — treat as malformed
            raise InvalidParameterSpecError(
                f"{yaml_key}.{section_name}: unknown subsection; "
                f"expected cognitive/emotional/observable"
            )
        if not isinstance(section_body, dict):
            raise InvalidParameterSpecError(
                f"{yaml_key}.{section_name}: not a dict"
            )
        for field_name, entry in section_body.items():
            if not isinstance(entry, dict) or "range" not in entry:
                raise InvalidParameterSpecError(
                    f"{yaml_key}.{section_name}.{field_name}: "
                    f"missing 'range' or not a dict; got {entry!r}"
                )
            lo, hi = _normalize_range(entry["range"])
            if lo > hi:
                raise InvalidParameterSpecError(
                    f"{yaml_key}.{section_name}.{field_name}: lo {lo} > hi {hi}"
                )
            # Delta values are float unless explicit integer cognitive field
            kind = "float"
            if section_name == "cognitive" and field_name in _INTEGER_COGNITIVE_FIELDS:
                kind = "int"
            live = _live_default_for_profile(profile, section_name, field_name)
            default = _choose_default(entry.get("default"), live, lo, hi, kind)
            dotted = f"{profile}.{section_name}.{field_name}"
            source_str = entry.get("source", "")
            if source_str:
                sources_out[dotted] = source_str
            out.append(
                ParameterSpec(
                    name=dotted,
                    lo=lo if kind == "float" else int(round(lo)),
                    hi=hi if kind == "float" else int(round(hi)),
                    kind=kind,
                    default=default,
                    source=source_str,
                )
            )
    return out


def _load_constraints(raw: Any) -> list[dict[str, Any]]:
    """Preserve constraint entries verbatim, failing loudly on malformed input."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise InvalidParameterSpecError(
            f"constraints must be a list; got {type(raw).__name__}"
        )
    out: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise InvalidParameterSpecError(
                f"constraints[{idx}]: entry must be a dict; got {type(entry).__name__}"
            )
        if "rule" not in entry:
            raise InvalidParameterSpecError(
                f"constraints[{idx}]: missing 'rule' field"
            )
        # Normalize: keep rule as raw text; preserve any other keys
        normalized = dict(entry)
        normalized["rule"] = str(entry["rule"])
        out.append(normalized)
    return out


def load_search_space(path: str | Path) -> LoadedSearchSpace:
    """Load a YAML parameter-range file into a `LoadedSearchSpace`.

    Args:
        path: filesystem path to the YAML file

    Returns:
        LoadedSearchSpace with populated SearchSpace + raw constraints +
        metadata + unsupported section names + source citations.

    Raises:
        InvalidParameterSpecError for malformed parameter entries
        ImportError if PyYAML is missing
        FileNotFoundError if path does not exist
    """
    yaml = _try_import_yaml()
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}

    if not isinstance(doc, dict):
        raise InvalidParameterSpecError(
            f"YAML root must be a mapping; got {type(doc).__name__}"
        )

    space = SearchSpace()
    sources: dict[str, str] = {}
    unsupported: list[str] = []

    for top_key, body in doc.items():
        if top_key in _SCALAR_PARAM_SECTIONS:
            for spec in _build_base_param_specs(top_key, body, sources):
                space.add(spec)
            continue
        if top_key.endswith(_PROFILE_DELTA_SUFFIX):
            for spec in _build_profile_delta_specs(top_key, body, sources):
                space.add(spec)
            continue
        if top_key in _UNSUPPORTED_SECTIONS:
            unsupported.append(top_key)
            continue
        if top_key == _METADATA_KEY:
            continue  # handled below
        if top_key == _CONSTRAINTS_KEY:
            continue  # handled below
        # Unknown top-level key — preserve as unsupported but don't crash
        unsupported.append(top_key)

    constraints = _load_constraints(doc.get(_CONSTRAINTS_KEY))
    metadata_raw = doc.get(_METADATA_KEY) or {}
    if not isinstance(metadata_raw, dict):
        metadata_raw = {}

    return LoadedSearchSpace(
        space=space,
        constraints=constraints,
        metadata=dict(metadata_raw),
        unsupported_sections=unsupported,
        sources=sources,
    )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def default_student_ranges_path() -> Path:
    """Return the canonical .harness/student_parameter_ranges.yaml path."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / ".harness" / "student_parameter_ranges.yaml"


def load_default_search_space() -> LoadedSearchSpace:
    """Convenience factory: load the canonical student parameter ranges YAML."""
    return load_search_space(default_student_ranges_path())


def build_default_search_space() -> SearchSpace:
    """Shortcut: return just the SearchSpace from the default YAML file.

    For callers that don't need constraints or metadata. Most real calibration
    runs should use `load_default_search_space()` instead to also get
    constraints for future enforcement.
    """
    return load_default_search_space().space
