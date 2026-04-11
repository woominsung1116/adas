"""Minimal constraint-aware filtering for autoresearch calibration.

Phase 4.5 loader parses harness YAML constraints but does not enforce
them. This module adds a *deliberately narrow* enforcement layer: only
four explicit rule patterns are supported, matching the constraints
currently declared in `.harness/student_parameter_ranges.yaml`. Any
other rule is preserved as "unsupported" and does NOT cause a trial
to fail — it is only surfaced in metadata.

Supported rule grammar (pattern-matched, NOT a general expression parser):

  1. abs(delta.<section>.<field>) > abs(delta.<section>.<field>)
     → profile's absolute value of first field must exceed the second

  2. delta.<section>.<field> > 0
     → profile's field must be strictly positive

  3. delta.<section>.<field> < 0
     → profile's field must be strictly negative

  4. delta.<section>.<field> >= 0  and  delta.<section>.<field> <= 0
     → non-strict variants (included for completeness)

Rules are bound to a specific `profile` (e.g. "adhd_inattentive") via
the constraint entry's `profile:` field. Field references resolve
against:

  Config key format:  <profile>.<section>.<field>
  Fallback:           live simulator PROFILE_DELTAS[<profile>][<section>][<field>]

If neither source supplies a value, the rule is reported as
"unresolvable" and does not count as a violation.

Intentionally unsupported:
  - compound expressions with `and`/`or`
  - the adhd_combined max-severity clause:
    "delta magnitude >= max(adhd_inattentive, adhd_hyperactive_impulsive)"
    (cross-profile comparison, out of scope for this pass)
  - arbitrary Python-style expression evaluation
  - set-valued constraints

These are explicitly surfaced in `check_constraints` results rather
than silently ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Rule representation
# ---------------------------------------------------------------------------


@dataclass
class SupportedRule:
    """A parsed rule that the evaluator can actually enforce."""

    # Categorical kind: "abs_gt" | "gt_zero" | "lt_zero" | "ge_zero" | "le_zero"
    kind: str
    # The profile that the rule is bound to
    profile: str
    # Primary field: (section, field_name), e.g. ("cognitive", "att_bandwidth")
    left: tuple[str, str]
    # For `abs_gt`, the right-hand comparison field. None for unary rules.
    right: tuple[str, str] | None = None
    # Original raw rule text (for diagnostics)
    raw: str = ""
    # Rationale from YAML, if present
    rationale: str = ""


@dataclass
class UnsupportedRule:
    """A rule the evaluator does not know how to enforce."""

    profile: str | None
    raw: str
    rationale: str = ""
    reason: str = ""  # why this rule is unsupported


@dataclass
class ConstraintCheckResult:
    """Structured outcome of evaluating a config against constraints."""

    valid: bool
    violations: list[SupportedRule] = field(default_factory=list)
    unsupported: list[UnsupportedRule] = field(default_factory=list)
    unresolvable: list[SupportedRule] = field(default_factory=list)

    def describe_violations(self) -> list[str]:
        """Human-readable violation messages."""
        out: list[str] = []
        for rule in self.violations:
            if rule.kind == "abs_gt":
                assert rule.right is not None
                out.append(
                    f"{rule.profile}: |delta.{rule.left[0]}.{rule.left[1]}| > "
                    f"|delta.{rule.right[0]}.{rule.right[1]}| violated"
                )
            elif rule.kind == "gt_zero":
                out.append(
                    f"{rule.profile}: delta.{rule.left[0]}.{rule.left[1]} > 0 violated"
                )
            elif rule.kind == "lt_zero":
                out.append(
                    f"{rule.profile}: delta.{rule.left[0]}.{rule.left[1]} < 0 violated"
                )
            elif rule.kind == "ge_zero":
                out.append(
                    f"{rule.profile}: delta.{rule.left[0]}.{rule.left[1]} >= 0 violated"
                )
            elif rule.kind == "le_zero":
                out.append(
                    f"{rule.profile}: delta.{rule.left[0]}.{rule.left[1]} <= 0 violated"
                )
        return out


# ---------------------------------------------------------------------------
# Pattern parser
# ---------------------------------------------------------------------------


# delta.<section>.<field> token pattern
_FIELD_REF = r"delta\.([a-zA-Z_]+)\.([a-zA-Z_][a-zA-Z0-9_]*)"


def parse_rule(
    raw_text: str, profile: str | None, rationale: str = ""
) -> SupportedRule | UnsupportedRule:
    """Try to match `raw_text` against a supported rule pattern.

    Returns a `SupportedRule` on success or `UnsupportedRule` with a
    `reason` string on failure. Never raises.
    """
    text = " ".join(raw_text.split())  # normalize whitespace

    if profile is None:
        return UnsupportedRule(
            profile=None,
            raw=raw_text,
            rationale=rationale,
            reason="rule missing 'profile' binding",
        )

    # Pattern 1: abs(delta.X.Y) > abs(delta.A.B)
    m = re.match(
        rf"^abs\(\s*{_FIELD_REF}\s*\)\s*>\s*abs\(\s*{_FIELD_REF}\s*\)$",
        text,
    )
    if m:
        left = (m.group(1), m.group(2))
        right = (m.group(3), m.group(4))
        return SupportedRule(
            kind="abs_gt",
            profile=profile,
            left=left,
            right=right,
            raw=raw_text,
            rationale=rationale,
        )

    # Pattern 2: delta.X.Y > 0  (strict positive)
    m = re.match(rf"^{_FIELD_REF}\s*>\s*0(\.0+)?$", text)
    if m:
        return SupportedRule(
            kind="gt_zero",
            profile=profile,
            left=(m.group(1), m.group(2)),
            raw=raw_text,
            rationale=rationale,
        )

    # Pattern 3: delta.X.Y < 0  (strict negative)
    m = re.match(rf"^{_FIELD_REF}\s*<\s*0(\.0+)?$", text)
    if m:
        return SupportedRule(
            kind="lt_zero",
            profile=profile,
            left=(m.group(1), m.group(2)),
            raw=raw_text,
            rationale=rationale,
        )

    # Pattern 4a: delta.X.Y >= 0
    m = re.match(rf"^{_FIELD_REF}\s*>=\s*0(\.0+)?$", text)
    if m:
        return SupportedRule(
            kind="ge_zero",
            profile=profile,
            left=(m.group(1), m.group(2)),
            raw=raw_text,
            rationale=rationale,
        )

    # Pattern 4b: delta.X.Y <= 0
    m = re.match(rf"^{_FIELD_REF}\s*<=\s*0(\.0+)?$", text)
    if m:
        return SupportedRule(
            kind="le_zero",
            profile=profile,
            left=(m.group(1), m.group(2)),
            raw=raw_text,
            rationale=rationale,
        )

    return UnsupportedRule(
        profile=profile,
        raw=raw_text,
        rationale=rationale,
        reason="rule does not match any supported pattern",
    )


def parse_constraints(
    raw_entries: list[dict[str, Any]],
) -> tuple[list[SupportedRule], list[UnsupportedRule]]:
    """Split raw YAML constraint entries into supported + unsupported lists."""
    supported: list[SupportedRule] = []
    unsupported: list[UnsupportedRule] = []
    for entry in raw_entries or []:
        profile = entry.get("profile")
        rule_text = entry.get("rule", "")
        rationale = entry.get("rationale", "")
        parsed = parse_rule(rule_text, profile, rationale)
        if isinstance(parsed, SupportedRule):
            supported.append(parsed)
        else:
            unsupported.append(parsed)
    return supported, unsupported


# ---------------------------------------------------------------------------
# Config field resolver
# ---------------------------------------------------------------------------


def _lookup_field_value(
    config: dict[str, Any],
    profile: str,
    section: str,
    field_name: str,
) -> float | None:
    """Get the value of `<profile>.<section>.<field>` from config or fallbacks.

    Resolution order:
      1. config dict: `<profile>.<section>.<field_name>`
      2. live PROFILE_DELTAS[profile][section][field_name] from cognitive_agent
      3. None (unresolvable)
    """
    dotted = f"{profile}.{section}.{field_name}"
    if dotted in config:
        return float(config[dotted])

    try:
        from src.simulation import cognitive_agent as ca
    except Exception:
        return None
    spec = ca.PROFILE_DELTAS.get(profile)
    if not isinstance(spec, dict):
        return None
    section_map = spec.get(section)
    if not isinstance(section_map, dict):
        return None
    val = section_map.get(field_name)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------


def check_constraints(
    config: dict[str, Any],
    supported_rules: list[SupportedRule],
    unsupported_rules: list[UnsupportedRule] | None = None,
) -> ConstraintCheckResult:
    """Evaluate a config dict against a set of parsed rules.

    Args:
        config: proposer-generated config (may be partial — missing fields
                fall back to live simulator values)
        supported_rules: parsed rules that the evaluator knows how to handle
        unsupported_rules: rules that were parsed but are not enforced;
                passed through into the result for transparency

    Returns:
        ConstraintCheckResult with valid flag + violation / unsupported /
        unresolvable lists.
    """
    violations: list[SupportedRule] = []
    unresolvable: list[SupportedRule] = []

    for rule in supported_rules:
        left_val = _lookup_field_value(
            config, rule.profile, rule.left[0], rule.left[1]
        )
        if rule.kind == "abs_gt":
            assert rule.right is not None
            right_val = _lookup_field_value(
                config, rule.profile, rule.right[0], rule.right[1]
            )
            if left_val is None or right_val is None:
                unresolvable.append(rule)
                continue
            if not (abs(left_val) > abs(right_val)):
                violations.append(rule)
        elif rule.kind == "gt_zero":
            if left_val is None:
                unresolvable.append(rule)
                continue
            if not (left_val > 0):
                violations.append(rule)
        elif rule.kind == "lt_zero":
            if left_val is None:
                unresolvable.append(rule)
                continue
            if not (left_val < 0):
                violations.append(rule)
        elif rule.kind == "ge_zero":
            if left_val is None:
                unresolvable.append(rule)
                continue
            if not (left_val >= 0):
                violations.append(rule)
        elif rule.kind == "le_zero":
            if left_val is None:
                unresolvable.append(rule)
                continue
            if not (left_val <= 0):
                violations.append(rule)
        else:
            # Defensive: future rule kinds we have not handled here
            unresolvable.append(rule)

    return ConstraintCheckResult(
        valid=not violations,
        violations=violations,
        unsupported=list(unsupported_rules or []),
        unresolvable=unresolvable,
    )


# ---------------------------------------------------------------------------
# Orchestrator integration error
# ---------------------------------------------------------------------------


def rule_field_keys(rule: SupportedRule) -> list[str]:
    """Return the dotted config keys a rule depends on.

    Used by the default setup to decide whether every field a rule
    references is actually tunable in the current search space. Rules
    whose fields are entirely frozen (not in the search space) cannot
    be satisfied or violated by the proposer — they resolve against
    live PROFILE_DELTAS only, and therefore any verdict is a property
    of the simulator defaults, not of autoresearch.
    """
    keys = [f"{rule.profile}.{rule.left[0]}.{rule.left[1]}"]
    if rule.right is not None:
        keys.append(f"{rule.profile}.{rule.right[0]}.{rule.right[1]}")
    return keys


def partition_rules_by_tunability(
    supported_rules: list[SupportedRule],
    tunable_keys: set[str],
) -> tuple[list[SupportedRule], list[SupportedRule]]:
    """Split supported rules into (enforceable, non_tunable).

    A rule is *enforceable* only if EVERY field it references is in
    the search space. If any referenced field is frozen in
    PROFILE_DELTAS, the proposer cannot fully control the rule's
    verdict — it becomes a fixed value of the simulator defaults.
    In the best case that is a no-op; in the worst case it is a
    constant-false verdict that degenerates the entire run into
    penalty trials (see `adhd_hyperactive_impulsive` under the
    current harness: `|impulse_override| > |att_bandwidth|` with
    `att_bandwidth` frozen at -1 can never be satisfied inside the
    proposer's range). We therefore refuse to enforce such rules
    here; they stay on the bundle as `non_tunable` for transparency.

    Returns:
        (enforceable, non_tunable) where both are disjoint subsets of
        the input list.
    """
    enforceable: list[SupportedRule] = []
    non_tunable: list[SupportedRule] = []
    for rule in supported_rules:
        keys = rule_field_keys(rule)
        if keys and all(k in tunable_keys for k in keys):
            enforceable.append(rule)
        else:
            non_tunable.append(rule)
    return enforceable, non_tunable


class ConstraintViolationError(Exception):
    """Raised by the evaluator path when a config violates supported constraints.

    Carries violation details so the orchestrator can produce a
    penalized trial with `error_type = "constraint_violation"`,
    distinct from `parameter_override` errors.
    """

    def __init__(
        self,
        violations: list[SupportedRule],
        config: dict[str, Any],
        unsupported: list[UnsupportedRule] | None = None,
    ) -> None:
        self.violations = list(violations)
        self.config = dict(config)
        self.unsupported = list(unsupported or [])
        details = "; ".join(
            ConstraintCheckResult(False, violations).describe_violations()
        )
        super().__init__(
            f"config violates {len(violations)} constraint"
            f"{'' if len(violations) == 1 else 's'}: {details}"
        )
