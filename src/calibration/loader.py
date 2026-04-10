"""YAML → calibration target loader.

Parses `.harness/naturalness_targets.yaml` and `.harness/epidemiology_targets.yaml`
into runtime `NaturalnessTarget` / `EpidemiologyTarget` objects, resolving
metric path strings through the registry in `metrics.py`.

Design:
  - Fails loudly for unknown metric paths (NotImplementedError)
  - Skips unsupported-but-optional entries with a warning path (opt-in)
  - Produces targets ready to pass to `compute_combined_loss`

Usage:
    from src.calibration import load_targets, compute_combined_loss

    nat, epi = load_targets(
        ".harness/naturalness_targets.yaml",
        ".harness/epidemiology_targets.yaml",
        skip_unsupported=True,
    )
    result = compute_combined_loss(bundle, nat, epi)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .loss import NaturalnessTarget, EpidemiologyTarget
from .metrics import METRIC_REGISTRY, resolve_metric


def _try_import_yaml():
    """Import PyYAML lazily with a helpful error message."""
    try:
        import yaml  # type: ignore
        return yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load calibration targets. "
            "Install with: uv pip install pyyaml"
        ) from exc


def _normalize_range(raw_range) -> tuple[float, float]:
    """Convert YAML range representation to a (lo, hi) tuple.

    Accepts:
      - list/tuple of two numbers: [0.05, 0.09]
      - dict with min/max keys: {min: 0.05, max: 0.09}
    """
    if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
        return (float(raw_range[0]), float(raw_range[1]))
    if isinstance(raw_range, dict) and "min" in raw_range and "max" in raw_range:
        return (float(raw_range["min"]), float(raw_range["max"]))
    raise ValueError(
        f"Cannot parse range {raw_range!r}. Expected [lo, hi] or {{min, max}}."
    )


def load_naturalness_targets(
    path: str | Path,
    skip_unsupported: bool = False,
) -> list[NaturalnessTarget]:
    """Load a naturalness target YAML file into a list of NaturalnessTarget.

    Args:
        path: path to the YAML file
        skip_unsupported: if True, silently skip entries with unknown metric
            paths (warning printed). If False, raise NotImplementedError.

    Returns:
        List of NaturalnessTarget objects with callable metric resolved.
    """
    yaml = _try_import_yaml()
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    # Accept both "patterns" (naturalness) and "targets" as top-level keys
    raw_targets = doc.get("patterns") or doc.get("targets") or []

    targets: list[NaturalnessTarget] = []
    for entry in raw_targets:
        name = entry.get("name")
        metric_path = entry.get("metric")
        if name is None or metric_path is None:
            raise ValueError(
                f"Target entry missing 'name' or 'metric': {entry}"
            )

        # Skip entries whose metric uses a special target_curve shape —
        # those are not scalar targets and cannot be handled by
        # range-based loss in this pass.
        if "target_curve" in entry and "range" not in entry:
            if skip_unsupported:
                continue
            raise NotImplementedError(
                f"Target {name!r} uses target_curve (not scalar); unsupported in this pass"
            )

        raw_range = entry.get("range")
        if raw_range is None:
            if skip_unsupported:
                continue
            raise ValueError(f"Target {name!r} missing 'range'")

        try:
            metric_fn = resolve_metric(metric_path)
        except NotImplementedError:
            if skip_unsupported:
                continue
            raise

        targets.append(
            NaturalnessTarget(
                name=name,
                metric=metric_fn,
                target_range=_normalize_range(raw_range),
                weight=float(entry.get("weight", 1.0)),
                source=entry.get("source", ""),
                pattern_description=entry.get("pattern", ""),
            )
        )

    return targets


def load_epidemiology_targets(
    path: str | Path,
    skip_unsupported: bool = False,
) -> list[EpidemiologyTarget]:
    """Load an epidemiology target YAML file into a list of EpidemiologyTarget."""
    yaml = _try_import_yaml()
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    raw_targets = doc.get("targets") or []

    targets: list[EpidemiologyTarget] = []
    for entry in raw_targets:
        name = entry.get("name")
        metric_path = entry.get("metric")
        if name is None or metric_path is None:
            raise ValueError(
                f"Target entry missing 'name' or 'metric': {entry}"
            )

        raw_range = entry.get("range")
        if raw_range is None:
            if skip_unsupported:
                continue
            raise ValueError(f"Target {name!r} missing 'range'")

        try:
            metric_fn = resolve_metric(metric_path)
        except NotImplementedError:
            if skip_unsupported:
                continue
            raise

        targets.append(
            EpidemiologyTarget(
                name=name,
                metric=metric_fn,
                target_range=_normalize_range(raw_range),
                weight=float(entry.get("weight", 1.0)),
                source=entry.get("source", ""),
                study_sample=entry.get("note", ""),
            )
        )

    return targets


def load_targets(
    naturalness_path: str | Path,
    epidemiology_path: str | Path,
    skip_unsupported: bool = True,
) -> tuple[list[NaturalnessTarget], list[EpidemiologyTarget]]:
    """Convenience wrapper: load both YAML files at once.

    Default skip_unsupported=True so that running with partial metric
    coverage doesn't block end-to-end calibration smoke tests.
    """
    nat = load_naturalness_targets(naturalness_path, skip_unsupported=skip_unsupported)
    epi = load_epidemiology_targets(epidemiology_path, skip_unsupported=skip_unsupported)
    return nat, epi


# Default YAML locations relative to project root
def default_harness_paths() -> tuple[Path, Path]:
    """Return (naturalness_yaml_path, epidemiology_yaml_path) relative to
    the project root (assuming cwd or this file's repo root).
    """
    # This file lives at src/calibration/loader.py → repo root = ../..
    repo_root = Path(__file__).resolve().parent.parent.parent
    return (
        repo_root / ".harness" / "naturalness_targets.yaml",
        repo_root / ".harness" / "epidemiology_targets.yaml",
    )
