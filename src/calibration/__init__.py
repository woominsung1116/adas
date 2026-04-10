"""Calibration package — autoresearch loss functions and targets.

Design (정리.md §25.13):
  - Tier 1: Naturalness patterns (60% weight)
  - Tier 2: Epidemiological anchors (40% weight)
  - Tier 3: Held-out validation (excluded from loss, used for post-hoc check)

All losses return a single scalar. Autoresearch minimizes the combined loss.
"""

from .loss import (
    LossResult,
    CalibrationTarget,
    NaturalnessTarget,
    EpidemiologyTarget,
    compute_naturalness_loss,
    compute_epidemiology_loss,
    compute_combined_loss,
    compute_sparsity_penalty,
)

__all__ = [
    "LossResult",
    "CalibrationTarget",
    "NaturalnessTarget",
    "EpidemiologyTarget",
    "compute_naturalness_loss",
    "compute_epidemiology_loss",
    "compute_combined_loss",
    "compute_sparsity_penalty",
]
