"""Teacher-side perception noise (Phase 6 slice 5).

This module introduces a narrow, explicit noise layer on the
teacher's encoding of *already-observable* behaviors. It does NOT
read latent student state, does NOT mutate the simulator, and
does NOT regenerate behaviors from hidden variables. It is purely
a teacher-side perception step that sits between:

    (1) the classroom's visible behavior output
         (e.g. ``StudentSummary.behaviors``, already filtered to
         the high-visibility vocabulary)

and

    (2) every teacher-facing sink that consumes those behaviors:
         * ``TeacherObservationBatch`` evidence accumulation
         * ``TeacherMemory`` observation records
         * the delayed-feedback queue's pre / post visible
           snapshots

Supported noise kinds
---------------------
Two, both deliberately small:

* **dropout** — with probability ``observation_dropout_prob`` per
  behavior, the teacher fails to encode it at all. The raw
  classroom behavior is not affected; only the teacher-facing
  copy loses the entry.

* **confusion** — with probability ``observation_confusion_prob``
  per surviving behavior, the teacher mis-reads it as a
  DIFFERENT behavior drawn uniformly from the same disruptive
  vocabulary. Non-disruptive behaviors pass through untouched
  because there is no meaningful "near" alternative in the
  current vocabulary.

Order
-----
For each input behavior:
  1. roll dropout → drop and continue
  2. if surviving, roll confusion → pick a replacement from the
     disruptive pool (excluding self)

Two behaviors are independent: the RNG advances once per roll,
so the caller gets a fixed deterministic trace under a fixed
seed.

Configuration
-------------
``TeacherNoiseConfig`` is a small frozen dataclass. The default
(``observation_dropout_prob=0.0, observation_confusion_prob=0.0``)
is a no-op: ``apply_observation_noise`` returns the input
unchanged, which lets existing tests and the default simulator
path stay identical to the pre-noise baseline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable


#: The visible disruptive vocabulary confusion operates over.
#: Mirrors ``_DISRUPTIVE_VISIBLE_BEHAVIORS`` in
#: ``orchestrator_v2`` — duplicated here (rather than imported)
#: to keep this module import-cycle-free. A single source of
#: truth for the set would be cleaner but out of scope here.
DISRUPTIVE_VOCABULARY: tuple[str, ...] = (
    "out_of_seat",
    "calling_out",
    "interrupting",
    "excessive_talking",
    "running_in_classroom",
    "fidgeting",
    "emotional_outburst",
)

_DISRUPTIVE_VOCABULARY_SET: frozenset[str] = frozenset(DISRUPTIVE_VOCABULARY)


@dataclass(frozen=True)
class TeacherNoiseConfig:
    """Deterministic teacher perception noise knobs.

    Fields:
      observation_dropout_prob:
        Per-behavior probability that the teacher fails to encode
        a visible behavior at all. Applied before confusion.
        Clamped to ``[0.0, 1.0]`` by ``__post_init__``.
      observation_confusion_prob:
        Per-behavior probability that a disruptive visible
        behavior (after dropout) is mis-read as a different
        disruptive behavior. Non-disruptive behaviors pass
        through unchanged.
    """

    observation_dropout_prob: float = 0.0
    observation_confusion_prob: float = 0.0

    def __post_init__(self) -> None:
        # Use object.__setattr__ because the dataclass is frozen.
        object.__setattr__(
            self,
            "observation_dropout_prob",
            _clamp01(self.observation_dropout_prob),
        )
        object.__setattr__(
            self,
            "observation_confusion_prob",
            _clamp01(self.observation_confusion_prob),
        )

    @property
    def is_disabled(self) -> bool:
        """True when both probabilities are zero (no-op path)."""
        return (
            self.observation_dropout_prob == 0.0
            and self.observation_confusion_prob == 0.0
        )

    def as_dict(self) -> dict:
        return {
            "observation_dropout_prob": self.observation_dropout_prob,
            "observation_confusion_prob": self.observation_confusion_prob,
        }


def _clamp01(value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def apply_observation_noise(
    visible_behaviors: Iterable[str],
    rng: random.Random,
    config: TeacherNoiseConfig,
) -> tuple[str, ...]:
    """Apply teacher perception noise to one batch of visible behaviors.

    Pure function: reads nothing but its arguments, mutates
    nothing, returns a new tuple in original encounter order.

    The teacher's perception is stateless between calls: two
    observations of the same student on different turns can
    produce different noisy traces even though the underlying
    behavior set is identical. This is intentional — it models
    "the teacher noticed it once but missed it later."

    Args:
      visible_behaviors: iterable of behavior strings the
        classroom already marked as teacher-visible (e.g. from
        ``StudentSummary.behaviors``). Ordering is preserved
        for the survivors.
      rng: a ``random.Random`` instance owned by the caller.
        The orchestrator seeds a dedicated RNG deterministically
        from the master seed so noisy runs are reproducible.
      config: ``TeacherNoiseConfig``. When both probabilities
        are zero, the function is a pass-through (returns
        ``tuple(visible_behaviors)``) with NO RNG calls — this
        keeps legacy callers bit-identical to the previous path.

    Returns:
      Tuple of noisy behaviors. Never contains a latent field;
      every element is either the original behavior string or a
      disruptive-vocabulary alternative.
    """
    source = tuple(visible_behaviors or ())
    if config.is_disabled or not source:
        return source

    dropout_p = config.observation_dropout_prob
    confusion_p = config.observation_confusion_prob

    out: list[str] = []
    for behavior in source:
        # 1. dropout roll — applied to every behavior regardless
        # of whether it is in the disruptive vocabulary.
        if dropout_p > 0.0 and rng.random() < dropout_p:
            continue

        # 2. confusion roll — only meaningful for disruptive
        # behaviors (the pool is the disruptive vocabulary).
        if (
            confusion_p > 0.0
            and behavior in _DISRUPTIVE_VOCABULARY_SET
            and rng.random() < confusion_p
        ):
            alternatives = [
                b for b in DISRUPTIVE_VOCABULARY if b != behavior
            ]
            if alternatives:
                out.append(rng.choice(alternatives))
                continue

        out.append(behavior)

    return tuple(out)


__all__ = [
    "DISRUPTIVE_VOCABULARY",
    "TeacherNoiseConfig",
    "apply_observation_noise",
]
