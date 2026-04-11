"""Calibration package — autoresearch loss functions, metrics, and targets.

Design (정리.md §25.13):
  - Tier 1: Naturalness patterns (60% weight)
  - Tier 2: Epidemiological anchors (40% weight)
  - Tier 3: Held-out validation (excluded from loss, used for post-hoc check)

Layers:
  - loss.py    : CalibrationTarget + combined loss functions
  - metrics.py : metric extraction registry + CalibrationResultBundle
  - loader.py  : YAML → runtime target objects
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

from .metrics import (
    ClassHistory,
    CalibrationResultBundle,
    METRIC_REGISTRY,
    resolve_metric,
    supported_metrics,
)

from .loader import (
    load_naturalness_targets,
    load_epidemiology_targets,
    load_targets,
    default_harness_paths,
)

from .adapters import (
    class_result_to_history,
    run_real_bundle,
)

from .prior_predictive import (
    PriorPredictiveReport,
    run_prior_predictive_check,
)

from .proposer import (
    ParameterSpec,
    SearchSpace,
    Trial,
    ProposerBase,
    RandomProposer,
    LatinHypercubeProposer,
    GridProposer,
    BayesianProposer,
    make_proposer,
)

from .orchestrator import (
    EvaluatorProtocol,
    RunState,
    OrchestratorResult,
    AutoresearchOrchestrator,
)

from .applier import (
    ConfigKey,
    parse_key,
    parameter_override,
    DefaultEvaluator,
    build_default_evaluator,
    ParameterOverrideError,
)

from .search_space_loader import (
    InvalidParameterSpecError,
    LoadedSearchSpace,
    load_search_space,
    load_default_search_space,
    build_default_search_space,
    default_student_ranges_path,
)

__all__ = [
    # loss
    "LossResult",
    "CalibrationTarget",
    "NaturalnessTarget",
    "EpidemiologyTarget",
    "compute_naturalness_loss",
    "compute_epidemiology_loss",
    "compute_combined_loss",
    "compute_sparsity_penalty",
    # metrics
    "ClassHistory",
    "CalibrationResultBundle",
    "METRIC_REGISTRY",
    "resolve_metric",
    "supported_metrics",
    # loader
    "load_naturalness_targets",
    "load_epidemiology_targets",
    "load_targets",
    "default_harness_paths",
    # adapters
    "class_result_to_history",
    "run_real_bundle",
    # prior predictive
    "PriorPredictiveReport",
    "run_prior_predictive_check",
    # proposer
    "ParameterSpec",
    "SearchSpace",
    "Trial",
    "ProposerBase",
    "RandomProposer",
    "LatinHypercubeProposer",
    "GridProposer",
    "BayesianProposer",
    "make_proposer",
    # orchestrator
    "EvaluatorProtocol",
    "RunState",
    "OrchestratorResult",
    "AutoresearchOrchestrator",
    # applier
    "ConfigKey",
    "parse_key",
    "parameter_override",
    "DefaultEvaluator",
    "build_default_evaluator",
    "ParameterOverrideError",
    # search space loader
    "InvalidParameterSpecError",
    "LoadedSearchSpace",
    "load_search_space",
    "load_default_search_space",
    "build_default_search_space",
    "default_student_ranges_path",
]
