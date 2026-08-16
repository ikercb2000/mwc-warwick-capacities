"""Tail Risk domain."""

from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any
import pandas as pd
from mwc_experiments.settings import (
    CLASSIFICATION_TRAIN_END,
    MAIN_RISK_FEATURES,
    PRIMARY_TAIL_ALPHA,
    RANDOM_STATE,
    VALIDATION_END,
)
from mwc_experiments.evaluation.interpretation import capacity_summary
from mwc_experiments.evaluation.metrics import (
    classification_metrics,
    optimal_f1_threshold,
)
from mwc_experiments.modeling.registries import (
    apply_parameter_grid_overrides,
    classification_candidates,
)
from mwc_experiments.modeling.selection import (
    refit_selected,
    select_classification_model,
)
from mwc_experiments.modeling.splits import chronological_split
from mwc_experiments.modeling.types import TemporalSplit
from mwc_experiments.workflows.common import model_parameter_count, quick_candidates

@dataclass(slots=True)
class TailClassificationResult:
    horizon: int
    alpha: float
    split: TemporalSplit
    final_split: TemporalSplit
    fold_summary: pd.DataFrame
    fold_metrics: pd.DataFrame
    metrics: pd.DataFrame
    discrimination_metrics: pd.DataFrame
    calibration_metrics: pd.DataFrame
    calibration_sample_summary: pd.DataFrame
    probabilities: pd.DataFrame
    thresholds: pd.DataFrame
    selected_parameters: pd.DataFrame
    failures: pd.DataFrame
    orientation_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    shapley_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    shapley: dict[str, pd.Series] = field(default_factory=dict)
    interactions: dict[str, pd.DataFrame] = field(default_factory=dict)
    fitted_models: dict[str, object] = field(default_factory=dict)
    calibrated_models: dict[str, object] = field(default_factory=dict)
