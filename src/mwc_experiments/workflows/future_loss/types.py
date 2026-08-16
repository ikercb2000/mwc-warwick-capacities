"""Future Loss domain."""

from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any
import numpy as np
import pandas as pd
from mwc_experiments.settings import HORIZONS, MAIN_RISK_FEATURES, RANDOM_STATE
from mwc_experiments.evaluation.interpretation import capacity_summary
from mwc_experiments.evaluation.metrics import regression_metrics
from mwc_experiments.modeling.registries import (
    apply_parameter_grid_overrides,
    regression_candidates,
)
from mwc_experiments.modeling.selection import refit_selected, select_regression_model
from mwc_experiments.modeling.splits import chronological_split
from mwc_experiments.modeling.types import TemporalSplit
from mwc_experiments.workflows.common import model_parameter_count, quick_candidates

@dataclass(slots=True)
class HorizonRegressionResult:
    horizon: int
    split: TemporalSplit
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    selected_parameters: pd.DataFrame
    failures: pd.DataFrame
    shapley: dict[str, pd.Series] = field(default_factory=dict)
    interactions: dict[str, pd.DataFrame] = field(default_factory=dict)
    fitted_models: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class FutureLossExperimentResult:
    portfolio: str
    features: tuple[str, ...]
    horizons: dict[int, HorizonRegressionResult]
