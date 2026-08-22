"""Factor Models domain."""

from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any
import numpy as np
import pandas as pd
from mwc_experiments.settings import EQUITY_TICKERS, FACTOR_COLUMNS, RANDOM_STATE
from mwc_experiments.evaluation.interpretation import capacity_summary
from mwc_experiments.evaluation.metrics import regression_metrics
from mwc_experiments.modeling.registries import (
    apply_parameter_grid_overrides,
    regression_candidates,
)
from mwc_experiments.modeling.selection import (
    refit_selected,
    select_regression_model,
)
from mwc_experiments.modeling.splits import chronological_split
from mwc_experiments.modeling.types import TemporalSplit
from mwc_experiments.workflows.common import model_parameter_count, quick_candidates

@dataclass(slots=True)
class FactorExperimentResult:
    evaluation_structure: str
    metrics: pd.DataFrame
    in_sample_metrics: pd.DataFrame
    predictions: dict[str, pd.DataFrame]
    residuals: dict[str, pd.DataFrame]
    in_sample_residuals: dict[str, pd.DataFrame]
    selected_parameters: pd.DataFrame
    failures: pd.DataFrame
    splits: dict[str, TemporalSplit]
    fold_summaries: dict[str, pd.DataFrame]
    fold_metrics: pd.DataFrame
    shapley: dict[str, pd.Series] = field(default_factory=dict)
    interactions: dict[str, pd.DataFrame] = field(default_factory=dict)
    fitted_models: dict[tuple[str, str], object] = field(default_factory=dict)

    def residual_covariance(self, model: str) -> pd.DataFrame:
        return self.residuals[model].cov()
