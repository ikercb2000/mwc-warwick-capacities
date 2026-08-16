"""Validation Stress domain."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from mwc_experiments.evaluation.metrics import regression_metrics
from mwc_experiments.modeling.registries import (
    apply_parameter_grid_overrides,
    regression_candidates,
)
from mwc_experiments.settings import (
    RANDOM_STATE,
    VALIDATION_STRESS_END,
    VALIDATION_STRESS_PERIOD_END,
    VALIDATION_STRESS_START,
    VALIDATION_STRESS_VALIDATION_START,
)
from mwc_experiments.workflows.common import quick_candidates

@dataclass(slots=True)
class ValidationStressComparisonResult:
    """Hold common-test results for both walk-forward validation regimes."""

    metrics: pd.DataFrame
    selection_summary: pd.DataFrame
    failures: pd.DataFrame
    sample_summary: pd.DataFrame
