"""Autoregression domain."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from capacities_ml_fin.base.interpretation import (
    pairwise_interaction_matrix,
    shapley_indices,
)
from capacities_ml_fin.ml.models import ChoquetAutoRegressor
from capacities_ml_fin.ml.optimization import KAdditivity

@dataclass(slots=True)
class ARComparisonResult:
    validation: pd.DataFrame
    test_metrics: pd.DataFrame
    predictions: pd.DataFrame
    selected_lags: dict[str, int]
    choquet_model: ChoquetAutoRegressor
    choquet_shapley: pd.Series
    choquet_interactions: pd.DataFrame
