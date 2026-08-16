"""Inference domain."""

from __future__ import annotations
import numpy as np
import pandas as pd
from capacities_ml_fin.risk import block_bootstrap_interval, hac_mean_interval

def squared_loss_differential(y_true, first_prediction, second_prediction) -> np.ndarray:
    actual = np.asarray(y_true, dtype=float)
    first = np.asarray(first_prediction, dtype=float)
    second = np.asarray(second_prediction, dtype=float)
    return (actual - first) ** 2 - (actual - second) ** 2


def absolute_loss_differential(y_true, first_prediction, second_prediction) -> np.ndarray:
    actual = np.asarray(y_true, dtype=float)
    first = np.asarray(first_prediction, dtype=float)
    second = np.asarray(second_prediction, dtype=float)
    return np.abs(actual - first) - np.abs(actual - second)


def hac_model_comparison(
    y_true,
    predictions: pd.DataFrame,
    *,
    reference: str,
    loss: str = "squared",
    max_lags: int = 10,
) -> pd.DataFrame:
    if reference not in predictions:
        raise KeyError(f"Unknown reference model {reference!r}.")
    differential_function = (
        squared_loss_differential if loss == "squared" else absolute_loss_differential
    )
    rows = []
    for model in predictions:
        if model == reference:
            continue
        differential = differential_function(
            y_true,
            predictions[model],
            predictions[reference],
        )
        result = hac_mean_interval(differential, max_lags=max_lags)
        rows.append(
            {
                "model": model,
                "reference": reference,
                "mean loss difference": result.mean,
                "standard error": result.standard_error,
                "lower 95%": result.lower_bound,
                "upper 95%": result.upper_bound,
                "favors model": result.mean < 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("mean loss difference")


def block_bootstrap_metric_interval(
    values,
    *,
    block_size: int = 20,
    n_resamples: int = 1_000,
    random_state: int = 42,
) -> tuple[float, float]:
    return block_bootstrap_interval(
        np.asarray(values, dtype=float),
        block_size=block_size,
        n_resamples=n_resamples,
        random_state=random_state,
    )
