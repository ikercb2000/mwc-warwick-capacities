"""Metrics domain."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

def out_of_sample_r2(y_true, y_pred, benchmark_prediction=None) -> float:
    actual = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if benchmark_prediction is None:
        benchmark = np.repeat(np.mean(actual), actual.size)
    else:
        benchmark = np.asarray(benchmark_prediction, dtype=float)
    denominator = np.sum((actual - benchmark) ** 2)
    if denominator <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((actual - prediction) ** 2) / denominator)


def tail_weighted_mae(
    y_true,
    y_pred,
    *,
    quantile: float = 0.90,
    tail_weight: float = 4.0,
) -> float:
    actual = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    threshold = np.quantile(actual, quantile)
    weights = np.where(actual >= threshold, tail_weight, 1.0)
    return float(np.average(np.abs(actual - prediction), weights=weights))


def regression_metrics(
    y_true,
    y_pred,
    *,
    benchmark_prediction=None,
) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    return {
        "MSE": float(mean_squared_error(actual, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, prediction))),
        "MAE": float(mean_absolute_error(actual, prediction)),
        "Tail-weighted MAE": tail_weighted_mae(actual, prediction),
        "OOS R2": out_of_sample_r2(actual, prediction, benchmark_prediction),
        "Correlation": float(np.corrcoef(actual, prediction)[0, 1])
        if np.std(prediction) > 0.0 and np.std(actual) > 0.0
        else float("nan"),
    }


def optimal_f1_threshold(y_true, probability) -> float:
    actual = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    candidates = np.unique(np.r_[0.01, probability, 0.99])
    scores = [f1_score(actual, probability >= threshold, zero_division=0) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def classification_metrics(
    y_true,
    probability,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0 - 1e-12)
    threshold_values = np.asarray(threshold, dtype=float)
    if threshold_values.ndim == 0:
        threshold_values = np.repeat(float(threshold_values), actual.size)
    else:
        threshold_values = threshold_values.reshape(-1)
    if threshold_values.size != actual.size:
        raise ValueError("threshold must be scalar or aligned with y_true.")
    predicted = (probability >= threshold_values).astype(int)
    result = {
        **classification_discrimination_metrics(actual, probability),
        "Precision": float(precision_score(actual, predicted, zero_division=0)),
        "Recall": float(recall_score(actual, predicted, zero_division=0)),
        "F1": float(f1_score(actual, predicted, zero_division=0)),
        **probability_calibration_metrics(actual, probability),
        "Threshold": float(threshold_values.mean()),
        "Predicted event rate": float(predicted.mean()),
        # Retained for backwards compatibility with existing result consumers.
        "Observed event rate": float(actual.mean()),
    }
    return result


def classification_discrimination_metrics(y_true, probability) -> dict[str, float]:
    """Return threshold-free ranking metrics for a probabilistic classifier."""
    actual = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if np.unique(actual).size < 2:
        return {"ROC AUC": float("nan"), "PR AUC": float("nan")}
    return {
        "ROC AUC": float(roc_auc_score(actual, probability)),
        "PR AUC": float(average_precision_score(actual, probability)),
    }


def probability_calibration_metrics(y_true, probability) -> dict[str, float]:
    """Return proper scoring rules and marginal probability calibration metrics."""
    actual = np.asarray(y_true, dtype=int)
    probability = np.clip(
        np.asarray(probability, dtype=float),
        1e-12,
        1.0 - 1e-12,
    )
    prevalence = float(actual.mean())
    mean_probability = float(probability.mean())
    gap = mean_probability - prevalence
    return {
        "Brier": float(brier_score_loss(actual, probability)),
        "Log loss": float(log_loss(actual, probability, labels=[0, 1])),
        "Mean predicted probability": mean_probability,
        "Observed event prevalence": prevalence,
        "Calibration gap": gap,
        "Absolute calibration gap": abs(gap),
    }


def high_loss_regime_metrics(
    y_true: pd.Series,
    predictions: pd.DataFrame,
    *,
    quantile: float = 0.90,
) -> pd.DataFrame:
    threshold = y_true.quantile(quantile)
    mask = y_true >= threshold
    rows = {
        model: regression_metrics(y_true.loc[mask], predictions.loc[mask, model])
        for model in predictions.columns
    }
    return pd.DataFrame(rows).T
