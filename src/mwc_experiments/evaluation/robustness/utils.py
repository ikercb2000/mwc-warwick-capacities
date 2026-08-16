"""Robustness domain."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from mwc_experiments.evaluation.metrics import regression_metrics
from .types import EmpiricalStressDefinition

def fit_empirical_stress_definition(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    feature_lower_quantile: float = 0.005,
    feature_upper_quantile: float = 0.995,
    loss_quantile: float = 0.99,
) -> EmpiricalStressDefinition:
    """Fit raw-data stress thresholds without using the held-out test sample."""
    if not 0.0 <= feature_lower_quantile < feature_upper_quantile <= 1.0:
        raise ValueError("Feature quantiles must satisfy 0 <= lower < upper <= 1.")
    if not 0.0 < loss_quantile < 1.0:
        raise ValueError("loss_quantile must lie strictly between zero and one.")
    common = X.join(y.rename("__loss__"), how="inner").dropna()
    if common.empty:
        raise ValueError("Cannot define stress thresholds from an empty sample.")
    features = common[X.columns]
    return EmpiricalStressDefinition(
        feature_lower=features.quantile(feature_lower_quantile),
        feature_upper=features.quantile(feature_upper_quantile),
        loss_threshold=float(common["__loss__"].quantile(loss_quantile)),
    )


def clipping_diagnostics(
    model: BaseEstimator,
    X: pd.DataFrame,
    *,
    sample: str,
) -> pd.DataFrame:
    """Compare raw feature extremes with a fitted pipeline's clipping bounds."""
    if not isinstance(model, Pipeline):
        raise TypeError("Expected a fitted sklearn Pipeline.")
    preprocessor = model.named_steps.get("preprocessor")
    if not isinstance(preprocessor, Pipeline):
        raise AttributeError("The fitted model has no preprocessing pipeline.")
    clipper = preprocessor.named_steps.get("clip")
    if clipper is None:
        raise AttributeError("The fitted model has no clipping transformer.")
    check_is_fitted(clipper, ["lower_bounds_", "upper_bounds_"])
    if X.shape[1] != len(clipper.lower_bounds_):
        raise ValueError("X does not match the fitted clipper dimension.")

    names = list(X.columns)
    lower = pd.Series(clipper.lower_bounds_, index=names)
    upper = pd.Series(clipper.upper_bounds_, index=names)
    below = X.lt(lower, axis="columns")
    above = X.gt(upper, axis="columns")
    clipped = below | above
    return pd.DataFrame(
        {
            "sample": sample,
            "raw minimum": X.min(),
            "lower clipping bound": lower,
            "raw maximum": X.max(),
            "upper clipping bound": upper,
            "below-bound observations": below.sum().astype(int),
            "above-bound observations": above.sum().astype(int),
            "clipped observations": clipped.sum().astype(int),
            "clipped rate": clipped.mean(),
        }
    )


def regression_regime_metrics(
    y_true: pd.Series,
    predictions: pd.DataFrame,
    stress_mask: pd.Series,
) -> pd.DataFrame:
    """Evaluate every model on one shared set of raw-data stress dates."""
    mask = stress_mask.reindex(y_true.index).fillna(False).astype(bool)
    if not mask.any():
        columns = [
            "stress observations",
            "MSE",
            "RMSE",
            "MAE",
            "Tail-weighted MAE",
            "OOS R2",
            "Correlation",
        ]
        return pd.DataFrame(
            {
                column: 0 if column == "stress observations" else np.nan
                for column in columns
            },
            index=predictions.columns,
        ).rename_axis("model")
    aligned = predictions.reindex(y_true.index)
    rows = {
        model: regression_metrics(
            y_true.loc[mask],
            aligned.loc[mask, model],
        )
        for model in aligned.columns
    }
    result = pd.DataFrame(rows).T
    result.insert(0, "stress observations", int(mask.sum()))
    return result.sort_values("RMSE").rename_axis("model")


def regression_estimation_robustness(
    models: Mapping[str, BaseEstimator],
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    full_predictions: pd.DataFrame,
    *,
    fit_extreme_mask: pd.Series,
    test_stress_mask: pd.Series,
) -> pd.DataFrame:
    """Compare full fits with fits excluding the same empirical extremes."""
    fit_mask = fit_extreme_mask.reindex(X_fit.index).fillna(False).astype(bool)
    stress_mask = test_stress_mask.reindex(X_test.index).fillna(False).astype(bool)
    retained = ~fit_mask
    if int(retained.sum()) < 10:
        raise ValueError("Too few non-extreme observations remain for refitting.")

    def rmse(actual: pd.Series, prediction: np.ndarray) -> float:
        values = actual.to_numpy(dtype=float) - np.asarray(prediction, dtype=float)
        return float(np.sqrt(np.mean(values**2)))

    rows: list[dict[str, object]] = []
    for name, model in models.items():
        if name not in full_predictions:
            continue
        full = full_predictions[name].reindex(X_test.index).to_numpy(dtype=float)
        row: dict[str, object] = {
            "model": name,
            "fit observations": len(X_fit),
            "extreme fit observations removed": int(fit_mask.sum()),
            "extreme fit rate": float(fit_mask.mean()),
            "test observations": len(X_test),
            "stress test observations": int(stress_mask.sum()),
        }
        try:
            reduced = clone(model).fit(X_fit.loc[retained], y_fit.loc[retained])
            reduced_prediction = np.asarray(
                reduced.predict(X_test),
                dtype=float,
            ).reshape(-1)
            change = reduced_prediction - full
            row.update(
                {
                    "full-fit test RMSE": rmse(y_test, full),
                    "reduced-fit test RMSE": rmse(y_test, reduced_prediction),
                    "test RMSE change": (
                        rmse(y_test, reduced_prediction) - rmse(y_test, full)
                    ),
                    "mean absolute prediction change": float(
                        np.mean(np.abs(change))
                    ),
                    "maximum absolute prediction change": float(
                        np.max(np.abs(change))
                    ),
                    "full-fit stress RMSE": (
                        rmse(y_test.loc[stress_mask], full[stress_mask.to_numpy()])
                        if stress_mask.any()
                        else float("nan")
                    ),
                    "reduced-fit stress RMSE": (
                        rmse(
                            y_test.loc[stress_mask],
                            reduced_prediction[stress_mask.to_numpy()],
                        )
                        if stress_mask.any()
                        else float("nan")
                    ),
                    "stress mean absolute prediction change": (
                        float(np.mean(np.abs(change[stress_mask.to_numpy()])))
                        if stress_mask.any()
                        else float("nan")
                    ),
                    "failure": "",
                }
            )
        except Exception as error:
            row["failure"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    if not rows:
        raise ValueError("No fitted models matched the prediction columns.")
    return pd.DataFrame(rows).set_index("model")
