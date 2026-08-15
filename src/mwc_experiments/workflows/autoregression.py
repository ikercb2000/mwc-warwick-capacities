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


def _fit_scale(values: pd.Series) -> tuple[float, float]:
    lower = float(values.min())
    upper = float(values.max())
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("Training losses must have a non-degenerate finite range.")
    return lower, upper


def _scale(values: pd.Series, lower: float, upper: float) -> pd.Series:
    return ((values - lower) / (upper - lower)).clip(0.0, 1.0)


def _inverse(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return lower + np.asarray(values, dtype=float) * (upper - lower)


def _sequential_choquet_predictions(
    model: ChoquetAutoRegressor,
    future_scaled: pd.Series,
    *,
    lower: float,
    upper: float,
) -> pd.Series:
    predictions = []
    for value in future_scaled.to_numpy(dtype=float):
        forecast = float(model.predict(fh=[1]).iloc[0])
        predictions.append(forecast)
        next_index = int(model.cutoff[0]) + 1
        model.update(
            pd.Series([value], index=pd.RangeIndex(next_index, next_index + 1)),
            update_params=False,
        )
    return pd.Series(
        _inverse(np.asarray(predictions), lower, upper),
        index=future_scaled.index,
    )


def _autoregressive_prediction(params, history: list[float], lags: int) -> float:
    coefficients = np.asarray(params, dtype=float).reshape(-1)
    if coefficients.size != lags + 1:
        raise ValueError("Expected an intercept followed by one coefficient per lag.")
    value = float(coefficients[0])
    for lag in range(1, lags + 1):
        value += float(coefficients[lag]) * history[-lag]
    return value


def _sequential_linear_ar_predictions(
    fitted,
    future_scaled: pd.Series,
    *,
    lags: int,
    lower: float,
    upper: float,
) -> pd.Series:
    history = list(np.asarray(fitted.model.endog, dtype=float))
    predictions = []
    for value in future_scaled.to_numpy(dtype=float):
        predictions.append(_autoregressive_prediction(fitted.params, history, lags))
        history.append(float(value))
    return pd.Series(
        _inverse(np.asarray(predictions), lower, upper),
        index=future_scaled.index,
    )


def _metrics(actual: pd.Series, prediction: pd.Series) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, prediction))),
        "MAE": float(mean_absolute_error(actual, prediction)),
        "Correlation": float(np.corrcoef(actual, prediction)[0, 1]),
    }


def compare_choquet_autoregression(
    losses: pd.Series,
    *,
    train_end: str = "2018-12-31",
    validation_end: str = "2019-12-31",
    candidate_lags: tuple[int, ...] = (1, 2, 3, 5),
) -> ARComparisonResult:
    """Select lag orders on validation and compare fixed-parameter one-step forecasts.The model is 
    fitted once before each evaluation block. Each one-step forecast is then produced using the
    actually observed history, but parameters are not re-estimated.
    """
    series = pd.Series(losses, copy=True).dropna().sort_index()
    train = series.loc[:train_end]
    validation = series.loc[pd.Timestamp(train_end) + pd.Timedelta(days=1) : validation_end]
    test = series.loc[pd.Timestamp(validation_end) + pd.Timedelta(days=1) :]
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Train, validation and test samples must all be non-empty.")

    lower, upper = _fit_scale(train)
    train_scaled = _scale(train, lower, upper)
    validation_scaled = _scale(validation, lower, upper)

    validation_rows = []
    for lags in candidate_lags:
        ar = AutoReg(train_scaled.to_numpy(), lags=lags, trend="c", old_names=False).fit()
        ar_prediction = _sequential_linear_ar_predictions(
            ar,
            validation_scaled,
            lags=lags,
            lower=lower,
            upper=upper,
        )
        validation_rows.append(
            {"model": "Linear AR", "lags": lags, **_metrics(validation, ar_prediction)}
        )

        choquet = ChoquetAutoRegressor(
            lags=lags,
            sparsity=KAdditivity(order=min(2, lags)),
            solver="scipy",
            solver_options={"options": {"maxiter": 2_000}},
            enforce_stationarity=True,
        )
        choquet.fit(pd.Series(train_scaled.to_numpy(), index=pd.RangeIndex(len(train))))
        choquet_prediction = _sequential_choquet_predictions(
            choquet,
            validation_scaled,
            lower=lower,
            upper=upper,
        )
        validation_rows.append(
            {
                "model": "Choquet AR",
                "lags": lags,
                **_metrics(validation, choquet_prediction),
                "phi": choquet.phi_,
                "AIC": choquet.aic(),
                "BIC": choquet.bic(),
            }
        )

    validation_table = pd.DataFrame(validation_rows).set_index(["model", "lags"])
    selected_lags = {
        model: int(group["RMSE"].idxmin()[1])
        for model, group in validation_table.groupby(level="model")
    }

    estimation = series.loc[:validation_end]
    lower, upper = _fit_scale(estimation)
    estimation_scaled = _scale(estimation, lower, upper)
    test_scaled = _scale(test, lower, upper)

    linear_lag = selected_lags["Linear AR"]
    linear = AutoReg(
        estimation_scaled.to_numpy(),
        lags=linear_lag,
        trend="c",
        old_names=False,
    ).fit()
    linear_prediction = _sequential_linear_ar_predictions(
        linear,
        test_scaled,
        lags=linear_lag,
        lower=lower,
        upper=upper,
    ).rename("Linear AR")

    choquet_lag = selected_lags["Choquet AR"]
    choquet = ChoquetAutoRegressor(
        lags=choquet_lag,
        sparsity=KAdditivity(order=min(2, choquet_lag)),
        solver="scipy",
        solver_options={"options": {"maxiter": 2_000}},
        enforce_stationarity=True,
    )
    choquet.fit(
        pd.Series(
            estimation_scaled.to_numpy(),
            index=pd.RangeIndex(len(estimation_scaled)),
        )
    )
    choquet_prediction = _sequential_choquet_predictions(
        choquet,
        test_scaled,
        lower=lower,
        upper=upper,
    ).rename("Choquet AR")

    historical_mean = pd.Series(
        np.repeat(float(estimation.mean()), len(test)),
        index=test.index,
        name="Historical mean",
    )
    predictions = pd.concat([historical_mean, linear_prediction, choquet_prediction], axis=1)
    metrics = pd.DataFrame(
        {name: _metrics(test, predictions[name]) for name in predictions}
    ).T.sort_values("RMSE")

    raw_shapley = shapley_indices(choquet.capacity_)
    lag_names = [f"lag_{lag}" for lag in range(1, choquet_lag + 1)]
    shapley = pd.Series(
        list(raw_shapley.values()), index=lag_names, name="Shapley importance"
    ).sort_values(ascending=False)
    interactions = pd.DataFrame(
        pairwise_interaction_matrix(choquet.capacity_),
        index=lag_names,
        columns=lag_names,
    )

    return ARComparisonResult(
        validation=validation_table,
        test_metrics=metrics,
        predictions=predictions,
        selected_lags=selected_lags,
        choquet_model=choquet,
        choquet_shapley=shapley,
        choquet_interactions=interactions,
    )
