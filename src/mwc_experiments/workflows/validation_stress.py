"""Compare model selection with and without stressed validation blocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid

from mwc_experiments.evaluation.metrics import regression_metrics
from mwc_experiments.modeling.registries import regression_candidates
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


def _purge_sample_end(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Remove targets whose forward window crosses the next temporal boundary."""
    if horizon == 0:
        return X, y
    if len(X) <= horizon:
        raise ValueError("Not enough observations to purge the forecast horizon.")
    return X.iloc[:-horizon], y.iloc[:-horizon]


def compare_validation_stress_regimes(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_names: tuple[str, ...],
    horizon: int = 0,
    quick: bool = True,
    random_state: int = RANDOM_STATE,
    validation_start: str = VALIDATION_STRESS_VALIDATION_START,
    validation_end: str = VALIDATION_STRESS_END,
    stress_start: str = VALIDATION_STRESS_START,
    stress_end: str = VALIDATION_STRESS_PERIOD_END,
) -> ValidationStressComparisonResult:
    """Select configurations from expanding, annual pseudo-OOS predictions.

    Candidate configurations generate the same 2018--2021 walk-forward
    predictions in both regimes. One selection score uses every prediction and
    the other excludes 2020--2021. Both selected configurations are finally
    refitted on the identical sample through 2021 and evaluated after 2021.
    """
    if horizon < 0:
        raise ValueError("horizon must be non-negative.")
    common = X.join(y.rename("__target__"), how="inner").dropna().sort_index()
    X_clean = common[X.columns]
    y_clean = common["__target__"]
    validation_start_date = pd.Timestamp(validation_start)
    validation_end_date = pd.Timestamp(validation_end)
    validation_mask = X_clean.index.to_series().between(
        validation_start_date,
        validation_end_date,
        inclusive="both",
    )
    X_validation = X_clean.loc[validation_mask]
    y_validation = y_clean.loc[validation_mask]
    X_validation, y_validation = _purge_sample_end(
        X_validation,
        y_validation,
        horizon,
    )
    X_test = X_clean.loc[X_clean.index > validation_end_date]
    y_test = y_clean.loc[X_test.index]
    X_final = X_clean.loc[X_clean.index <= validation_end_date]
    y_final = y_clean.loc[X_final.index]
    X_final, y_final = _purge_sample_end(X_final, y_final, horizon)
    if min(len(X_validation), len(X_test), len(X_final)) == 0:
        raise ValueError("A walk-forward validation partition is empty.")

    candidates = regression_candidates(
        X.shape[1],
        random_state=random_state,
        include_mlp="MLP" in model_names,
        include_dummy="Historical mean" in model_names,
        include_regularized_choquet=(
            "Choquet 2-additive L1" in model_names
        ),
    )
    candidates = {name: candidates[name] for name in model_names}
    if quick:
        candidates = quick_candidates(candidates)

    stress_dates = X_validation.index.to_series().between(
        pd.Timestamp(stress_start),
        pd.Timestamp(stress_end),
        inclusive="both",
    )
    regime_masks = {
        "including stress": pd.Series(True, index=X_validation.index),
        "excluding 2020-2021": ~stress_dates,
    }
    if not regime_masks["excluding 2020-2021"].any():
        raise ValueError("No calm walk-forward observations remain.")

    validation_years = sorted(X_validation.index.year.unique())
    configuration_scores: dict[
        str,
        list[tuple[dict[str, object], dict[str, float]]],
    ] = {}
    failures: list[dict[str, str]] = []
    for model, candidate in candidates.items():
        configurations: list[tuple[dict[str, object], dict[str, float]]] = []
        grid = list(ParameterGrid(candidate.param_grid)) if candidate.param_grid else [{}]
        for parameters in grid:
            predictions = pd.Series(np.nan, index=X_validation.index)
            try:
                for year in validation_years:
                    block_index = X_validation.index[X_validation.index.year == year]
                    if block_index.empty:
                        continue
                    X_fold_train = X_clean.loc[X_clean.index < block_index.min()]
                    y_fold_train = y_clean.loc[X_fold_train.index]
                    X_fold_train, y_fold_train = _purge_sample_end(
                        X_fold_train,
                        y_fold_train,
                        horizon,
                    )
                    estimator = clone(candidate.estimator).set_params(**parameters)
                    estimator.fit(X_fold_train, y_fold_train)
                    predictions.loc[block_index] = estimator.predict(
                        X_validation.loc[block_index]
                    )
                if predictions.isna().any():
                    raise ValueError("Walk-forward predictions contain missing values.")
                scores = {
                    regime: float(
                        np.sqrt(
                            mean_squared_error(
                                y_validation.loc[mask],
                                predictions.loc[mask],
                            )
                        )
                    )
                    for regime, mask in regime_masks.items()
                }
                configurations.append((dict(parameters), scores))
            except Exception as error:
                failures.append(
                    {
                        "regime": "walk-forward",
                        "model": model,
                        "message": (
                            f"params={parameters}: {type(error).__name__}: {error}"
                        ),
                    }
                )
        configuration_scores[model] = configurations

    benchmark = np.repeat(y_final.mean(), len(y_test))
    rows: list[dict[str, object]] = []
    for regime, mask in regime_masks.items():
        for model, candidate in candidates.items():
            configurations = configuration_scores[model]
            if not configurations:
                continue
            parameters, scores = min(
                configurations,
                key=lambda configuration: configuration[1][regime],
            )
            try:
                fitted = clone(candidate.estimator).set_params(**parameters)
                fitted.fit(X_final, y_final)
                prediction = fitted.predict(X_test)
                rows.append(
                    {
                        "regime": regime,
                        "model": model,
                        "validation observations": int(mask.sum()),
                        "validation folds": len(validation_years),
                        "validation RMSE": scores[regime],
                        "best parameters": parameters,
                        **regression_metrics(
                            y_test,
                            prediction,
                            benchmark_prediction=benchmark,
                        ),
                    }
                )
            except Exception as error:
                failures.append(
                    {
                        "regime": regime,
                        "model": model,
                        "message": f"{type(error).__name__}: {error}",
                    }
                )
    if not rows:
        raise RuntimeError("No validation-stress comparison model succeeded.")

    metrics = pd.DataFrame(rows)
    metrics["validation rank"] = metrics.groupby("regime")[
        "validation RMSE"
    ].rank(method="min")
    metrics["test rank"] = metrics.groupby("regime")["RMSE"].rank(
        method="min"
    )
    metrics["selected by validation"] = metrics["validation rank"] == 1.0
    summary_rows: list[dict[str, object]] = []
    for regime, regime_metrics in metrics.groupby("regime"):
        selected = regime_metrics.loc[
            regime_metrics["validation RMSE"].idxmin()
        ]
        oracle = regime_metrics.loc[regime_metrics["RMSE"].idxmin()]
        summary_rows.append(
            {
                "regime": regime,
                "validation-selected model": selected["model"],
                "selected validation RMSE": selected["validation RMSE"],
                "selected model test RMSE": selected["RMSE"],
                "best test model": oracle["model"],
                "best test RMSE": oracle["RMSE"],
                "selection regret": selected["RMSE"] - oracle["RMSE"],
            }
        )
    selection_summary = pd.DataFrame(summary_rows).set_index("regime")
    metrics = metrics.set_index(["regime", "model"]).sort_index()

    initial_index = X_clean.index[X_clean.index < validation_start_date]
    if horizon:
        initial_index = initial_index[:-horizon]
    calm_index = X_validation.index[
        regime_masks["excluding 2020-2021"].to_numpy()
    ]
    samples = {
        "initial training": initial_index,
        "walk-forward validation": X_validation.index,
        "validation excluding 2020-2021": calm_index,
        "final refit": X_final.index,
        "test": X_test.index,
    }
    sample_summary = pd.DataFrame(
        [
            {
                "sample": name,
                "observations": len(index),
                "start": index.min(),
                "end": index.max(),
            }
            for name, index in samples.items()
        ]
    ).set_index("sample")
    return ValidationStressComparisonResult(
        metrics=metrics,
        selection_summary=selection_summary,
        failures=pd.DataFrame(failures),
        sample_summary=sample_summary,
    )
