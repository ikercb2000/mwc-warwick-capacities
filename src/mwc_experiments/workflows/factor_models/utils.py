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
from .types import FactorExperimentResult

def run_factor_experiment(
    factor_frames: dict[str, pd.DataFrame],
    *,
    assets: tuple[str, ...] = EQUITY_TICKERS,
    features: tuple[str, ...] = FACTOR_COLUMNS,
    quick: bool = True,
    model_names: tuple[str, ...] | None = None,
    parameter_grids: Mapping[str, Mapping[str, list[Any]]] | None = None,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> FactorExperimentResult:
    """Fit every factor-model benchmark separately to each equity."""
    candidates = regression_candidates(
        len(features),
        random_state=random_state,
        include_mlp=False,
        include_dummy=False,
        include_regularized_choquet=(model_names is None or "Choquet 2-additive L1" in model_names),
    )
    candidates = apply_parameter_grid_overrides(
        candidates,
        parameter_grids,
        n_features=len(features),
    )
    if quick:
        candidates = quick_candidates(candidates)
    if model_names is not None:
        candidates = {name: candidates[name] for name in model_names}

    metric_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    predictions_by_model: dict[str, dict[str, pd.Series]] = {
        name: {} for name in candidates
    }
    residuals_by_model: dict[str, dict[str, pd.Series]] = {
        name: {} for name in candidates
    }
    in_sample_residuals_by_model: dict[str, dict[str, pd.Series]] = {
        name: {} for name in candidates
    }
    in_sample_metric_rows: list[dict[str, object]] = []
    shapley: dict[str, pd.Series] = {}
    interactions: dict[str, pd.DataFrame] = {}
    fitted_models: dict[tuple[str, str], object] = {}
    splits: dict[str, TemporalSplit] = {}

    for asset in assets:
        frame = factor_frames[asset]
        split = chronological_split(
            frame[list(features)],
            frame["target_excess_loss"],
            horizon=0,
        )
        splits[asset] = split
        benchmark = np.repeat(
            pd.concat([split.y_train, split.y_validation]).mean(),
            len(split.y_test),
        )

        for model_name, candidate in candidates.items():
            if verbose:
                print(f"[factor] {asset} — {model_name}", flush=True)
            try:
                selected = select_regression_model(
                    model_name,
                    candidate,
                    split.X_train,
                    split.y_train,
                    split.X_validation,
                    split.y_validation,
                )
                fitted, refit_runtime = refit_selected(
                    selected,
                    split.X_train,
                    split.y_train,
                    split.X_validation,
                    split.y_validation,
                )
                prediction = pd.Series(
                    fitted.predict(split.X_test),
                    index=split.X_test.index,
                    name=asset,
                )
                X_fit = pd.concat([split.X_train, split.X_validation])
                y_fit = pd.concat([split.y_train, split.y_validation])
                in_sample_prediction = pd.Series(
                    fitted.predict(X_fit),
                    index=X_fit.index,
                    name=asset,
                )
                in_sample_result = regression_metrics(
                    y_fit,
                    in_sample_prediction,
                    benchmark_prediction=np.repeat(y_fit.mean(), len(y_fit)),
                )
                in_sample_result["R2"] = in_sample_result.pop("OOS R2")
                in_sample_metric_rows.append(
                    {
                        "asset": asset,
                        "model": model_name,
                        **in_sample_result,
                    }
                )
                residual = split.y_test - prediction
                in_sample_residual = y_fit - in_sample_prediction
                metrics = regression_metrics(
                    split.y_test,
                    prediction,
                    benchmark_prediction=benchmark,
                )
                metric_rows.append(
                    {
                        "asset": asset,
                        "model": model_name,
                        **metrics,
                        "validation RMSE": selected.validation_score,
                        "selection runtime seconds": selected.runtime_seconds,
                        "refit runtime seconds": refit_runtime,
                        "parameter count": model_parameter_count(fitted),
                    }
                )
                parameter_rows.append(
                    {
                        "asset": asset,
                        "model": model_name,
                        "best parameters": selected.best_params,
                    }
                )
                for message in selected.failures:
                    failure_rows.append({"asset": asset, "model": model_name, "message": message})
                predictions_by_model[model_name][asset] = prediction
                residuals_by_model[model_name][asset] = residual
                in_sample_residuals_by_model[model_name][
                    asset
                ] = in_sample_residual
                fitted_models[(asset, model_name)] = fitted

                if model_name == "Choquet 2-additive":
                    asset_shapley, asset_interactions = capacity_summary(
                        fitted,
                        list(features),
                    )
                    shapley[asset] = asset_shapley
                    interactions[asset] = asset_interactions
            except Exception as error:
                failure_rows.append(
                    {
                        "asset": asset,
                        "model": model_name,
                        "message": f"{type(error).__name__}: {error}",
                    }
                )

    if not metric_rows:
        raise RuntimeError("No factor model completed successfully.")

    metrics = pd.DataFrame(metric_rows).set_index(["asset", "model"]).sort_index()
    parameters = pd.DataFrame(parameter_rows).set_index(["asset", "model"]) if parameter_rows else pd.DataFrame()
    failures = pd.DataFrame(failure_rows)
    predictions = {
        model: pd.concat(series, axis=1).sort_index()
        for model, series in predictions_by_model.items()
        if series
    }
    residuals = {
        model: pd.concat(series, axis=1).sort_index()
        for model, series in residuals_by_model.items()
        if series
    }
    in_sample_residuals = {
        model: pd.concat(series, axis=1).sort_index()
        for model, series in in_sample_residuals_by_model.items()
        if series
    }
    return FactorExperimentResult(
        metrics=metrics,
        in_sample_metrics=(
            pd.DataFrame(in_sample_metric_rows)
            .set_index(["asset", "model"])
            .sort_index()
        ),
        predictions=predictions,
        residuals=residuals,
        in_sample_residuals=in_sample_residuals,
        selected_parameters=parameters,
        failures=failures,
        splits=splits,
        shapley=shapley,
        interactions=interactions,
        fitted_models=fitted_models,
    )
