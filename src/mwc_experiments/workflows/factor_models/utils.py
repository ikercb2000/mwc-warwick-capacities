"""Factor-model experiment with comparable fixed and rolling evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from mwc_experiments.evaluation.interpretation import capacity_summary
from mwc_experiments.evaluation.metrics import regression_metrics
from mwc_experiments.modeling.inspection import fitted_q
from mwc_experiments.modeling.registries import (
    apply_parameter_grid_overrides,
    regression_candidates,
)
from mwc_experiments.modeling.selection import (
    refit_selected,
    select_regression_model,
)
from mwc_experiments.modeling.splits import (
    aggregate_walk_forward_split,
    evaluation_splits,
    walk_forward_fold_summary,
)
from mwc_experiments.settings import EQUITY_TICKERS, FACTOR_COLUMNS, RANDOM_STATE
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
    clipping: bool = False,
    evaluation_structure: str = "rolling_5y",
    oos_start: str = "2020-01-01",
    training_window_years: int = 5,
    validation_window_months: int = 12,
    oos_block_years: int = 1,
    verbose: bool = True,
) -> FactorExperimentResult:
    """Fit factor benchmarks using fixed or five-year rolling OOS folds."""
    candidates = regression_candidates(
        len(features),
        random_state=random_state,
        clipping=clipping,
        include_mlp=False,
        include_dummy=False,
        include_regularized_choquet=(
            model_names is None
            or bool(
                {
                    "Choquet 2-additive L1",
                    "Choquet 2-additive scaled-q L1",
                }.intersection(model_names)
            )
        ),
    )
    candidates = apply_parameter_grid_overrides(
        candidates, parameter_grids, n_features=len(features)
    )
    if quick:
        candidates = quick_candidates(candidates)
    if model_names is not None:
        candidates = {name: candidates[name] for name in model_names}

    metric_rows: list[dict[str, object]] = []
    fold_metric_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    in_sample_metric_rows: list[dict[str, object]] = []
    predictions_by_model: dict[str, dict[str, pd.Series]] = {
        name: {} for name in candidates
    }
    residuals_by_model: dict[str, dict[str, pd.Series]] = {
        name: {} for name in candidates
    }
    in_sample_residuals_by_model: dict[str, dict[str, pd.Series]] = {
        name: {} for name in candidates
    }
    fitted_models: dict[tuple[str, str], object] = {}
    splits = {}
    fold_summaries: dict[str, pd.DataFrame] = {}
    shapley: dict[str, pd.Series] = {}
    interactions: dict[str, pd.DataFrame] = {}

    for asset in assets:
        frame = factor_frames[asset]
        folds = evaluation_splits(
            frame[list(features)],
            frame["target_excess_loss"],
            evaluation_structure=evaluation_structure,
            horizon=0,
            oos_start=oos_start,
            training_window_years=training_window_years,
            validation_window_months=validation_window_months,
            oos_block_years=oos_block_years,
        )
        splits[asset] = aggregate_walk_forward_split(folds)
        fold_summaries[asset] = walk_forward_fold_summary(
            folds, evaluation_structure=evaluation_structure
        )
        asset_predictions: dict[str, list[pd.Series]] = {
            name: [] for name in candidates
        }
        asset_residuals: dict[str, list[pd.Series]] = {
            name: [] for name in candidates
        }
        benchmark_chunks: list[pd.Series] = []
        failed_models: set[str] = set()

        for fold in folds:
            split = fold.split
            fit_target = pd.concat([split.y_train, split.y_validation])
            benchmark_chunks.append(
                pd.Series(fit_target.mean(), index=split.y_test.index)
            )
            for model_name, candidate in candidates.items():
                if model_name in failed_models:
                    continue
                if verbose:
                    print(
                        f"[factor {evaluation_structure}, fold={fold.fold}] "
                        f"{asset} — {model_name}",
                        flush=True,
                    )
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
                    asset_predictions[model_name].append(prediction)
                    asset_residuals[model_name].append(split.y_test - prediction)
                    fold_metric_rows.append(
                        {
                            "asset": asset,
                            "fold": fold.fold,
                            "model": model_name,
                            "evaluation_structure": evaluation_structure,
                            **regression_metrics(
                                split.y_test,
                                prediction,
                                benchmark_prediction=np.repeat(
                                    fit_target.mean(), len(split.y_test)
                                ),
                            ),
                            "validation RMSE": selected.validation_score,
                            "selection runtime seconds": selected.runtime_seconds,
                            "refit runtime seconds": refit_runtime,
                            "parameter count": model_parameter_count(fitted),
                        }
                    )
                    parameter_rows.append(
                        {
                            "asset": asset,
                            "fold": fold.fold,
                            "model": model_name,
                            "evaluation_structure": evaluation_structure,
                            "OOS start": fold.oos_start,
                            "OOS end": fold.oos_end,
                            "best parameters": selected.best_params,
                            "fitted q": fitted_q(fitted),
                        }
                    )
                    for message in selected.failures:
                        failure_rows.append(
                            {
                                "asset": asset,
                                "fold": fold.fold,
                                "model": model_name,
                                "message": message,
                            }
                        )
                    if fold is folds[-1]:
                        X_fit = pd.concat([split.X_train, split.X_validation])
                        y_fit = pd.concat([split.y_train, split.y_validation])
                        fitted_models[(asset, model_name)] = fitted
                        fitted_values = pd.Series(
                            fitted.predict(X_fit), index=X_fit.index, name=asset
                        )
                        in_sample_residuals_by_model[model_name][asset] = (
                            y_fit - fitted_values
                        )
                        in_metrics = regression_metrics(
                            y_fit,
                            fitted_values,
                            benchmark_prediction=np.repeat(y_fit.mean(), len(y_fit)),
                        )
                        in_metrics["R2"] = in_metrics.pop("OOS R2")
                        in_sample_metric_rows.append(
                            {
                                "asset": asset,
                                "model": model_name,
                                "evaluation_structure": evaluation_structure,
                                **in_metrics,
                            }
                        )
                        if model_name == "Choquet 1-additive":
                            values, matrix = capacity_summary(fitted, list(features))
                            shapley[asset] = values
                            interactions[asset] = matrix
                except Exception as error:
                    failed_models.add(model_name)
                    asset_predictions[model_name].clear()
                    asset_residuals[model_name].clear()
                    fitted_models.pop((asset, model_name), None)
                    failure_rows.append(
                        {
                            "asset": asset,
                            "fold": fold.fold,
                            "model": model_name,
                            "message": f"{type(error).__name__}: {error}",
                        }
                    )

        benchmark = pd.concat(benchmark_chunks).sort_index()
        for model_name, chunks in asset_predictions.items():
            if len(chunks) != len(folds):
                continue
            prediction = pd.concat(chunks).sort_index()
            residual = pd.concat(asset_residuals[model_name]).sort_index()
            predictions_by_model[model_name][asset] = prediction
            residuals_by_model[model_name][asset] = residual
            model_folds = pd.DataFrame(fold_metric_rows).query(
                "asset == @asset and model == @model_name"
            )
            metric_rows.append(
                {
                    "asset": asset,
                    "model": model_name,
                    "evaluation_structure": evaluation_structure,
                    **regression_metrics(
                        frame.loc[prediction.index, "target_excess_loss"],
                        prediction,
                        benchmark_prediction=benchmark.loc[prediction.index],
                    ),
                    "validation RMSE": float(
                        model_folds["validation RMSE"].mean()
                    ),
                    "selection runtime seconds": float(
                        model_folds["selection runtime seconds"].sum()
                    ),
                    "refit runtime seconds": float(
                        model_folds["refit runtime seconds"].sum()
                    ),
                    "parameter count": int(
                        model_folds["parameter count"].iloc[-1]
                    ),
                    "OOS folds": len(folds),
                }
            )

    if not metric_rows:
        raise RuntimeError("No factor model completed successfully.")
    return FactorExperimentResult(
        evaluation_structure=evaluation_structure,
        metrics=pd.DataFrame(metric_rows).set_index(["asset", "model"]).sort_index(),
        in_sample_metrics=pd.DataFrame(in_sample_metric_rows)
        .set_index(["asset", "model"])
        .sort_index(),
        predictions={
            model: pd.concat(series, axis=1).sort_index()
            for model, series in predictions_by_model.items()
            if series
        },
        residuals={
            model: pd.concat(series, axis=1).sort_index()
            for model, series in residuals_by_model.items()
            if series
        },
        in_sample_residuals={
            model: pd.concat(series, axis=1).sort_index()
            for model, series in in_sample_residuals_by_model.items()
            if series
        },
        selected_parameters=(
            pd.DataFrame(parameter_rows)
            .set_index(["asset", "fold", "model"])
            .sort_index()
        ),
        failures=pd.DataFrame(failure_rows),
        splits=splits,
        fold_summaries=fold_summaries,
        fold_metrics=(
            pd.DataFrame(fold_metric_rows)
            .set_index(["asset", "fold", "model"])
            .sort_index()
        ),
        shapley=shapley,
        interactions=interactions,
        fitted_models=fitted_models,
    )