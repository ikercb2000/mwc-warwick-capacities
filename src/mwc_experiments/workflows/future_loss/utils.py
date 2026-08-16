"""Future Loss domain."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from mwc_experiments.evaluation.interpretation import (
    capacity_summary,
    orientation_table,
)
from mwc_experiments.evaluation.metrics import regression_metrics
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
    rolling_walk_forward_splits,
    walk_forward_fold_summary,
)
from mwc_experiments.settings import HORIZONS, MAIN_RISK_FEATURES, RANDOM_STATE
from mwc_experiments.workflows.common import (
    model_parameter_count,
    quick_candidates,
)

from .types import FutureLossExperimentResult, HorizonRegressionResult


def run_future_loss_experiment(
    dataset: pd.DataFrame,
    *,
    portfolio: str = "equal",
    features: tuple[str, ...] = MAIN_RISK_FEATURES,
    horizons: tuple[int, ...] = HORIZONS,
    quick: bool = True,
    model_names: tuple[str, ...] | None = None,
    parameter_grids: Mapping[str, Mapping[str, list[Any]]] | None = None,
    random_state: int = RANDOM_STATE,
    oos_start: str = "2020-01-01",
    training_window_years: int = 5,
    validation_window_months: int = 12,
    oos_block_years: int = 1,
    verbose: bool = True,
) -> FutureLossExperimentResult:
    """Run rolling walk-forward loss forecasts and concatenate every OOS block."""
    results: dict[int, HorizonRegressionResult] = {}

    for horizon in horizons:
        target = f"future_loss_h{horizon}"
        folds = list(
            rolling_walk_forward_splits(
                dataset[list(features)],
                dataset[target].astype(float),
                oos_start=oos_start,
                training_window_years=training_window_years,
                validation_window_months=validation_window_months,
                oos_block_years=oos_block_years,
                horizon=horizon,
            )
        )
        candidates = regression_candidates(
            len(features),
            random_state=random_state,
            include_mlp=True,
            include_dummy=True,
            include_regularized_choquet=(
                model_names is None or "Choquet 2-additive L1" in model_names
            ),
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

        prediction_chunks: dict[str, list[pd.Series]] = {
            name: [] for name in candidates
        }
        benchmark_chunks: list[pd.Series] = []
        fold_metric_rows: list[dict[str, object]] = []
        parameter_rows: list[dict[str, object]] = []
        failure_rows: list[dict[str, object]] = []
        orientation_rows: list[pd.DataFrame] = []
        shapley_rows: list[pd.DataFrame] = []
        failed_models: set[str] = set()
        fitted_models: dict[str, object] = {}
        shapley: dict[str, pd.Series] = {}
        interactions: dict[str, pd.DataFrame] = {}

        for fold in folds:
            split = fold.split
            training_mean = pd.concat(
                [split.y_train, split.y_validation]
            ).mean()
            benchmark_chunks.append(
                pd.Series(training_mean, index=split.y_test.index)
            )
            for name, candidate in candidates.items():
                if name in failed_models:
                    continue
                if verbose:
                    print(
                        f"[future loss h={horizon}, fold={fold.fold}] {name}",
                        flush=True,
                    )
                try:
                    selected = select_regression_model(
                        name,
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
                        name=name,
                    )
                    prediction_chunks[name].append(prediction)
                    fitted_models[name] = fitted
                    fold_metric_rows.append(
                        {
                            "fold": fold.fold,
                            "model": name,
                            **regression_metrics(
                                split.y_test,
                                prediction,
                                benchmark_prediction=np.repeat(
                                    training_mean,
                                    len(split.y_test),
                                ),
                            ),
                            "validation RMSE": selected.validation_score,
                            "selection runtime seconds": selected.runtime_seconds,
                            "refit runtime seconds": refit_runtime,
                            "parameter count": model_parameter_count(fitted),
                            "best parameters": selected.best_params,
                        }
                    )
                    parameter_rows.append(
                        {
                            "fold": fold.fold,
                            "model": name,
                            "OOS start": fold.oos_start,
                            "OOS end": fold.oos_end,
                            "best parameters": selected.best_params,
                        }
                    )
                    for message in selected.failures:
                        failure_rows.append(
                            {
                                "fold": fold.fold,
                                "model": name,
                                "message": message,
                            }
                        )
                    try:
                        orientations = orientation_table(fitted).rename_axis(
                            "feature"
                        ).reset_index()
                        orientations.insert(0, "model", name)
                        orientations.insert(0, "fold", fold.fold)
                        orientation_rows.append(orientations)
                    except AttributeError:
                        pass
                    if name.startswith("Choquet"):
                        values, matrix = capacity_summary(fitted, list(features))
                        values_frame = values.rename("Shapley importance").rename_axis(
                            "feature"
                        ).reset_index()
                        values_frame.insert(0, "model", name)
                        values_frame.insert(0, "fold", fold.fold)
                        shapley_rows.append(values_frame)
                        if fold is folds[-1]:
                            shapley[name] = values
                            interactions[name] = matrix
                except Exception as error:
                    failed_models.add(name)
                    prediction_chunks[name].clear()
                    fitted_models.pop(name, None)
                    failure_rows.append(
                        {
                            "fold": fold.fold,
                            "model": name,
                            "message": f"{type(error).__name__}: {error}",
                        }
                    )

        successful = {
            name: pd.concat(chunks).sort_index()
            for name, chunks in prediction_chunks.items()
            if len(chunks) == len(folds)
        }
        if not successful:
            raise RuntimeError(
                f"No model completed every walk-forward fold for horizon {horizon}."
            )
        predictions = pd.concat(successful, axis=1).sort_index()
        split = aggregate_walk_forward_split(folds)
        benchmark = pd.concat(benchmark_chunks).sort_index().loc[predictions.index]
        fold_metrics = pd.DataFrame(fold_metric_rows)
        fold_metrics = fold_metrics[
            fold_metrics["model"].isin(successful)
        ].set_index(["fold", "model"]).sort_index()
        metric_rows: list[dict[str, object]] = []
        for name, prediction in successful.items():
            model_folds = fold_metrics.xs(name, level="model")
            metric_rows.append(
                {
                    "model": name,
                    **regression_metrics(
                        split.y_test.loc[prediction.index],
                        prediction,
                        benchmark_prediction=benchmark,
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
                    "parameter count": model_parameter_count(
                        fitted_models[name]
                    ),
                    "OOS folds": len(folds),
                }
            )
        final_fold_number = folds[-1].fold
        final_models = {
            name: model
            for name, model in fitted_models.items()
            if name in successful
        }
        results[horizon] = HorizonRegressionResult(
            horizon=horizon,
            split=split,
            final_split=folds[-1].split,
            fold_summary=walk_forward_fold_summary(folds),
            fold_metrics=fold_metrics,
            metrics=pd.DataFrame(metric_rows).set_index("model").sort_values(
                "RMSE"
            ),
            predictions=predictions,
            selected_parameters=(
                pd.DataFrame(parameter_rows)
                .query("model in @successful")
                .set_index(["fold", "model"])
                .sort_index()
            ),
            failures=pd.DataFrame(failure_rows),
            orientation_history=(
                pd.concat(orientation_rows, ignore_index=True)
                .query("model in @successful")
                .set_index(["fold", "model", "feature"])
                .sort_index()
                if orientation_rows
                else pd.DataFrame()
            ),
            shapley_history=(
                pd.concat(shapley_rows, ignore_index=True)
                .query("model in @successful")
                .set_index(["fold", "model", "feature"])
                .sort_index()
                if shapley_rows
                else pd.DataFrame()
            ),
            shapley=shapley,
            interactions=interactions,
            fitted_models=final_models,
        )

    return FutureLossExperimentResult(
        portfolio=portfolio,
        features=features,
        horizons=results,
    )
