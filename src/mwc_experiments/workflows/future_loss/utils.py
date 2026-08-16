"""Future Loss domain."""

from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any
import numpy as np
import pandas as pd
from mwc_experiments.settings import HORIZONS, MAIN_RISK_FEATURES, RANDOM_STATE
from mwc_experiments.evaluation.interpretation import capacity_summary
from mwc_experiments.evaluation.metrics import regression_metrics
from mwc_experiments.modeling.registries import (
    apply_parameter_grid_overrides,
    regression_candidates,
)
from mwc_experiments.modeling.selection import refit_selected, select_regression_model
from mwc_experiments.modeling.splits import chronological_split
from mwc_experiments.modeling.types import TemporalSplit
from mwc_experiments.workflows.common import model_parameter_count, quick_candidates
from .types import HorizonRegressionResult, FutureLossExperimentResult

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
    verbose: bool = True,
) -> FutureLossExperimentResult:
    results: dict[int, HorizonRegressionResult] = {}

    for horizon in horizons:
        target = f"future_loss_h{horizon}"
        split = chronological_split(
            dataset[list(features)],
            dataset[target].astype(float),
            horizon=horizon,
        )
        candidates = regression_candidates(
            len(features),
            random_state=random_state,
            include_mlp=True,
            include_dummy=True,
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
        predictions: dict[str, pd.Series] = {}
        fitted_models: dict[str, object] = {}
        shapley: dict[str, pd.Series] = {}
        interactions: dict[str, pd.DataFrame] = {}

        training_mean = pd.concat([split.y_train, split.y_validation]).mean()
        benchmark = np.repeat(training_mean, len(split.y_test))

        for name, candidate in candidates.items():
            if verbose:
                print(f"[future loss h={horizon}] {name}", flush=True)
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
                predictions[name] = prediction
                fitted_models[name] = fitted
                metric_rows.append(
                    {
                        "model": name,
                        **regression_metrics(
                            split.y_test,
                            prediction,
                            benchmark_prediction=benchmark,
                        ),
                        "validation RMSE": selected.validation_score,
                        "selection runtime seconds": selected.runtime_seconds,
                        "refit runtime seconds": refit_runtime,
                        "parameter count": model_parameter_count(fitted),
                    }
                )
                parameter_rows.append(
                    {"model": name, "best parameters": selected.best_params}
                )
                for message in selected.failures:
                    failure_rows.append({"model": name, "message": message})
                if name.startswith("Choquet"):
                    values, matrix = capacity_summary(fitted, list(features))
                    shapley[name] = values
                    interactions[name] = matrix
            except Exception as error:
                failure_rows.append(
                    {"model": name, "message": f"{type(error).__name__}: {error}"}
                )

        if not metric_rows:
            raise RuntimeError(f"No model completed successfully for horizon {horizon}.")
        results[horizon] = HorizonRegressionResult(
            horizon=horizon,
            split=split,
            metrics=pd.DataFrame(metric_rows).set_index("model").sort_values("RMSE"),
            predictions=pd.concat(predictions, axis=1).sort_index(),
            selected_parameters=pd.DataFrame(parameter_rows).set_index("model") if parameter_rows else pd.DataFrame(),
            failures=pd.DataFrame(failure_rows),
            shapley=shapley,
            interactions=interactions,
            fitted_models=fitted_models,
        )

    return FutureLossExperimentResult(
        portfolio=portfolio,
        features=features,
        horizons=results,
    )
