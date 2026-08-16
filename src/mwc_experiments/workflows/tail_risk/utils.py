"""Tail Risk domain."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from mwc_experiments.evaluation.interpretation import (
    capacity_summary,
    orientation_table,
)
from mwc_experiments.evaluation.metrics import (
    classification_metrics,
    optimal_f1_threshold,
)
from mwc_experiments.modeling.registries import (
    apply_parameter_grid_overrides,
    classification_candidates,
)
from mwc_experiments.modeling.selection import (
    refit_selected,
    select_classification_model,
)
from mwc_experiments.modeling.splits import (
    aggregate_walk_forward_split,
    rolling_walk_forward_splits,
    walk_forward_fold_summary,
)
from mwc_experiments.settings import (
    MAIN_RISK_FEATURES,
    PRIMARY_TAIL_ALPHA,
    RANDOM_STATE,
)
from mwc_experiments.workflows.common import (
    model_parameter_count,
    quick_candidates,
)

from .types import TailClassificationResult


def _partition_selection_and_calibration(
    split,
    *,
    calibration_start: pd.Timestamp,
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a fold's past-only holdout into selection and calibration."""
    calibration_mask = split.X_validation.index >= calibration_start
    X_calibration = split.X_validation.loc[calibration_mask]
    y_calibration = split.y_validation.loc[calibration_mask]
    X_selection = split.X_validation.loc[~calibration_mask]
    y_selection = split.y_validation.loc[~calibration_mask]
    if horizon > 0:
        if len(X_selection) <= horizon:
            raise ValueError(
                "Not enough selection-validation observations for purging."
            )
        X_selection = X_selection.iloc[:-horizon]
        y_selection = y_selection.iloc[:-horizon]
    if min(len(X_selection), len(X_calibration)) == 0:
        raise ValueError("Selection and calibration must both be non-empty.")
    if y_selection.nunique() < 2:
        raise ValueError("Selection validation must contain both event classes.")
    if y_calibration.nunique() < 2:
        raise ValueError("Calibration must contain both event classes.")
    split.X_validation = X_selection
    split.y_validation = y_selection
    return X_calibration, y_calibration


def _fold_sample_rows(
    fold_number: int,
    split,
    X_calibration: pd.DataFrame,
    y_calibration: pd.Series,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, X, y in (
        ("train", split.X_train, split.y_train),
        ("selection_validation", split.X_validation, split.y_validation),
        ("calibration", X_calibration, y_calibration),
        ("OOS", split.X_test, split.y_test),
    ):
        rows.append(
            {
                "fold": fold_number,
                "sample": name,
                "observations": len(X),
                "start": X.index.min(),
                "end": X.index.max(),
                "events": int(y.sum()),
                "event prevalence": float(y.mean()),
            }
        )
    return rows


def run_tail_classification_experiment(
    dataset: pd.DataFrame,
    *,
    features: tuple[str, ...] = MAIN_RISK_FEATURES,
    horizons: tuple[int, ...] = (1, 5, 10),
    alpha: float = PRIMARY_TAIL_ALPHA,
    quick: bool = True,
    model_names: tuple[str, ...] | None = None,
    parameter_grids: Mapping[str, Mapping[str, list[Any]]] | None = None,
    random_state: int = RANDOM_STATE,
    oos_start: str = "2020-01-01",
    training_window_years: int = 5,
    selection_window_months: int = 18,
    calibration_window_months: int = 24,
    oos_block_years: int = 1,
    calibration_methods: tuple[str, ...] = ("sigmoid",),
    class_weight_modes: tuple[str, ...] = ("balanced", "unweighted"),
    verbose: bool = True,
) -> dict[int, TailClassificationResult]:
    """Run rolling selection, calibration and OOS tail classification."""
    unsupported = set(calibration_methods) - {"sigmoid"}
    if unsupported:
        raise ValueError(
            "calibration_methods only supports 'sigmoid'; got "
            f"{sorted(unsupported)}."
        )
    if min(selection_window_months, calibration_window_months) < 1:
        raise ValueError("Selection and calibration windows must be positive.")
    supported_weight_modes = {"balanced", "unweighted"}
    unsupported_weight_modes = set(class_weight_modes) - supported_weight_modes
    if unsupported_weight_modes or not class_weight_modes:
        raise ValueError(
            "class_weight_modes must contain 'balanced', 'unweighted', or both; "
            f"got {sorted(set(class_weight_modes))}."
        )
    if len(set(class_weight_modes)) != len(class_weight_modes):
        raise ValueError("class_weight_modes must not contain duplicates.")

    results: dict[int, TailClassificationResult] = {}
    alpha_label = str(alpha).replace(".", "p")
    total_holdout_months = selection_window_months + calibration_window_months

    for horizon in horizons:
        target = f"tail_event_h{horizon}_a{alpha_label}"
        if target not in dataset:
            raise KeyError(
                f"Missing {target}. Build the dataset with alpha={alpha} first."
            )
        folds = list(
            rolling_walk_forward_splits(
                dataset[list(features)],
                dataset[target].astype(float),
                oos_start=oos_start,
                training_window_years=training_window_years,
                validation_window_months=total_holdout_months,
                oos_block_years=oos_block_years,
                horizon=horizon,
            )
        )
        candidates = {}
        candidate_family: dict[str, str] = {}
        candidate_weight: dict[str, str] = {}
        weight_independent_models = {"Rolling prior probability", "MLP"}
        for weight_mode in class_weight_modes:
            mode_candidates = classification_candidates(
                len(features),
                random_state=random_state,
                include_mlp=True,
                class_weight=(
                    "balanced" if weight_mode == "balanced" else None
                ),
            )
            mode_candidates = apply_parameter_grid_overrides(
                mode_candidates,
                parameter_grids,
                n_features=len(features),
            )
            if quick:
                mode_candidates = quick_candidates(mode_candidates)
            if model_names is not None:
                mode_candidates = {
                    name: mode_candidates[name] for name in model_names
                }
            for family, candidate in mode_candidates.items():
                weight_independent = family in weight_independent_models
                name = (
                    family
                    if weight_independent or len(class_weight_modes) == 1
                    else f"{family} [{weight_mode}]"
                )
                if name in candidates:
                    continue
                candidates[name] = candidate
                candidate_family[name] = family
                candidate_weight[name] = (
                    "not applicable" if weight_independent else weight_mode
                )

        probability_chunks: dict[str, list[pd.Series]] = {}
        threshold_chunks: dict[str, list[pd.Series]] = {}
        variant_base: dict[str, str] = {}
        variant_method: dict[str, str] = {}
        for name in candidates:
            for variant, method in (
                (name, "uncalibrated"),
                *((f"{name} [{method}]", method) for method in calibration_methods),
            ):
                probability_chunks[variant] = []
                threshold_chunks[variant] = []
                variant_base[variant] = name
                variant_method[variant] = method

        fold_metric_rows: list[dict[str, object]] = []
        parameter_rows: list[dict[str, object]] = []
        failure_rows: list[dict[str, object]] = []
        sample_rows: list[dict[str, object]] = []
        orientation_rows: list[pd.DataFrame] = []
        shapley_rows: list[pd.DataFrame] = []
        failed_base_models: set[str] = set()
        failed_variants: set[str] = set()
        fitted_models: dict[str, object] = {}
        calibrated_models: dict[str, object] = {}
        shapley: dict[str, pd.Series] = {}
        interactions: dict[str, pd.DataFrame] = {}

        for fold in folds:
            split = fold.split
            split.y_train = split.y_train.astype(int)
            split.y_validation = split.y_validation.astype(int)
            split.y_test = split.y_test.astype(int)
            calibration_start = fold.oos_start - pd.DateOffset(
                months=calibration_window_months
            )
            X_calibration, y_calibration = _partition_selection_and_calibration(
                split,
                calibration_start=calibration_start,
                horizon=horizon,
            )
            sample_rows.extend(
                _fold_sample_rows(
                    fold.fold,
                    split,
                    X_calibration,
                    y_calibration,
                )
            )

            for name, candidate in candidates.items():
                if name in failed_base_models:
                    continue
                if verbose:
                    print(
                        f"[tail alpha={alpha:g}, h={horizon}, "
                        f"fold={fold.fold}] {name}",
                        flush=True,
                    )
                try:
                    selected = select_classification_model(
                        name,
                        candidate,
                        split.X_train,
                        split.y_train,
                        split.X_validation,
                        split.y_validation,
                    )
                    base_threshold = optimal_f1_threshold(
                        split.y_validation,
                        selected.validation_predictions,
                    )
                    fitted, refit_runtime = refit_selected(
                        selected,
                        split.X_train,
                        split.y_train,
                        split.X_validation,
                        split.y_validation,
                    )
                    base_probability = pd.Series(
                        fitted.predict_proba(split.X_test)[:, 1],
                        index=split.X_test.index,
                        name=name,
                    )
                    probability_chunks[name].append(base_probability)
                    threshold_chunks[name].append(
                        pd.Series(base_threshold, index=split.X_test.index)
                    )
                    fitted_models[name] = fitted
                    common = {
                        "fold": fold.fold,
                        "base model": candidate_family[name],
                        "class weight": candidate_weight[name],
                        "validation PR AUC": selected.validation_score,
                        "selection runtime seconds": selected.runtime_seconds,
                        "refit runtime seconds": refit_runtime,
                        "parameter count": model_parameter_count(fitted),
                        "calibration observations": len(X_calibration),
                        "calibration event prevalence": float(
                            y_calibration.mean()
                        ),
                    }
                    fold_metric_rows.append(
                        {
                            "model": name,
                            "probability calibration": "uncalibrated",
                            "threshold source": "selection validation",
                            "calibration runtime seconds": 0.0,
                            **classification_metrics(
                                split.y_test,
                                base_probability,
                                threshold=base_threshold,
                            ),
                            **common,
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

                    for method in calibration_methods:
                        variant = f"{name} [{method}]"
                        if variant in failed_variants:
                            continue
                        try:
                            started = perf_counter()
                            calibrated = CalibratedClassifierCV(
                                estimator=FrozenEstimator(fitted),
                                method=method,
                            )
                            calibrated.fit(X_calibration, y_calibration)
                            calibration_runtime = perf_counter() - started
                            calibration_probability = calibrated.predict_proba(
                                X_calibration
                            )[:, 1]
                            threshold = optimal_f1_threshold(
                                y_calibration,
                                calibration_probability,
                            )
                            probability = pd.Series(
                                calibrated.predict_proba(split.X_test)[:, 1],
                                index=split.X_test.index,
                                name=variant,
                            )
                            probability_chunks[variant].append(probability)
                            threshold_chunks[variant].append(
                                pd.Series(threshold, index=split.X_test.index)
                            )
                            calibrated_models[variant] = calibrated
                            fold_metric_rows.append(
                                {
                                    "model": variant,
                                    "probability calibration": method,
                                    "threshold source": "calibration",
                                    "calibration runtime seconds": (
                                        calibration_runtime
                                    ),
                                    **classification_metrics(
                                        split.y_test,
                                        probability,
                                        threshold=threshold,
                                    ),
                                    **common,
                                }
                            )
                        except Exception as error:
                            failed_variants.add(variant)
                            probability_chunks[variant].clear()
                            threshold_chunks[variant].clear()
                            calibrated_models.pop(variant, None)
                            failure_rows.append(
                                {
                                    "fold": fold.fold,
                                    "model": variant,
                                    "message": f"{type(error).__name__}: {error}",
                                }
                            )

                    if name.startswith("Choquistic"):
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
                    failed_base_models.add(name)
                    for variant in [
                        name,
                        *(f"{name} [{method}]" for method in calibration_methods),
                    ]:
                        probability_chunks[variant].clear()
                        threshold_chunks[variant].clear()
                        calibrated_models.pop(variant, None)
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
            for name, chunks in probability_chunks.items()
            if len(chunks) == len(folds)
        }
        if not successful:
            raise RuntimeError(
                f"No classifier completed every fold for horizon {horizon}."
            )
        probabilities = pd.concat(successful, axis=1).sort_index()
        thresholds = pd.concat(
            {
                name: pd.concat(threshold_chunks[name]).sort_index()
                for name in successful
            },
            axis=1,
        ).sort_index()
        split = aggregate_walk_forward_split(folds)
        fold_metrics = pd.DataFrame(fold_metric_rows)
        fold_metrics = fold_metrics[
            fold_metrics["model"].isin(successful)
        ].set_index(["fold", "model"]).sort_index()
        metric_rows: list[dict[str, object]] = []
        for name, probability in successful.items():
            model_folds = fold_metrics.xs(name, level="model")
            base = variant_base[name]
            metric_rows.append(
                {
                    "model": name,
                    "base model": candidate_family[base],
                    "class weight": candidate_weight[base],
                    "probability calibration": variant_method[name],
                    "threshold source": (
                        "selection validation"
                        if variant_method[name] == "uncalibrated"
                        else "calibration"
                    ),
                    **classification_metrics(
                        split.y_test.loc[probability.index],
                        probability,
                        threshold=thresholds[name],
                    ),
                    "validation PR AUC": float(
                        model_folds["validation PR AUC"].mean()
                    ),
                    "selection runtime seconds": float(
                        model_folds["selection runtime seconds"].sum()
                    ),
                    "refit runtime seconds": float(
                        model_folds["refit runtime seconds"].sum()
                    ),
                    "calibration runtime seconds": float(
                        model_folds["calibration runtime seconds"].sum()
                    ),
                    "parameter count": model_parameter_count(
                        fitted_models[base]
                    ),
                    "calibration observations": int(
                        model_folds["calibration observations"].sum()
                    ),
                    "calibration event prevalence": float(
                        model_folds["calibration event prevalence"].mean()
                    ),
                    "OOS folds": len(folds),
                }
            )
        metrics = pd.DataFrame(metric_rows).set_index("model").sort_values(
            "PR AUC", ascending=False
        )
        discrimination_columns = [
            "base model",
            "class weight",
            "probability calibration",
            "ROC AUC",
            "PR AUC",
            "validation PR AUC",
        ]
        calibration_columns = [
            "base model",
            "class weight",
            "probability calibration",
            "Brier",
            "Log loss",
            "Mean predicted probability",
            "Observed event prevalence",
            "Calibration gap",
            "Absolute calibration gap",
        ]
        successful_bases = {
            variant_base[name] for name in successful
        }
        results[horizon] = TailClassificationResult(
            horizon=horizon,
            alpha=alpha,
            split=split,
            final_split=folds[-1].split,
            fold_summary=walk_forward_fold_summary(folds),
            fold_metrics=fold_metrics,
            metrics=metrics,
            discrimination_metrics=metrics[discrimination_columns].copy(),
            calibration_metrics=metrics[calibration_columns].copy(),
            calibration_sample_summary=(
                pd.DataFrame(sample_rows)
                .set_index(["fold", "sample"])
                .sort_index()
            ),
            probabilities=probabilities,
            thresholds=thresholds,
            selected_parameters=(
                pd.DataFrame(parameter_rows)
                .query("model in @successful_bases")
                .set_index(["fold", "model"])
                .sort_index()
            ),
            failures=pd.DataFrame(failure_rows),
            orientation_history=(
                pd.concat(orientation_rows, ignore_index=True)
                .query("model in @successful_bases")
                .set_index(["fold", "model", "feature"])
                .sort_index()
                if orientation_rows
                else pd.DataFrame()
            ),
            shapley_history=(
                pd.concat(shapley_rows, ignore_index=True)
                .query("model in @successful_bases")
                .set_index(["fold", "model", "feature"])
                .sort_index()
                if shapley_rows
                else pd.DataFrame()
            ),
            shapley=shapley,
            interactions=interactions,
            fitted_models={
                name: model
                for name, model in fitted_models.items()
                if name in successful_bases
            },
            calibrated_models={
                name: model
                for name, model in calibrated_models.items()
                if name in successful
            },
        )

    return results
