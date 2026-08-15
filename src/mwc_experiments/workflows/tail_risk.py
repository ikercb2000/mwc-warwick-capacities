from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from mwc_experiments.settings import (
    CLASSIFICATION_TRAIN_END,
    MAIN_RISK_FEATURES,
    PRIMARY_TAIL_ALPHA,
    RANDOM_STATE,
    VALIDATION_END,
)
from mwc_experiments.evaluation.interpretation import capacity_summary
from mwc_experiments.evaluation.metrics import (
    classification_metrics,
    optimal_f1_threshold,
)
from mwc_experiments.modeling.registries import classification_candidates
from mwc_experiments.modeling.selection import (
    refit_selected,
    select_classification_model,
)
from mwc_experiments.modeling.splits import chronological_split
from mwc_experiments.modeling.types import TemporalSplit
from mwc_experiments.workflows.common import model_parameter_count, quick_candidates


@dataclass(slots=True)
class TailClassificationResult:
    horizon: int
    alpha: float
    split: TemporalSplit
    metrics: pd.DataFrame
    probabilities: pd.DataFrame
    thresholds: pd.Series
    selected_parameters: pd.DataFrame
    failures: pd.DataFrame
    shapley: dict[str, pd.Series] = field(default_factory=dict)
    interactions: dict[str, pd.DataFrame] = field(default_factory=dict)
    fitted_models: dict[str, object] = field(default_factory=dict)


def run_tail_classification_experiment(
    dataset: pd.DataFrame,
    *,
    features: tuple[str, ...] = MAIN_RISK_FEATURES,
    horizons: tuple[int, ...] = (1, 5, 10),
    alpha: float = PRIMARY_TAIL_ALPHA,
    quick: bool = True,
    model_names: tuple[str, ...] | None = None,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> dict[int, TailClassificationResult]:
    results: dict[int, TailClassificationResult] = {}
    alpha_label = str(alpha).replace(".", "p")

    for horizon in horizons:
        target = f"tail_event_h{horizon}_a{alpha_label}"
        if target not in dataset:
            raise KeyError(
                f"Missing {target}. Build the dataset with alpha={alpha} first."
            )
        split = chronological_split(
            dataset[list(features)],
            dataset[target].astype(float),
            train_end=CLASSIFICATION_TRAIN_END,
            validation_end=VALIDATION_END,
            horizon=horizon,
        )
        split.y_train = split.y_train.astype(int)
        split.y_validation = split.y_validation.astype(int)
        split.y_test = split.y_test.astype(int)

        candidates = classification_candidates(
            len(features),
            random_state=random_state,
            include_mlp=True,
        )
        if quick:
            candidates = quick_candidates(candidates)
        if model_names is not None:
            candidates = {name: candidates[name] for name in model_names}

        metric_rows: list[dict[str, object]] = []
        parameter_rows: list[dict[str, object]] = []
        failure_rows: list[dict[str, object]] = []
        probabilities: dict[str, pd.Series] = {}
        thresholds: dict[str, float] = {}
        fitted_models: dict[str, object] = {}
        shapley: dict[str, pd.Series] = {}
        interactions: dict[str, pd.DataFrame] = {}

        for name, candidate in candidates.items():
            if verbose:
                print(f"[tail alpha={alpha:g}, h={horizon}] {name}", flush=True)
            try:
                selected = select_classification_model(
                    name,
                    candidate,
                    split.X_train,
                    split.y_train,
                    split.X_validation,
                    split.y_validation,
                )
                threshold = optimal_f1_threshold(
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
                probability = pd.Series(
                    fitted.predict_proba(split.X_test)[:, 1],
                    index=split.X_test.index,
                    name=name,
                )
                probabilities[name] = probability
                thresholds[name] = threshold
                fitted_models[name] = fitted
                metric_rows.append(
                    {
                        "model": name,
                        **classification_metrics(
                            split.y_test,
                            probability,
                            threshold=threshold,
                        ),
                        "validation PR AUC": selected.validation_score,
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
                if name.startswith("Choquistic"):
                    values, matrix = capacity_summary(fitted, list(features))
                    shapley[name] = values
                    interactions[name] = matrix
            except Exception as error:
                failure_rows.append(
                    {"model": name, "message": f"{type(error).__name__}: {error}"}
                )

        if not metric_rows:
            raise RuntimeError(
                f"No classifier completed successfully for horizon {horizon}."
            )
        results[horizon] = TailClassificationResult(
            horizon=horizon,
            alpha=alpha,
            split=split,
            metrics=pd.DataFrame(metric_rows).set_index("model").sort_values("PR AUC", ascending=False),
            probabilities=pd.concat(probabilities, axis=1).sort_index(),
            thresholds=pd.Series(thresholds, name="validation-selected threshold"),
            selected_parameters=pd.DataFrame(parameter_rows).set_index("model") if parameter_rows else pd.DataFrame(),
            failures=pd.DataFrame(failure_rows),
            shapley=shapley,
            interactions=interactions,
            fitted_models=fitted_models,
        )

    return results
