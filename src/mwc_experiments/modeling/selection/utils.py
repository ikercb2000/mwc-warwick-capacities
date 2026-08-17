"""Selection domain."""

from __future__ import annotations
from time import perf_counter
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import average_precision_score, mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from mwc_experiments.modeling.types import (
    Candidate,
    CorrelationOrientationTransformer,
    SelectedModel,
)


def classification_score(
    estimator: BaseEstimator,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return a positive-class probability or an uncalibrated Choquet score."""
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(X)[:, 1], dtype=float)
    if isinstance(estimator, Pipeline):
        classifier = estimator.named_steps.get("classifier")
        if classifier is not None and hasattr(classifier, "choquet_score"):
            transformed = estimator[:-1].transform(X)
            return np.asarray(classifier.choquet_score(transformed), dtype=float)
    raise TypeError(
        "Classifier must expose predict_proba or a Choquet choquet_score."
    )


def _frozen_orientation_parameters(
    estimator: BaseEstimator,
) -> dict[str, object]:
    """Extract fitted training orientations as refit-safe estimator parameters."""
    frozen: dict[str, object] = {}
    for parameter_name, component in estimator.get_params(deep=True).items():
        if isinstance(component, CorrelationOrientationTransformer):
            frozen[f"{parameter_name}__frozen_diagnostics"] = (
                component.fitted_diagnostics()
            )
    return frozen


def training_orientation_parameters(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, object]:
    """Fit only preprocessing to obtain orientation parameters from training."""
    if not isinstance(estimator, Pipeline):
        return {}
    preprocessor = estimator.named_steps.get("preprocessor")
    if not isinstance(preprocessor, Pipeline) or not any(
        isinstance(component, CorrelationOrientationTransformer)
        for component in preprocessor.get_params(deep=True).values()
    ):
        return {}
    fitted_preprocessor = clone(preprocessor).fit(X_train, y_train)
    frozen: dict[str, object] = {}
    for parameter_name, component in fitted_preprocessor.get_params(
        deep=True
    ).items():
        if isinstance(component, CorrelationOrientationTransformer):
            frozen[
                f"preprocessor__{parameter_name}__frozen_diagnostics"
            ] = component.fitted_diagnostics()
    return frozen

def _select_model(
    name: str,
    candidate: Candidate,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    *,
    predict_validation: Callable[[BaseEstimator, pd.DataFrame], np.ndarray],
    score_validation: Callable[[pd.Series, np.ndarray], float],
    maximize: bool,
) -> SelectedModel:
    """Select one candidate configuration using a task-specific prediction score."""
    best_model: BaseEstimator | None = None
    best_params: dict[str, object] = {}
    best_score = -np.inf if maximize else np.inf
    best_predictions: np.ndarray | None = None
    failures: list[str] = []
    started = perf_counter()

    grid = list(ParameterGrid(candidate.param_grid)) if candidate.param_grid else [{}]
    for params in grid:
        estimator = clone(candidate.estimator).set_params(**params)
        try:
            estimator.fit(X_train, y_train)
            predictions = np.asarray(
                predict_validation(estimator, X_validation),
                dtype=float,
            ).reshape(-1)
            score = float(score_validation(y_validation, predictions))
        except Exception as error:
            failures.append(f"params={params}: {type(error).__name__}: {error}")
            continue

        improves = score > best_score if maximize else score < best_score
        if improves:
            best_model = estimator
            best_params = dict(params)
            best_score = score
            best_predictions = predictions

    if best_model is None or best_predictions is None:
        raise RuntimeError(f"Every configuration failed for {name}: {failures}")
    return SelectedModel(
        name=name,
        estimator=best_model,
        best_params=best_params,
        validation_score=best_score,
        validation_predictions=best_predictions,
        runtime_seconds=perf_counter() - started,
        failures=failures,
    )


def select_regression_model(
    name: str,
    candidate: Candidate,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> SelectedModel:
    """Select a regression configuration by minimum validation RMSE."""
    return _select_model(
        name,
        candidate,
        X_train,
        y_train,
        X_validation,
        y_validation,
        predict_validation=lambda estimator, X: estimator.predict(X),
        score_validation=lambda y, prediction: float(
            np.sqrt(mean_squared_error(y, prediction))
        ),
        maximize=False,
    )


def select_classification_model(
    name: str,
    candidate: Candidate,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> SelectedModel:
    """Select a classifier configuration by maximum validation PR AUC."""
    return _select_model(
        name,
        candidate,
        X_train,
        y_train,
        X_validation,
        y_validation,
        predict_validation=classification_score,
        score_validation=lambda y, probability: float(
            average_precision_score(y, probability)
        ),
        maximize=True,
    )


def refit_selected(
    selected: SelectedModel,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[BaseEstimator, float]:
    """Refit on train-validation while freezing training-only orientations."""
    estimator = clone(selected.estimator)
    frozen_orientations = _frozen_orientation_parameters(selected.estimator)
    if frozen_orientations:
        estimator.set_params(**frozen_orientations)
    X = pd.concat([X_train, X_validation])
    y = pd.concat([y_train, y_validation])
    started = perf_counter()
    estimator.fit(X, y)
    return estimator, perf_counter() - started
