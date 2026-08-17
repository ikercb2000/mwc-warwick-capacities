"""Inspection domain."""

from __future__ import annotations
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

def unwrap_fitted_estimator(estimator: BaseEstimator) -> BaseEstimator:
    """Return the concrete final estimator inside fitted sklearn containers."""
    current = estimator
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, Pipeline):
            check_is_fitted(current)
            current = current.steps[-1][1]
            continue
        if isinstance(current, TransformedTargetRegressor):
            check_is_fitted(current, ["regressor_"])
            current = current.regressor_
            continue
        return current
    raise RuntimeError("Estimator containers form an unexpected cycle.")


def fitted_q(estimator: BaseEstimator) -> float | None:
    """Return a fitted scaled-Choquet q coefficient when one is available."""
    inner = unwrap_fitted_estimator(estimator)
    value = getattr(inner, "q_", None)
    return None if value is None else float(value)
