"""Shared classes for preprocessing, registration, selection and splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class QuantileClipper(TransformerMixin, BaseEstimator):
    """Clip each feature using quantiles estimated only on the training sample."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995) -> None:
        """Configure the lower and upper clipping quantiles."""
        self.lower = lower
        self.upper = upper

    def fit(self, X: ArrayLike, y: Any = None) -> "QuantileClipper":
        """Estimate feature-wise clipping bounds from the fitting sample."""
        if not 0.0 <= self.lower < self.upper <= 1.0:
            raise ValueError("Require 0 <= lower < upper <= 1.")
        matrix = check_array(X, dtype=float, ensure_2d=True)
        self.n_features_in_ = matrix.shape[1]
        columns = getattr(X, "columns", None)
        if columns is not None:
            self.feature_names_in_ = np.asarray(columns, dtype=object)
        self.lower_bounds_ = np.nanquantile(matrix, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(matrix, self.upper, axis=0)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray | pd.DataFrame:
        """Clip observations to the fitted feature-wise bounds."""
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_"])
        matrix = check_array(X, dtype=float, ensure_2d=True)
        clipped = np.clip(matrix, self.lower_bounds_, self.upper_bounds_)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(clipped, index=X.index, columns=X.columns)
        return clipped

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return unchanged feature names for sklearn pipeline introspection."""
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_.copy()
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


class CorrelationOrientationTransformer(TransformerMixin, BaseEstimator):
    """Orient features so larger transformed values are associated with larger y.

    Signs are estimated exclusively from the fitting sample. This supplies the common
    monotone direction required by a capacity while preserving feature magnitudes.
    Constant or numerically uncorrelated variables retain their original direction.
    """

    def __init__(self, minimum_absolute_correlation: float = 1e-12) -> None:
        """Configure the correlation magnitude treated as numerically zero."""
        self.minimum_absolute_correlation = minimum_absolute_correlation

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> "CorrelationOrientationTransformer":
        """Estimate feature orientation signs from training correlations with y."""
        matrix = check_array(X, dtype=float, ensure_2d=True)
        target = np.asarray(y, dtype=float).reshape(-1)
        if matrix.shape[0] != target.size:
            raise ValueError("X and y have incompatible lengths.")
        self.n_features_in_ = matrix.shape[1]
        columns = getattr(X, "columns", None)
        if columns is not None:
            self.feature_names_in_ = np.asarray(columns, dtype=object)

        centered_x = matrix - np.nanmean(matrix, axis=0)
        centered_y = target - np.nanmean(target)
        numerator = np.nansum(centered_x * centered_y[:, None], axis=0)
        denominator = np.sqrt(
            np.nansum(centered_x**2, axis=0) * np.nansum(centered_y**2)
        )
        correlations = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0.0,
        )
        signs = np.where(
            np.abs(correlations) <= self.minimum_absolute_correlation,
            1.0,
            np.sign(correlations),
        )
        self.correlations_ = correlations
        self.signs_ = signs
        return self

    def transform(self, X: ArrayLike) -> np.ndarray | pd.DataFrame:
        """Apply the orientation signs estimated during fitting."""
        check_is_fitted(self, ["signs_"])
        matrix = check_array(X, dtype=float, ensure_2d=True)
        oriented = matrix * self.signs_
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(oriented, index=X.index, columns=X.columns)
        return oriented

    def orientation_table(self) -> pd.DataFrame:
        """Return fitted correlations, signs and readable transformations."""
        check_is_fitted(self, ["signs_", "correlations_"])
        names = (
            self.feature_names_in_
            if hasattr(self, "feature_names_in_")
            else np.asarray([f"x{i}" for i in range(self.n_features_in_)])
        )
        return pd.DataFrame(
            {
                "training_correlation": self.correlations_,
                "orientation_sign": self.signs_,
                "transformation": np.where(
                    self.signs_ > 0,
                    "unchanged",
                    "multiplied by -1",
                ),
            },
            index=names,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return unchanged feature names for sklearn pipeline introspection."""
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_.copy()
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


@dataclass(frozen=True, slots=True)
class Candidate:
    """Pair an unfitted sklearn estimator with its validation parameter grid."""

    estimator: BaseEstimator
    param_grid: dict[str, list[Any]]


@dataclass(slots=True)
class SelectedModel:
    """Record the best fitted validation model and its selection diagnostics."""

    name: str
    estimator: BaseEstimator
    best_params: dict[str, object]
    validation_score: float
    validation_predictions: np.ndarray
    runtime_seconds: float
    failures: list[str]


@dataclass(slots=True)
class TemporalSplit:
    """Hold chronological train, validation and test partitions."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    @property
    def summary(self) -> pd.DataFrame:
        """Summarise the size and date coverage of each temporal partition."""
        rows = []
        for name, X in (
            ("train", self.X_train),
            ("validation", self.X_validation),
            ("test", self.X_test),
        ):
            rows.append(
                {
                    "sample": name,
                    "observations": len(X),
                    "start": X.index.min(),
                    "end": X.index.max(),
                }
            )
        return pd.DataFrame(rows).set_index("sample")