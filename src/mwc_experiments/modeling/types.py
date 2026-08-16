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


@dataclass(frozen=True, slots=True)
class OrientationDiagnostics:
    """Freeze orientation evidence learned exclusively from the training sample."""

    training_observations: int
    correlations: tuple[float, ...]
    subperiod_correlations: tuple[tuple[float, ...], ...]
    sign_agreement: tuple[float, ...]
    sign_stable: tuple[bool, ...]
    meets_threshold: tuple[bool, ...]
    orientation_supported: tuple[bool, ...]
    signs: tuple[float, ...]
    transformations: tuple[str, ...]


class CorrelationOrientationTransformer(TransformerMixin, BaseEstimator):
    """Orient features using correlation evidence from chronological training data.

    Weak correlations leave a feature unchanged. Sign stability is always diagnosed
    over contiguous chronological subperiods and can optionally be required before
    applying an orientation. During the final train-validation refit, diagnostics
    learned on training can be frozen so validation and test never determine signs.
    """

    def __init__(
        self,
        minimum_absolute_correlation: float = 0.2,
        stability_subperiods: int = 3,
        require_sign_stability: bool = False,
        frozen_diagnostics: OrientationDiagnostics | None = None,
    ) -> None:
        """Configure correlation strength, chronological stability and freezing."""
        self.minimum_absolute_correlation = minimum_absolute_correlation
        self.stability_subperiods = stability_subperiods
        self.require_sign_stability = require_sign_stability
        self.frozen_diagnostics = frozen_diagnostics

    @staticmethod
    def _correlations(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Calculate feature-wise Pearson correlations with safe constant handling."""
        centered_x = matrix - np.mean(matrix, axis=0)
        centered_y = target - np.mean(target)
        numerator = np.sum(centered_x * centered_y[:, None], axis=0)
        denominator = np.sqrt(
            np.sum(centered_x**2, axis=0) * np.sum(centered_y**2)
        )
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0.0,
        )

    def _load_frozen_diagnostics(self, n_features: int) -> None:
        """Restore training-only evidence without inspecting the refit target."""
        diagnostics = self.frozen_diagnostics
        if diagnostics is None:
            raise RuntimeError("No frozen orientation diagnostics are available.")
        if len(diagnostics.correlations) != n_features:
            raise ValueError("Frozen orientation feature count does not match X.")
        subperiod_correlations = np.asarray(
            diagnostics.subperiod_correlations,
            dtype=float,
        )
        if (
            subperiod_correlations.ndim != 2
            or subperiod_correlations.shape[1] != n_features
        ):
            raise ValueError("Frozen subperiod correlations do not match X.")
        self.training_observations_ = diagnostics.training_observations
        self.correlations_ = np.asarray(diagnostics.correlations, dtype=float)
        self.subperiod_correlations_ = subperiod_correlations
        self.sign_agreement_ = np.asarray(diagnostics.sign_agreement, dtype=float)
        self.sign_stable_ = np.asarray(diagnostics.sign_stable, dtype=bool)
        self.meets_threshold_ = np.asarray(diagnostics.meets_threshold, dtype=bool)
        self.orientation_supported_ = np.asarray(
            diagnostics.orientation_supported,
            dtype=bool,
        )
        self.signs_ = np.asarray(diagnostics.signs, dtype=float)
        self.transformations_ = np.asarray(
            diagnostics.transformations,
            dtype=object,
        )
        self.orientation_source_ = "training only (frozen before validation refit)"

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> "CorrelationOrientationTransformer":
        """Estimate feature orientation signs from the fitting sample only."""
        if isinstance(X, pd.DataFrame) and not X.index.is_monotonic_increasing:
            raise ValueError(
                "Chronological orientation requires X in increasing index order."
            )
        if not 0.0 <= self.minimum_absolute_correlation <= 1.0:
            raise ValueError("minimum_absolute_correlation must be in [0, 1].")
        if (
            isinstance(self.stability_subperiods, bool)
            or not isinstance(self.stability_subperiods, int)
            or self.stability_subperiods < 1
        ):
            raise ValueError("stability_subperiods must be a positive integer.")
        matrix = check_array(X, dtype=float, ensure_2d=True)
        target = check_array(y, dtype=float, ensure_2d=False).reshape(-1)
        if matrix.shape[0] != target.size:
            raise ValueError("X and y have incompatible lengths.")
        if matrix.shape[0] < 2 * self.stability_subperiods:
            raise ValueError(
                "Each orientation stability subperiod requires at least two rows."
            )
        self.n_features_in_ = matrix.shape[1]
        columns = getattr(X, "columns", None)
        if columns is not None:
            self.feature_names_in_ = np.asarray(columns, dtype=object)

        if self.frozen_diagnostics is not None:
            self._load_frozen_diagnostics(self.n_features_in_)
            return self

        correlations = self._correlations(matrix, target)
        chronological_indices = np.array_split(
            np.arange(matrix.shape[0]),
            self.stability_subperiods,
        )
        subperiod_correlations = np.vstack(
            [
                self._correlations(matrix[index], target[index])
                for index in chronological_indices
            ]
        )
        full_signs = np.sign(correlations)
        subperiod_signs = np.sign(subperiod_correlations)
        sign_agreement = np.mean(
            subperiod_signs == full_signs[None, :],
            axis=0,
        )
        sign_stable = np.all(
            subperiod_signs == full_signs[None, :],
            axis=0,
        )
        meets_threshold = (
            np.abs(correlations) >= self.minimum_absolute_correlation
        )
        orientation_supported = meets_threshold & (
            sign_stable if self.require_sign_stability else True
        )
        signs = np.where(
            orientation_supported & (correlations < 0.0),
            -1.0,
            1.0,
        )
        transformations = np.full(
            self.n_features_in_,
            "unchanged (positive training correlation)",
            dtype=object,
        )
        transformations[~meets_threshold] = (
            "unchanged (below absolute correlation threshold)"
        )
        transformations[
            meets_threshold & self.require_sign_stability & ~sign_stable
        ] = "unchanged (unstable chronological sign)"
        transformations[signs < 0.0] = "multiplied by -1"

        self.training_observations_ = matrix.shape[0]
        self.correlations_ = correlations
        self.subperiod_correlations_ = subperiod_correlations
        self.sign_agreement_ = sign_agreement
        self.sign_stable_ = sign_stable
        self.meets_threshold_ = meets_threshold
        self.orientation_supported_ = orientation_supported
        self.signs_ = signs
        self.transformations_ = transformations
        self.orientation_source_ = "fitting sample"
        return self

    def fitted_diagnostics(self) -> OrientationDiagnostics:
        """Return immutable evidence suitable for a leakage-free final refit."""
        check_is_fitted(self, ["signs_", "correlations_"])
        return OrientationDiagnostics(
            training_observations=int(self.training_observations_),
            correlations=tuple(float(value) for value in self.correlations_),
            subperiod_correlations=tuple(
                tuple(float(value) for value in row)
                for row in self.subperiod_correlations_
            ),
            sign_agreement=tuple(float(value) for value in self.sign_agreement_),
            sign_stable=tuple(bool(value) for value in self.sign_stable_),
            meets_threshold=tuple(bool(value) for value in self.meets_threshold_),
            orientation_supported=tuple(
                bool(value) for value in self.orientation_supported_
            ),
            signs=tuple(float(value) for value in self.signs_),
            transformations=tuple(str(value) for value in self.transformations_),
        )

    def transform(self, X: ArrayLike) -> np.ndarray | pd.DataFrame:
        """Apply the orientation signs estimated during training."""
        check_is_fitted(self, ["signs_"])
        matrix = check_array(X, dtype=float, ensure_2d=True)
        oriented = matrix * self.signs_
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(oriented, index=X.index, columns=X.columns)
        return oriented

    def orientation_table(self) -> pd.DataFrame:
        """Return correlations, chronological stability and final orientations."""
        check_is_fitted(self, ["signs_", "correlations_"])
        names = (
            self.feature_names_in_
            if hasattr(self, "feature_names_in_")
            else np.asarray([f"x{i}" for i in range(self.n_features_in_)])
        )
        result = pd.DataFrame(
            {
                "training_correlation": self.correlations_,
                "absolute_training_correlation": np.abs(self.correlations_),
                "minimum_absolute_correlation": (
                    self.minimum_absolute_correlation
                ),
                "meets_correlation_threshold": self.meets_threshold_,
                "subperiod_sign_agreement": self.sign_agreement_,
                "sign_stable": self.sign_stable_,
                "stability_required": self.require_sign_stability,
                "orientation_supported": self.orientation_supported_,
                "orientation_sign": self.signs_,
                "transformation": self.transformations_,
                "orientation_training_observations": self.training_observations_,
                "orientation_source": self.orientation_source_,
            },
            index=names,
        )
        for position, correlations in enumerate(
            self.subperiod_correlations_,
            start=1,
        ):
            result[f"training_subperiod_{position}_correlation"] = correlations
        return result

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


@dataclass(slots=True)
class WalkForwardFold:
    """One leakage-free rolling estimation and out-of-sample block."""

    fold: int
    window_start: pd.Timestamp
    validation_start: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    split: TemporalSplit
