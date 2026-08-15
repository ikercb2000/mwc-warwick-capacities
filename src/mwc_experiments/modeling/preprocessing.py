"""Factory functions for training-only feature preprocessing pipelines."""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from capacities_ml_fin.ml.preprocessing import CapacityNormalizer

from mwc_experiments.modeling.types import (
    CorrelationOrientationTransformer,
    QuantileClipper,
)

def make_capacity_preprocessor(
    *,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
) -> Pipeline:
    """Build clipping, orientation and unit-interval capacity preprocessing."""
    return Pipeline(
        steps=[
            ("clip", QuantileClipper(lower_quantile, upper_quantile)),
            ("orient", CorrelationOrientationTransformer()),
            ("scale", CapacityNormalizer(feature_range=(0.0, 1.0), clip=True)),
        ]
    )


def make_standard_preprocessor(
    *,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
) -> Pipeline:
    """Build clipping and standardization without target-driven orientation."""
    return Pipeline(
        steps=[
            ("clip", QuantileClipper(lower_quantile, upper_quantile)),
            ("scale", StandardScaler()),
        ]
    )


def make_oriented_standard_preprocessor(
    *,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
) -> Pipeline:
    """Build the classical-model orientation ablation preprocessor."""
    return Pipeline(
        steps=[
            ("clip", QuantileClipper(lower_quantile, upper_quantile)),
            ("orient", CorrelationOrientationTransformer()),
            ("scale", StandardScaler()),
        ]
    )
