"""Preprocessing domain."""

from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from capacities_ml_fin.ml.preprocessing import CapacityNormalizer
from mwc_experiments.modeling.types import (
    CorrelationOrientationTransformer,
    QuantileClipper,
)
from mwc_experiments.settings import (
    ORIENTATION_MINIMUM_ABSOLUTE_CORRELATION,
    ORIENTATION_REQUIRE_SIGN_STABILITY,
    ORIENTATION_STABILITY_SUBPERIODS,
)

def make_capacity_preprocessor(
    *,
    clipping: bool = False,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
    minimum_absolute_correlation: float = (
        ORIENTATION_MINIMUM_ABSOLUTE_CORRELATION
    ),
    stability_subperiods: int = ORIENTATION_STABILITY_SUBPERIODS,
    require_sign_stability: bool = ORIENTATION_REQUIRE_SIGN_STABILITY,
) -> Pipeline:
    """Build optional clipping, orientation and capacity preprocessing."""
    return Pipeline(
        steps=[
            (
                "clip",
                QuantileClipper(
                    lower_quantile,
                    upper_quantile,
                    enabled=clipping,
                ),
            ),
            (
                "orient",
                CorrelationOrientationTransformer(
                    minimum_absolute_correlation=minimum_absolute_correlation,
                    stability_subperiods=stability_subperiods,
                    require_sign_stability=require_sign_stability,
                ),
            ),
            ("scale", CapacityNormalizer(feature_range=(0.0, 1.0), clip=True)),
        ]
    )


def make_standard_preprocessor(
    *,
    clipping: bool = False,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
) -> Pipeline:
    """Build optional clipping and standardization without orientation."""
    return Pipeline(
        steps=[
            (
                "clip",
                QuantileClipper(
                    lower_quantile,
                    upper_quantile,
                    enabled=clipping,
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )


def make_oriented_standard_preprocessor(
    *,
    clipping: bool = False,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
    minimum_absolute_correlation: float = (
        ORIENTATION_MINIMUM_ABSOLUTE_CORRELATION
    ),
    stability_subperiods: int = ORIENTATION_STABILITY_SUBPERIODS,
    require_sign_stability: bool = ORIENTATION_REQUIRE_SIGN_STABILITY,
) -> Pipeline:
    """Build the classical-model orientation ablation preprocessor."""
    return Pipeline(
        steps=[
            (
                "clip",
                QuantileClipper(
                    lower_quantile,
                    upper_quantile,
                    enabled=clipping,
                ),
            ),
            (
                "orient",
                CorrelationOrientationTransformer(
                    minimum_absolute_correlation=minimum_absolute_correlation,
                    stability_subperiods=stability_subperiods,
                    require_sign_stability=require_sign_stability,
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )
