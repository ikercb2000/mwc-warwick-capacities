"""Public API for the autoregression domain."""

from .types import (
    ARComparisonResult,
)

from .utils import (
    _fit_scale,
    _scale,
    _inverse,
    _sequential_choquet_predictions,
    _autoregressive_prediction,
    _sequential_linear_ar_predictions,
    _metrics,
    compare_choquet_autoregression,
)

__all__ = [
    "ARComparisonResult",
    "compare_choquet_autoregression",
]
