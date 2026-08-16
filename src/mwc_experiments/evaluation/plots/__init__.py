"""Public API for the plots domain."""

from .utils import (
    plot_metric_ranking,
    plot_actual_predictions,
    plot_matrix,
    plot_shapley,
    plot_classifier_diagnostics,
)

__all__ = [
    "plot_metric_ranking",
    "plot_actual_predictions",
    "plot_matrix",
    "plot_shapley",
    "plot_classifier_diagnostics",
]
