"""Public API for the tail risk domain."""

from .types import (
    TailClassificationResult,
)

from .utils import (
    run_tail_classification_experiment,
)

__all__ = [
    "TailClassificationResult",
    "run_tail_classification_experiment",
]
