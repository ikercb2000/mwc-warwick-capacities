"""Public API for the factor models domain."""

from .types import (
    FactorExperimentResult,
)

from .utils import (
    run_factor_experiment,
)

__all__ = [
    "FactorExperimentResult",
    "run_factor_experiment",
]
