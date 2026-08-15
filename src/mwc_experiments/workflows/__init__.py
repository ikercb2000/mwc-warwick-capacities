"""Expose the public experiment workflow API."""

from mwc_experiments.workflows.autoregression import (
    ARComparisonResult,
    compare_choquet_autoregression,
)
from mwc_experiments.workflows.common import select_model_family_by_validation
from mwc_experiments.workflows.factor_models import (
    FactorExperimentResult,
    run_factor_experiment,
)
from mwc_experiments.workflows.future_loss import (
    FutureLossExperimentResult,
    HorizonRegressionResult,
    run_future_loss_experiment,
)
from mwc_experiments.workflows.risk_simulation import (
    capital_backtest_table,
    coverage_test_table,
    diversification_table,
    ordered_risk_contributions,
    rolling_capital_panel,
    simulate_regime_switching_losses,
    static_risk_table,
)
from mwc_experiments.workflows.stability import (
    CapacityStabilityResult,
    expanding_capacity_stability,
)
from mwc_experiments.workflows.tail_risk import (
    TailClassificationResult,
    run_tail_classification_experiment,
)

__all__ = [
    "ARComparisonResult",
    "CapacityStabilityResult",
    "FactorExperimentResult",
    "FutureLossExperimentResult",
    "HorizonRegressionResult",
    "TailClassificationResult",
    "capital_backtest_table",
    "compare_choquet_autoregression",
    "coverage_test_table",
    "diversification_table",
    "expanding_capacity_stability",
    "ordered_risk_contributions",
    "rolling_capital_panel",
    "select_model_family_by_validation",
    "run_factor_experiment",
    "run_future_loss_experiment",
    "run_tail_classification_experiment",
    "simulate_regime_switching_losses",
    "static_risk_table",
]
