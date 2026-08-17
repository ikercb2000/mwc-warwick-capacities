"""Run small model-family checks without producing final experiment artifacts."""

from __future__ import annotations

# The smoke pipeline intentionally lives in this executable script.

from mwc_experiments.data import load_or_build_processed_data
from mwc_experiments.paths import RepoPaths
from mwc_experiments.workflows import (
    run_factor_experiment,
    run_future_loss_experiment,
    run_tail_classification_experiment,
)


paths = RepoPaths.discover()
data = load_or_build_processed_data(paths)

factor = run_factor_experiment(
    {"AAPL": data.factor_frames["AAPL"]},
    assets=("AAPL",),
    quick=True,
    model_names=(
        "OLS",
        "Choquet 1-additive",
        "Choquet 1-additive scaled-q",
        "Choquet 2-additive",
        "Choquet 2-additive L1",
        "Choquet 2-additive scaled-q",
        "Choquet 2-additive scaled-q L1",
    ),
)
print("\nFactor smoke test")
print(factor.metrics.to_string())

regression = run_future_loss_experiment(
    data.equal_weight_dataset,
    horizons=(1,),
    quick=True,
    model_names=(
        "Historical mean",
        "OLS",
        "Choquet 1-additive",
        "Choquet 1-additive scaled-q",
        "Choquet 2-additive",
        "Choquet 2-additive L1",
        "Choquet 2-additive scaled-q",
        "Choquet 2-additive scaled-q L1",
    ),
)
print("\nFuture-loss smoke test")
print(regression.horizons[1].metrics.to_string())

classification = run_tail_classification_experiment(
    data.equal_weight_dataset,
    horizons=(1,),
    alpha=0.95,
    quick=True,
    model_names=(
        "Rolling prior probability",
        "Logistic",
        "Choquistic 2-additive",
    ),
)
print("\nTail-classification smoke test")
print(classification[1].metrics.to_string())
