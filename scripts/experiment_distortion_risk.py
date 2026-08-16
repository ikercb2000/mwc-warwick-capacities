"""Run and persist the complete distortion-risk simulation experiment."""

from __future__ import annotations

# The full pipeline intentionally lives in this executable script.

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from capacities_ml_fin.risk import (
    ExpectedShortfallDistortion,
    ProportionalHazardsDistortion,
    check_risk_measure_axioms,
    distortion_risk_measure,
    expected_shortfall,
)
from mwc_experiments.configuration import load_experiment_config
from mwc_experiments.workflows import (
    capital_backtest_table,
    coverage_test_table,
    diversification_table,
    ordered_risk_contributions,
    rolling_capital_panel,
    simulate_regime_switching_losses,
    static_risk_table,
)
from mwc_experiments.workflows.experiment_support import (
    prepare_output_paths,
    print_artifacts,
    save_figure,
    save_table,
)


paths = prepare_output_paths()
config = load_experiment_config("distortion_risk", paths.root)
execution_config = config["execution"]
simulation_config = config["simulation"]
risk_config = config["risk"]
axiom_config = config["axiom_test"]
n_observations = int(simulation_config["n_observations"])
random_state = int(execution_config["random_state"])
expected_shortfall_alpha = float(risk_config["expected_shortfall_alpha"])
proportional_hazards_gammas = tuple(
    float(value) for value in risk_config["proportional_hazards_gammas"]
)
simulated = simulate_regime_switching_losses(
    n_observations=n_observations,
    random_state=random_state,
)
losses = simulated["portfolio_loss"]
static = static_risk_table(losses)
contributions = ordered_risk_contributions(
    losses,
    distortion=ExpectedShortfallDistortion(expected_shortfall_alpha),
)
capital = rolling_capital_panel(
    losses,
    window=int(risk_config["capital_window"]),
)
backtests = capital_backtest_table(
    losses,
    capital,
    stress_mask=simulated["stress"],
)
coverage = coverage_test_table(losses, capital)
diversification = diversification_table(simulated)
stress_comparison = backtests[[
    "average_capital",
    "exceedance_frequency",
    "mean_exceedance",
]].unstack("sample")

first = simulated["asset_a_loss"].to_numpy()
second = simulated["asset_b_loss"].to_numpy()
axiom_rows: list[dict[str, object]] = []
measures = {
    f"ES {100 * expected_shortfall_alpha:g}%": lambda values: expected_shortfall(
        values,
        expected_shortfall_alpha,
    ),
    **{
        f"PH gamma={gamma:.2f}": (
            lambda values, gamma=gamma: distortion_risk_measure(
                values,
                ProportionalHazardsDistortion(gamma),
            )
        )
        for gamma in proportional_hazards_gammas
    },
}
for name, measure in measures.items():
    report = check_risk_measure_axioms(
        measure,
        first,
        second,
        cash=float(axiom_config["cash"]),
        scale=float(axiom_config["scale"]),
    )
    axiom_rows.append({"measure": name, **asdict(report)})
axioms = pd.DataFrame(axiom_rows).set_index("measure")

artifacts: dict[str, Path] = {
    "simulated losses": save_table(
        simulated,
        "experiment_3_simulated_losses.parquet",
        paths,
    ),
    "static risk measures": save_table(
        static,
        "experiment_3_static_risk_measures.csv",
        paths,
    ),
    "rolling capital": save_table(
        capital,
        "experiment_3_rolling_capital.parquet",
        paths,
    ),
    "capital backtests": save_table(
        backtests,
        "experiment_3_capital_backtests.csv",
        paths,
    ),
    "coverage tests": save_table(
        coverage,
        "experiment_3_coverage_tests.csv",
        paths,
    ),
    "stress comparison": save_table(
        stress_comparison,
        "experiment_3_stress_comparison.csv",
        paths,
    ),
    "diversification": save_table(
        diversification,
        "experiment_3_diversification.csv",
        paths,
    ),
    "ES contributions": save_table(
        contributions,
        "experiment_3_es_contributions.parquet",
        paths,
    ),
    "axiom checks": save_table(
        axioms,
        "experiment_3_axiom_checks.csv",
        paths,
    ),
}

figure, axes = plt.subplots(2, 1, figsize=(13, 8))
losses.plot(ax=axes[0], linewidth=0.7, title="Simulated portfolio losses")
axes[0].fill_between(
    simulated.index,
    axes[0].get_ylim()[0],
    axes[0].get_ylim()[1],
    where=simulated["stress"].to_numpy(),
    alpha=0.12,
    label="stress regime",
)
axes[0].legend()
losses.plot.hist(
    ax=axes[1],
    bins=100,
    density=True,
    title="Unconditional loss distribution",
)
for axis in axes:
    axis.grid(alpha=0.2)
figure.tight_layout()
artifacts["loss process figure"] = save_figure(
    figure,
    "experiment_3_loss_process.png",
    paths,
)

calm = simulated.loc[
    ~simulated["stress"],
    ["asset_a_loss", "asset_b_loss", "asset_c_loss"],
].corr()
stress = simulated.loc[
    simulated["stress"],
    ["asset_a_loss", "asset_b_loss", "asset_c_loss"],
].corr()
figure, axes = plt.subplots(1, 2, figsize=(11, 4))
image = None
for axis, matrix, title in zip(
    axes,
    (calm, stress),
    ("Calm-state correlation", "Stress-state correlation"),
):
    image = axis.imshow(matrix, vmin=-1, vmax=1)
    axis.set_xticks(range(3), matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(3), matrix.index)
    axis.set_title(title)
figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
artifacts["regime correlation figure"] = save_figure(
    figure,
    "experiment_3_regime_correlations.png",
    paths,
)

axis = static["capital"].plot(
    kind="barh",
    figsize=(9, 6),
    title="Capital under alternative distortions",
)
axis.set_xlabel("Capital per unit notional")
axis.grid(axis="x", alpha=0.25)
axis.figure.tight_layout()
artifacts["static capital figure"] = save_figure(
    axis.figure,
    "experiment_3_static_capital.png",
    paths,
)

ordered = contributions.sort_values("loss")
figure, axis = plt.subplots(figsize=(10, 4))
axis.plot(
    ordered["loss"],
    ordered["contribution"].cumsum(),
    linewidth=1.2,
)
axis.set_title("Cumulative ordered contributions to ES 97.5%")
axis.set_xlabel("Loss support")
axis.set_ylabel("Cumulative contribution")
axis.grid(alpha=0.25)
figure.tight_layout()
artifacts["ES contribution figure"] = save_figure(
    figure,
    "experiment_3_es_contributions.png",
    paths,
)

figure, axis = plt.subplots(figsize=(14, 5))
losses.plot(ax=axis, linewidth=0.5, label="loss", alpha=0.7)
capital.plot(ax=axis, linewidth=1.0)
axis.set_title("One-step-ahead rolling capital forecasts")
axis.legend(ncol=3)
axis.grid(alpha=0.2)
figure.tight_layout()
artifacts["rolling capital figure"] = save_figure(
    figure,
    "experiment_3_rolling_capital.png",
    paths,
)

figure, axes = plt.subplots(1, 2, figsize=(13, 4))
backtests.xs("all", level="sample")["average_capital"].plot(
    kind="bar", ax=axes[0], title="Average capital"
)
backtests.xs("all", level="sample")[
    "mean_absolute_capital_change"
].plot(kind="bar", ax=axes[1], title="Capital instability")
for axis in axes:
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=30)
figure.tight_layout()
artifacts["backtest figure"] = save_figure(
    figure,
    "experiment_3_capital_backtests.png",
    paths,
)

axis = diversification["diversification benefit"].plot(
    kind="bar",
    figsize=(9, 4),
    title="Recognised diversification benefit",
)
axis.axhline(0.0, linewidth=1)
axis.grid(axis="y", alpha=0.25)
axis.figure.tight_layout()
artifacts["diversification figure"] = save_figure(
    axis.figure,
    "experiment_3_diversification.png",
    paths,
)
print_artifacts(artifacts)
