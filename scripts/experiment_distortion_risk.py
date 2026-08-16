"""Run distortion-risk analysis on observed Bloomberg market losses."""

from __future__ import annotations

# The full pipeline intentionally lives in this executable script.

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from capacities_ml_fin.risk import (
    ExpectedShortfallDistortion,
    ProportionalHazardsDistortion,
    check_risk_measure_axioms,
    distortion_risk_measure,
    expected_shortfall,
)
from mwc_experiments.configuration import load_experiment_config
from mwc_experiments.data import load_raw_market_data, prepare_market_data
from mwc_experiments.workflows import (
    capital_backtest_table,
    coverage_test_table,
    diversification_table,
    ordered_risk_contributions,
    rolling_capital_panel,
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
real_data_config = config["real_data"]
risk_config = config["risk"]
axiom_config = config["axiom_test"]
expected_shortfall_alpha = float(risk_config["expected_shortfall_alpha"])
proportional_hazards_gammas = tuple(
    float(value) for value in risk_config["proportional_hazards_gammas"]
)

real_assets = tuple(str(value) for value in real_data_config["assets"])
correlation_assets = tuple(
    str(value) for value in real_data_config["correlation_assets"]
)
diversification_assets = tuple(
    str(value) for value in real_data_config["diversification_assets"]
)
if len(diversification_assets) != 2:
    raise ValueError("real_data.diversification_assets must contain two assets.")
unknown_assets = set((*correlation_assets, *diversification_assets)) - set(
    real_assets
)
if unknown_assets:
    raise ValueError(
        "Correlation and diversification assets must be included in real_data.assets: "
        + ", ".join(sorted(unknown_assets))
    )

prepared = prepare_market_data(load_raw_market_data(paths))
equity_losses = prepared.equity_losses.loc[:, list(real_assets)].dropna()
real = equity_losses.rename(
    columns={asset: f"{asset}_loss" for asset in real_assets}
)
real["portfolio_loss"] = equity_losses.mean(axis=1)
real["vix"] = prepared.portfolio_features_equal["vix"].reindex(real.index)
real = real.dropna()
real["stress"] = real["vix"] >= float(
    real_data_config["stress_vix_threshold"]
)

losses = real["portfolio_loss"]
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
    stress_mask=real["stress"],
)
coverage = coverage_test_table(losses, capital)
stress_comparison = backtests[[
    "average_capital",
    "exceedance_frequency",
    "mean_exceedance",
]].unstack("sample")

first_asset, second_asset = diversification_assets
diversification_input = pd.DataFrame(
    {
        "asset_a_loss": real[f"{first_asset}_loss"],
        "asset_b_loss": real[f"{second_asset}_loss"],
    }
)
diversification = diversification_table(diversification_input)
diversification.insert(0, "asset A", first_asset)
diversification.insert(1, "asset B", second_asset)

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
axiom_rows: list[dict[str, object]] = []
first = real[f"{first_asset}_loss"].to_numpy()
second = real[f"{second_asset}_loss"].to_numpy()
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
    "real losses": save_table(
        real,
        "experiment_3_real_losses.parquet",
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
losses.plot(ax=axes[0], linewidth=0.7, title="Observed portfolio losses")
axes[0].fill_between(
    real.index,
    axes[0].get_ylim()[0],
    axes[0].get_ylim()[1],
    where=real["stress"].to_numpy(),
    alpha=0.12,
    label=f'VIX >= {float(real_data_config["stress_vix_threshold"]):g}',
)
axes[0].legend()
losses.plot.hist(
    ax=axes[1],
    bins=100,
    density=True,
    title="Observed loss distribution",
)
for axis in axes:
    axis.grid(alpha=0.2)
figure.tight_layout()
artifacts["loss process figure"] = save_figure(
    figure,
    "experiment_3_loss_process.png",
    paths,
)

correlation_columns = [f"{asset}_loss" for asset in correlation_assets]
calm_correlation = real.loc[~real["stress"], correlation_columns].corr()
stress_correlation = real.loc[real["stress"], correlation_columns].corr()
figure, axes = plt.subplots(1, 2, figsize=(11, 4))
image = None
for axis, matrix, title in zip(
    axes,
    (calm_correlation, stress_correlation),
    ("Observed calm-period correlation", "Observed stress-period correlation"),
):
    image = axis.imshow(matrix, vmin=-1, vmax=1)
    labels = list(correlation_assets)
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
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
    title="Capital under alternative distortions - observed losses",
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
axis.set_title("Cumulative ordered contributions to empirical ES 97.5%")
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
axis.set_title("One-step-ahead rolling capital - observed losses")
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
    kind="bar",
    ax=axes[0],
    title="Average capital - observed losses",
)
backtests.xs("all", level="sample")[
    "mean_absolute_capital_change"
].plot(
    kind="bar",
    ax=axes[1],
    title="Capital instability - observed losses",
)
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
    title=f"Observed diversification benefit - {first_asset} and {second_asset}",
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
