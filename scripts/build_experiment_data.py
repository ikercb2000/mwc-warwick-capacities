"""Build, audit and persist the complete model-ready data layer."""

from __future__ import annotations

# The complete data-preparation pipeline lives in this executable script.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mwc_experiments.configuration import load_experiment_config
from mwc_experiments.data import build_experiment_data, save_processed_data
from mwc_experiments.workflows.experiment_support import (
    prepare_output_paths,
    print_artifacts,
    save_figure,
    save_table,
)


paths = prepare_output_paths()
config = load_experiment_config("data_preparation", paths.root)
dataset_config = config["dataset"]
HORIZONS = tuple(int(value) for value in dataset_config["horizons"])
TAIL_ALPHAS = tuple(float(value) for value in dataset_config["tail_alphas"])
data = build_experiment_data(
    paths,
    horizons=HORIZONS,
    tail_alphas=TAIL_ALPHAS,
    tail_window=int(dataset_config["tail_window"]),
    min_tail_history=int(dataset_config["minimum_tail_history"]),
)
processed = save_processed_data(data, paths)
prepared = data.prepared
dataset = data.equal_weight_dataset
artifacts: dict[str, Path] = {
    "equal-weight dataset": processed["equal_weight"],
    "cap-weight dataset": processed["cap_weight"],
}
artifacts.update(
    {
        f"factor frame {asset}": path
        for asset, path in processed["factor_frames"].items()
    }
)

portfolio_summary = pd.DataFrame(
    {
        "equal weight": prepared.equal_weight_losses,
        "lagged market-cap weight": prepared.cap_weight_losses,
    }
).describe().T
event_rows: list[dict[str, float | int]] = []
for horizon in HORIZONS:
    for alpha in TAIL_ALPHAS:
        label = str(alpha).replace(".", "p")
        valid = dataset[f"tail_event_h{horizon}_a{label}"].dropna().astype(int)
        event_rows.append(
            {
                "horizon": horizon,
                "alpha": alpha,
                "observations": len(valid),
                "events": int(valid.sum()),
                "event frequency": float(valid.mean()),
            }
        )

tables = {
    "raw audit": save_table(data.raw.audit(), "data_raw_audit.csv", paths, index=False),
    "equity return summary": save_table(
        prepared.equity_returns.describe().T,
        "data_equity_return_summary.csv",
        paths,
    ),
    "portfolio summary": save_table(
        portfolio_summary, "data_portfolio_summary.csv", paths
    ),
    "cap-weight summary": save_table(
        prepared.lagged_cap_weights.dropna().describe().T[["mean", "std", "min", "max"]],
        "data_cap_weight_summary.csv",
        paths,
    ),
    "dataset missingness": save_table(
        dataset.isna().mean().sort_values(ascending=False).rename("missing fraction"),
        "data_dataset_missingness.csv",
        paths,
    ),
    "dataset summary": save_table(
        dataset.describe().T, "data_dataset_summary.csv", paths
    ),
    "tail-event audit": save_table(
        pd.DataFrame(event_rows).set_index(["horizon", "alpha"]),
        "data_tail_event_audit.csv",
        paths,
    ),
}
artifacts.update(tables)

wealth = np.exp(prepared.equity_returns.fillna(0.0).cumsum())
axis = wealth.plot(figsize=(12, 5), linewidth=1.0)
axis.set_title("Equity total-return wealth indices, including NVDA")
axis.set_ylabel("wealth")
axis.grid(alpha=0.2)
axis.figure.tight_layout()
artifacts["equity wealth figure with NVDA"] = save_figure(
    axis.figure, "data_equity_wealth_with_nvda.png", paths
)

wealth_without_nvda = wealth.drop(columns="NVDA")
axis = wealth_without_nvda.plot(figsize=(12, 5), linewidth=1.0)
axis.set_title("Equity total-return wealth indices, excluding NVDA")
axis.set_ylabel("wealth")
axis.grid(alpha=0.2)
axis.figure.tight_layout()
artifacts["equity wealth figure without NVDA"] = save_figure(
    axis.figure, "data_equity_wealth_without_nvda.png", paths
)

correlation = prepared.equity_returns.corr()
for vix_title, fred_panel, output_name in (
    ("including VIX", prepared.fred, "data_raw_market_panels_with_vix.png"),
    (
        "excluding VIX",
        prepared.fred.drop(columns="vix"),
        "data_raw_market_panels_without_vix.png",
    ),
):
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    image = axes[0].imshow(correlation, vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(correlation)), correlation.columns, rotation=90)
    axes[0].set_yticks(range(len(correlation)), correlation.index)
    axes[0].set_title("Full-sample equity return correlation")
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    fred_panel.plot(ax=axes[1], linewidth=0.8)
    axes[1].set_title(f"Available FRED series, {vix_title}")
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    artifacts[f"raw panels figure {vix_title.lower()}"] = save_figure(
        figure, output_name, paths
    )

portfolio_wealth = pd.DataFrame(
    {
        "equal weight": np.exp(prepared.equal_weight_returns.fillna(0.0).cumsum()),
        "lagged market-cap weight": np.exp(
            prepared.cap_weight_returns.fillna(0.0).cumsum()
        ),
    }
)
axis = portfolio_wealth.plot(figsize=(11, 4), title="Portfolio wealth indices")
axis.grid(alpha=0.2)
axis.figure.tight_layout()
artifacts["portfolio wealth figure"] = save_figure(
    axis.figure, "data_portfolio_wealth.png", paths
)

axis = prepared.lagged_cap_weights.dropna().iloc[::63].plot.area(
    figsize=(12, 5), title="Lagged market-cap portfolio weights"
)
axis.set_ylabel("weight")
axis.figure.tight_layout()
artifacts["cap weights figure"] = save_figure(
    axis.figure, "data_cap_weights.png", paths
)

features = [
    "vix",
    "realized_volatility_20d",
    "drawdown_severity",
    "liquidity_stress_20d",
    "baa_credit_spread_pct",
    "average_equity_correlation_60d",
]
axes = dataset[features].plot(subplots=True, figsize=(12, 12), sharex=True)
figure = np.asarray(axes).reshape(-1)[0].figure
figure.tight_layout()
artifacts["predictor history figure"] = save_figure(
    figure, "data_predictor_history.png", paths
)
print_artifacts(artifacts)
