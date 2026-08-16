"""Run and persist the complete Choquet autoregression robustness experiment."""

from __future__ import annotations

# The full pipeline intentionally lives in this executable script.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

from mwc_experiments.configuration import load_experiment_config
from mwc_experiments.data import load_or_build_processed_data
from mwc_experiments.evaluation import (
    hac_model_comparison,
    plot_actual_predictions,
    plot_matrix,
    plot_shapley,
)
from mwc_experiments.workflows import compare_choquet_autoregression
from mwc_experiments.workflows.experiment_support import (
    prepare_output_paths,
    print_artifacts,
    save_figure,
    save_table,
)


verbose = True
paths = prepare_output_paths()
config = load_experiment_config("autoregression", paths.root)
dataset_config = config["dataset"]
split_config = config["split"]
selection_config = config["selection"]
analysis_config = config["analysis"]
dataset = load_or_build_processed_data(paths).equal_weight_dataset
losses = dataset[str(dataset_config["loss_column"])].dropna()
adf = adfuller(
    losses.to_numpy(),
    autolag=str(analysis_config["adf_autolag"]),
)
adf_table = pd.Series(
    {
        "ADF statistic": adf[0],
        "p-value": adf[1],
        "used lags": adf[2],
        "observations": adf[3],
        **{
            f"critical value {key}": value
            for key, value in adf[4].items()
        },
    },
    name="value",
)
if verbose:
    print("[autoregression] selecting linear and Choquet lag orders", flush=True)
result = compare_choquet_autoregression(
    losses,
    train_end=str(split_config["train_end"]),
    validation_end=str(split_config["validation_end"]),
    candidate_lags=tuple(int(value) for value in selection_config["candidate_lags"]),
)
actual_test = losses.reindex(result.predictions.index)
errors = result.predictions.sub(actual_test, axis=0)
hac = hac_model_comparison(
    actual_test,
    result.predictions,
    reference=str(analysis_config["hac_reference"]),
    loss="squared",
    max_lags=int(analysis_config["hac_max_lags"]),
)
vix = dataset[str(dataset_config["stress_column"])].reindex(
    result.predictions.index
)
high_vix = vix >= vix.quantile(float(analysis_config["stress_quantile"]))
regime_rows: list[dict[str, float | int | str]] = []
for model in result.predictions:
    error = actual_test - result.predictions[model]
    regime_rows.append(
        {
            "model": model,
            "RMSE all": float(np.sqrt(np.mean(error**2))),
            "RMSE top-VIX decile": float(
                np.sqrt(np.mean(error[high_vix] ** 2))
            ),
            "MAE top-VIX decile": float(
                np.mean(np.abs(error[high_vix]))
            ),
            "top-VIX observations": int(high_vix.sum()),
        }
    )
regime_metrics = (
    pd.DataFrame(regime_rows)
    .set_index("model")
    .sort_values("RMSE top-VIX decile")
)
model_details = pd.DataFrame(
    {
        "value": {
            "selected linear lags": result.selected_lags["Linear AR"],
            "selected Choquet lags": result.selected_lags["Choquet AR"],
            "Choquet phi": result.choquet_model.phi_,
            "stationarity bound respected": abs(result.choquet_model.phi_) < 1,
        }
    }
)

artifacts: dict[str, Path] = {
    "loss summary": save_table(
        losses.describe().rename("portfolio loss"),
        "robustness_ar_loss_summary.csv",
        paths,
    ),
    "ADF diagnostics": save_table(
        adf_table,
        "robustness_ar_adf.csv",
        paths,
    ),
    "validation": save_table(
        result.validation,
        "robustness_ar_validation.csv",
        paths,
    ),
    "test metrics": save_table(
        result.test_metrics,
        "robustness_ar_test_metrics.csv",
        paths,
    ),
    "predictions": save_table(
        result.predictions,
        "robustness_ar_predictions.parquet",
        paths,
    ),
    "Shapley": save_table(
        result.choquet_shapley,
        "robustness_ar_shapley.csv",
        paths,
    ),
    "interactions": save_table(
        result.choquet_interactions,
        "robustness_ar_interactions.csv",
        paths,
    ),
    "HAC comparison": save_table(
        hac,
        "robustness_ar_hac_comparison.csv",
        paths,
    ),
    "high-VIX metrics": save_table(
        regime_metrics,
        "robustness_ar_high_vix_metrics.csv",
        paths,
    ),
    "model details": save_table(
        model_details,
        "robustness_ar_model_details.csv",
        paths,
    ),
}

figure, axes = plt.subplots(3, 1, figsize=(13, 10))
losses.plot(ax=axes[0], linewidth=0.7, title="Equal-weight portfolio loss")
plot_acf(losses, lags=int(analysis_config["diagnostic_lags"]), ax=axes[1])
plot_pacf(
    losses,
    lags=int(analysis_config["diagnostic_lags"]),
    ax=axes[2],
    method="ywm",
)
figure.tight_layout()
artifacts["time-series diagnostics figure"] = save_figure(
    figure,
    "robustness_ar_time_series_diagnostics.png",
    paths,
)

axis = plot_actual_predictions(
    actual_test,
    result.predictions,
    title="One-step-ahead portfolio-loss forecasts",
    start=str(analysis_config["forecast_plot_start"]),
)
artifacts["forecast figure"] = save_figure(
    axis.figure,
    "robustness_ar_forecasts.png",
    paths,
)

figure, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
errors.plot(ax=axes[0], linewidth=0.7, title="Forecast errors")
error_window = int(analysis_config["error_volatility_window"])
errors.rolling(error_window).std().plot(
    ax=axes[1], title=f"Rolling {error_window}-day forecast-error volatility"
)
for axis in axes:
    axis.grid(alpha=0.2)
figure.tight_layout()
artifacts["forecast-error figure"] = save_figure(
    figure,
    "robustness_ar_forecast_errors.png",
    paths,
)

axis = plot_shapley(
    result.choquet_shapley,
    title="Choquet AR lag importance",
)
artifacts["lag importance figure"] = save_figure(
    axis.figure,
    "robustness_ar_shapley.png",
    paths,
)
axis = plot_matrix(
    result.choquet_interactions,
    title="Choquet AR lag interaction indices",
)
artifacts["lag interaction figure"] = save_figure(
    axis.figure,
    "robustness_ar_interactions.png",
    paths,
)
print_artifacts(artifacts)
