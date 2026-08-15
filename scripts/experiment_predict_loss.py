"""Run and persist the complete future portfolio-loss experiment."""

from __future__ import annotations

# The full pipeline intentionally lives in this executable script.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mwc_experiments.data import load_or_build_processed_data
from mwc_experiments.evaluation import (
    hac_model_comparison,
    high_loss_regime_metrics,
    orientation_table,
    plot_actual_predictions,
    plot_matrix,
    plot_metric_ranking,
    plot_shapley,
    top_interactions,
)
from mwc_experiments.settings import HORIZONS, MAIN_RISK_FEATURES
from mwc_experiments.workflows import (
    expanding_capacity_stability,
    run_future_loss_experiment,
    select_model_family_by_validation,
)
from mwc_experiments.workflows.experiment_support import (
    artifact_slug,
    parse_experiment_args,
    prepare_output_paths,
    print_artifacts,
    save_figure,
    save_table,
)


STABILITY_CUTOFFS = (
    "2018-12-31",
    "2020-12-31",
    "2022-12-30",
    "2024-12-31",
    "2026-07-30",
)
CAP_WEIGHT_MODELS = (
    "Historical mean",
    "OLS",
    "Monotone linear",
    "Explicit interactions",
    "Gradient boosting",
    "Choquet 1-additive",
    "Choquet 2-additive",
    "Choquet 2-additive L1",
)


args = parse_experiment_args("Run the complete future-loss experiment.")
quick = args.quick
verbose = not args.quiet
paths = prepare_output_paths()
data = load_or_build_processed_data(paths)
dataset = data.equal_weight_dataset
result = run_future_loss_experiment(
    dataset,
    portfolio="equal",
    horizons=HORIZONS,
    quick=quick,
    verbose=verbose,
)
cap_result = run_future_loss_experiment(
    data.cap_weight_dataset,
    portfolio="cap",
    horizons=(5,),
    quick=quick,
    model_names=CAP_WEIGHT_MODELS,
    verbose=verbose,
)
stability = expanding_capacity_stability(
    dataset[list(MAIN_RISK_FEATURES)],
    dataset["future_loss_h5"],
    cutoffs=STABILITY_CUTOFFS,
    task="regression",
    purge=5,
    verbose=verbose,
)
artifacts: dict[str, Path] = {}
metric_panels: list[pd.DataFrame] = []

for horizon, horizon_result in result.horizons.items():
    metrics = horizon_result.metrics.copy()
    metrics["horizon"] = horizon
    metric_panels.append(metrics.reset_index())
    artifacts[f"metrics h{horizon}"] = save_table(
        horizon_result.metrics,
        f"experiment_2a_regression_metrics_h{horizon}.csv",
        paths,
    )
    artifacts[f"predictions h{horizon}"] = save_table(
        horizon_result.predictions,
        f"experiment_2a_predictions_h{horizon}.parquet",
        paths,
    )
    artifacts[f"selected parameters h{horizon}"] = save_table(
        horizon_result.selected_parameters,
        f"experiment_2a_selected_parameters_h{horizon}.csv",
        paths,
    )
    artifacts[f"validation-selected Choquet h{horizon}"] = save_table(
        select_model_family_by_validation(
            horizon_result.metrics,
            family_prefix="Choquet",
            score_column="validation RMSE",
        ),
        f"experiment_2a_validation_selected_choquet_h{horizon}.csv",
        paths,
    )
    artifacts[f"failures h{horizon}"] = save_table(
        horizon_result.failures,
        f"experiment_2a_failures_h{horizon}.csv",
        paths,
        index=False,
    )

    tail_metrics = high_loss_regime_metrics(
        horizon_result.split.y_test,
        horizon_result.predictions,
        quantile=0.90,
    ).sort_values("RMSE")
    artifacts[f"high-loss metrics h{horizon}"] = save_table(
        tail_metrics,
        f"experiment_2a_high_loss_metrics_h{horizon}.csv",
        paths,
    )
    reference = (
        "Choquet 2-additive"
        if "Choquet 2-additive" in horizon_result.predictions
        else horizon_result.metrics.index[0]
    )
    artifacts[f"HAC comparison h{horizon}"] = save_table(
        hac_model_comparison(
            horizon_result.split.y_test,
            horizon_result.predictions,
            reference=reference,
            loss="squared",
            max_lags=max(10, horizon),
        ),
        f"experiment_2a_hac_comparison_h{horizon}.csv",
        paths,
    )

    for model, shapley in horizon_result.shapley.items():
        legacy_name = model.lower().replace(" ", "_")
        artifacts[f"Shapley h{horizon} {model}"] = save_table(
            shapley,
            f"experiment_2a_shapley_h{horizon}_{legacy_name}.csv",
            paths,
        )
        artifacts[f"interactions h{horizon} {model}"] = save_table(
            horizon_result.interactions[model],
            f"experiment_2a_interactions_h{horizon}_{legacy_name}.csv",
            paths,
        )
        artifacts[f"top interactions h{horizon} {model}"] = save_table(
            top_interactions(horizon_result.interactions[model], n=12),
            f"experiment_2a_top_interactions_h{horizon}_{artifact_slug(model)}.csv",
            paths,
            index=False,
        )

    axis = plot_metric_ranking(
        horizon_result.metrics, "RMSE", lower_is_better=True
    )
    artifacts[f"RMSE ranking figure h{horizon}"] = save_figure(
        axis.figure,
        f"experiment_2a_rmse_ranking_h{horizon}.png",
        paths,
    )
    preferred = [
        model
        for model in CAP_WEIGHT_MODELS
        if model in horizon_result.predictions
    ]
    axis = plot_actual_predictions(
        horizon_result.split.y_test,
        horizon_result.predictions,
        models=preferred,
        title=f"Observed and predicted future loss — h={horizon}",
        start="2020-01-01",
    )
    artifacts[f"forecast figure h{horizon}"] = save_figure(
        axis.figure,
        f"experiment_2a_forecasts_h{horizon}.png",
        paths,
    )
    for model, shapley in horizon_result.shapley.items():
        slug = artifact_slug(model)
        axis = plot_shapley(
            shapley,
            title=f"{model}: Shapley importance, h={horizon}",
        )
        artifacts[f"Shapley figure h{horizon} {model}"] = save_figure(
            axis.figure,
            f"experiment_2a_shapley_h{horizon}_{slug}.png",
            paths,
        )
        axis = plot_matrix(
            horizon_result.interactions[model],
            title=f"{model}: pairwise interaction indices, h={horizon}",
        )
        artifacts[f"interaction figure h{horizon} {model}"] = save_figure(
            axis.figure,
            f"experiment_2a_interactions_h{horizon}_{slug}.png",
            paths,
        )

all_metrics = pd.concat(metric_panels, ignore_index=True).set_index(
    ["model", "horizon"]
)
artifacts["all metrics"] = save_table(
    all_metrics,
    "experiment_2a_all_metrics.csv",
    paths,
)
cap_metrics = cap_result.horizons[5].metrics
artifacts["cap-weight robustness"] = save_table(
    cap_metrics,
    "experiment_2a_cap_weight_robustness_h5.csv",
    paths,
)
artifacts["cap-weight validation-selected Choquet"] = save_table(
    select_model_family_by_validation(
        cap_metrics,
        family_prefix="Choquet",
        score_column="validation RMSE",
    ),
    "experiment_2a_cap_weight_validation_selected_choquet_h5.csv",
    paths,
)
comparison = pd.concat(
    {
        "equal weight": result.horizons[5].metrics["RMSE"],
        "lagged market-cap weight": cap_metrics["RMSE"],
    },
    axis=1,
).sort_values("equal weight")
artifacts["portfolio comparison"] = save_table(
    comparison,
    "experiment_2a_portfolio_comparison_h5.csv",
    paths,
)
artifacts["Shapley stability"] = save_table(
    stability.shapley,
    "experiment_2a_shapley_stability_h5.csv",
    paths,
)
artifacts["interaction stability"] = save_table(
    stability.interaction_long,
    "experiment_2a_interaction_stability_h5_long.csv",
    paths,
    index=False,
)
artifacts["interaction stability summary"] = save_table(
    stability.interaction_stability(),
    "experiment_2a_interaction_stability_h5_summary.csv",
    paths,
)
artifacts["stability failures"] = save_table(
    stability.failures,
    "experiment_2a_stability_failures_h5.csv",
    paths,
    index=False,
)
representative = result.horizons[5].fitted_models.get("Choquet 2-additive")
if representative is not None:
    artifacts["orientation"] = save_table(
        orientation_table(representative),
        "experiment_2a_orientation_h5_choquet_2-additive.csv",
        paths,
    )

axis = plot_matrix(
    dataset[list(MAIN_RISK_FEATURES)].corr(),
    title="Correlation of the main risk predictors",
)
artifacts["predictor correlation figure"] = save_figure(
    axis.figure,
    "experiment_2a_predictor_correlation.png",
    paths,
)

figure, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
for axis, horizon in zip(axes, HORIZONS):
    dataset[f"future_loss_h{horizon}"].plot(ax=axis, linewidth=0.8)
    axis.set_title(f"Future portfolio loss, h={horizon}")
    axis.grid(alpha=0.2)
figure.tight_layout()
artifacts["future-loss history figure"] = save_figure(
    figure,
    "experiment_2a_future_loss_history.png",
    paths,
)

figure, axis = plt.subplots(figsize=(11, 5))
plotted = all_metrics.reset_index()
for model, group in plotted.groupby("model"):
    axis.plot(group["horizon"], group["RMSE"], marker="o", label=model)
axis.set_xticks(list(HORIZONS))
axis.set_xlabel("Forecast horizon")
axis.set_ylabel("Out-of-sample RMSE")
axis.set_title("RMSE across forecast horizons")
axis.legend(ncol=2, fontsize=8)
axis.grid(alpha=0.25)
figure.tight_layout()
artifacts["cross-horizon RMSE figure"] = save_figure(
    figure,
    "experiment_2a_rmse_across_horizons.png",
    paths,
)

axis = stability.shapley.plot(
    figsize=(12, 5),
    marker="o",
    title="5-day loss: expanding-window Shapley stability",
)
axis.set_ylabel("Shapley importance")
axis.grid(alpha=0.25)
axis.figure.tight_layout()
artifacts["stability figure"] = save_figure(
    axis.figure,
    "experiment_2a_shapley_stability_h5.png",
    paths,
)

axis = comparison.plot(
    kind="bar",
    figsize=(11, 5),
    title="Five-day RMSE by portfolio construction",
)
axis.set_ylabel("RMSE")
axis.grid(axis="y", alpha=0.25)
axis.figure.tight_layout()
artifacts["portfolio robustness figure"] = save_figure(
    axis.figure,
    "experiment_2a_portfolio_robustness_h5.png",
    paths,
)
print_artifacts(artifacts)
