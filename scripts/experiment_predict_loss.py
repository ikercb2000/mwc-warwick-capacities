"""Run and persist the complete future portfolio-loss experiment."""

from __future__ import annotations

# The full pipeline intentionally lives in this executable script.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mwc_experiments.configuration import (
    load_experiment_config,
    parameter_grid_overrides,
)
from mwc_experiments.data import load_or_build_processed_data
from mwc_experiments.evaluation import (
    clipping_diagnostics,
    fit_empirical_stress_definition,
    hac_model_comparison,
    high_loss_regime_metrics,
    orientation_table,
    orientation_tables,
    plot_actual_predictions,
    plot_matrix,
    plot_metric_ranking,
    plot_shapley,
    regression_estimation_robustness,
    regression_regime_metrics,
    top_interactions,
)
from mwc_experiments.workflows import (
    compare_validation_stress_regimes,
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


args = parse_experiment_args(
    "Run the complete future-loss experiment.",
    experiment_id="future_loss",
)
quick = args.quick
verbose = not args.quiet
paths = prepare_output_paths()
config = load_experiment_config("future_loss", paths.root)
dataset_config = config["dataset"]
model_config = config["models"]
analysis_config = config["analysis"]
walk_forward_config = config["walk_forward"]
stress_config = config["validation_stress"]
aggregation_config = config["aggregation"]
execution_config = config["execution"]
HORIZONS = tuple(int(value) for value in dataset_config["horizons"])
MAIN_RISK_FEATURES = tuple(str(value) for value in dataset_config["features"])
MAIN_MODELS = tuple(str(value) for value in model_config["main"])
ORIENTATION_MODELS = tuple(str(value) for value in model_config["orientation"])
CAP_WEIGHT_MODELS = tuple(str(value) for value in model_config["cap_weight"])
EXTREME_ROBUSTNESS_MODELS = tuple(
    str(value) for value in model_config["extreme_robustness"]
)
STABILITY_CUTOFFS = tuple(
    str(value) for value in analysis_config["stability_cutoffs"]
)
CAP_WEIGHT_HORIZON = int(analysis_config["cap_weight_horizon"])
STABILITY_HORIZON = int(analysis_config["stability_horizon"])
PARAMETER_GRIDS = parameter_grid_overrides(config)
AGGREGATION_MODEL = str(aggregation_config["model_name"])
AGGREGATION_BASE_MODELS = tuple(
    str(value) for value in aggregation_config["base_models"]
)
data = load_or_build_processed_data(paths)
dataset = data.equal_weight_dataset
result = run_future_loss_experiment(
    dataset,
    portfolio="equal",
    features=MAIN_RISK_FEATURES,
    horizons=HORIZONS,
    quick=quick,
    model_names=MAIN_MODELS,
    random_state=int(execution_config["random_state"]),
    parameter_grids=PARAMETER_GRIDS,
    oos_start=str(walk_forward_config["oos_start"]),
    training_window_years=int(
        walk_forward_config["training_window_years"]
    ),
    validation_window_months=int(
        walk_forward_config["validation_window_months"]
    ),
    oos_block_years=int(walk_forward_config["oos_block_years"]),
    aggregation_model_name=AGGREGATION_MODEL,
    aggregation_base_models=AGGREGATION_BASE_MODELS,
    verbose=verbose,
)
cap_result = run_future_loss_experiment(
    data.cap_weight_dataset,
    portfolio="cap",
    features=MAIN_RISK_FEATURES,
    horizons=(CAP_WEIGHT_HORIZON,),
    quick=quick,
    model_names=CAP_WEIGHT_MODELS,
    random_state=int(execution_config["random_state"]),
    parameter_grids=PARAMETER_GRIDS,
    oos_start=str(walk_forward_config["oos_start"]),
    training_window_years=int(
        walk_forward_config["training_window_years"]
    ),
    validation_window_months=int(
        walk_forward_config["validation_window_months"]
    ),
    oos_block_years=int(walk_forward_config["oos_block_years"]),
    verbose=verbose,
)
stability = expanding_capacity_stability(
    dataset[list(MAIN_RISK_FEATURES)],
    dataset[f"future_loss_h{STABILITY_HORIZON}"],
    cutoffs=STABILITY_CUTOFFS,
    task="regression",
    purge=int(analysis_config["stability_purge"]),
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
    q_scales = (
        horizon_result.selected_parameters[["fitted q"]].dropna()
        if "fitted q" in horizon_result.selected_parameters
        else pd.DataFrame()
    )
    artifacts[f"scaled-q estimates h{horizon}"] = save_table(
        q_scales,
        f"experiment_2a_q_scales_h{horizon}.csv",
        paths,
    )
    artifacts[f"walk-forward folds h{horizon}"] = save_table(
        horizon_result.fold_summary,
        f"experiment_2a_walk_forward_folds_h{horizon}.csv",
        paths,
    )
    artifacts[f"walk-forward metrics h{horizon}"] = save_table(
        horizon_result.fold_metrics,
        f"experiment_2a_walk_forward_metrics_h{horizon}.csv",
        paths,
    )
    artifacts[f"orientation history h{horizon}"] = save_table(
        horizon_result.orientation_history,
        f"experiment_2a_orientation_history_h{horizon}.csv",
        paths,
    )
    artifacts[f"Shapley history h{horizon}"] = save_table(
        horizon_result.shapley_history,
        f"experiment_2a_shapley_history_h{horizon}.csv",
        paths,
    )
    artifacts[f"validation-selected Choquet h{horizon}"] = save_table(
        select_model_family_by_validation(
            horizon_result.metrics.drop(
                index=AGGREGATION_MODEL,
                errors="ignore",
            ),
            family_prefix="Choquet",
            score_column="validation RMSE",
        ),
        f"experiment_2a_validation_selected_choquet_h{horizon}.csv",
        paths,
    )
    artifacts[f"orientation ablation h{horizon}"] = save_table(
        horizon_result.metrics[
            horizon_result.metrics.index.isin(ORIENTATION_MODELS)
        ],
        f"experiment_2a_orientation_ablation_h{horizon}.csv",
        paths,
    )
    artifacts[f"final orientations h{horizon}"] = save_table(
        orientation_tables(
            horizon_result.fitted_models,
            key_names=("model",),
        ),
        f"experiment_2a_orientations_h{horizon}.csv",
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
        quantile=float(analysis_config["high_loss_quantile"]),
    ).sort_values("RMSE")
    artifacts[f"high-loss metrics h{horizon}"] = save_table(
        tail_metrics,
        f"experiment_2a_high_loss_metrics_h{horizon}.csv",
        paths,
    )

    split = horizon_result.final_split
    final_predictions = horizon_result.predictions.loc[split.y_test.index]
    X_fit = pd.concat([split.X_train, split.X_validation])
    y_fit = pd.concat([split.y_train, split.y_validation])
    stress = fit_empirical_stress_definition(X_fit, y_fit)
    fit_extreme_mask = stress.mask(X_fit, y_fit)
    test_stress_mask = stress.mask(split.X_test, split.y_test)
    stress_audit = pd.DataFrame(
        [
            stress.audit(X_fit, y_fit, sample="estimation"),
            stress.audit(split.X_test, split.y_test, sample="test"),
        ]
    ).set_index("sample")
    artifacts[f"empirical stress audit h{horizon}"] = save_table(
        stress_audit,
        f"experiment_2a_empirical_stress_audit_h{horizon}.csv",
        paths,
    )
    artifacts[f"empirical stress metrics h{horizon}"] = save_table(
        regression_regime_metrics(
            split.y_test,
            final_predictions,
            test_stress_mask,
        ),
        f"experiment_2a_empirical_stress_metrics_h{horizon}.csv",
        paths,
    )
    clipping_reference = horizon_result.fitted_models.get("OLS")
    if clipping_reference is not None:
        clipping_audit = pd.concat(
            [
                clipping_diagnostics(
                    clipping_reference,
                    X_fit,
                    sample="estimation",
                ),
                clipping_diagnostics(
                    clipping_reference,
                    split.X_test,
                    sample="test",
                ),
            ]
        )
        clipping_audit.index.name = "feature"
        artifacts[f"clipping audit h{horizon}"] = save_table(
            clipping_audit,
            f"experiment_2a_clipping_audit_h{horizon}.csv",
            paths,
        )
    robustness_models = {
        model: horizon_result.fitted_models[model]
        for model in EXTREME_ROBUSTNESS_MODELS
        if model in horizon_result.fitted_models
    }
    artifacts[f"extreme-estimation robustness h{horizon}"] = save_table(
        regression_estimation_robustness(
            robustness_models,
            X_fit,
            y_fit,
            split.X_test,
            split.y_test,
            final_predictions,
            fit_extreme_mask=fit_extreme_mask,
            test_stress_mask=test_stress_mask,
        ),
        f"experiment_2a_extreme_estimation_robustness_h{horizon}.csv",
        paths,
    )
    validation_comparison = compare_validation_stress_regimes(
        dataset[list(MAIN_RISK_FEATURES)],
        dataset[f"future_loss_h{horizon}"],
        model_names=MAIN_MODELS,
        horizon=horizon,
        quick=quick,
        random_state=int(execution_config["random_state"]),
        validation_start=str(stress_config["validation_start"]),
        validation_end=str(stress_config["validation_end"]),
        stress_start=str(stress_config["stress_start"]),
        stress_end=str(stress_config["stress_end"]),
        parameter_grids=PARAMETER_GRIDS,
    )
    artifacts[f"validation-stress comparison h{horizon}"] = save_table(
        validation_comparison.metrics,
        f"experiment_2a_validation_stress_comparison_h{horizon}.csv",
        paths,
    )
    artifacts[f"validation-stress samples h{horizon}"] = save_table(
        validation_comparison.sample_summary,
        f"experiment_2a_validation_stress_samples_h{horizon}.csv",
        paths,
    )
    artifacts[f"validation-stress selection summary h{horizon}"] = save_table(
        validation_comparison.selection_summary,
        f"experiment_2a_validation_stress_selection_summary_h{horizon}.csv",
        paths,
    )
    artifacts[f"validation-stress failures h{horizon}"] = save_table(
        validation_comparison.failures,
        f"experiment_2a_validation_stress_failures_h{horizon}.csv",
        paths,
        index=False,
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
            max_lags=max(int(analysis_config["hac_minimum_lags"]), horizon),
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
            top_interactions(
                horizon_result.interactions[model],
                n=int(analysis_config["top_interactions"]),
            ),
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
        for model in (
            *CAP_WEIGHT_MODELS,
            "MLP",
            "Fuzzy Choquet neural network",
            AGGREGATION_MODEL,
        )
        if model in horizon_result.predictions
    ]
    axis = plot_actual_predictions(
        horizon_result.split.y_test,
        horizon_result.predictions,
        models=preferred,
        title=f"Observed and predicted future loss — h={horizon}",
        start=str(analysis_config["forecast_plot_start"]),
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
cap_metrics = cap_result.horizons[CAP_WEIGHT_HORIZON].metrics
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
        "equal weight": result.horizons[CAP_WEIGHT_HORIZON].metrics["RMSE"],
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
representative = result.horizons[STABILITY_HORIZON].fitted_models.get(
    "Choquet 2-additive"
)
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
axis.legend(
    loc="center left",
    bbox_to_anchor=(1.01, 0.5),
    fontsize=8,
)
axis.figure.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
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
