"""Run and persist the complete financial factor-model experiment."""

from __future__ import annotations

# The full pipeline intentionally lives in this executable script.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mwc_experiments.configuration import (
    load_experiment_config,
    parameter_grid_overrides,
)
from mwc_experiments.data import load_or_build_processed_data
from mwc_experiments.evaluation import (
    capacity_summary,
    clipping_diagnostics,
    fit_empirical_stress_definition,
    hac_model_comparison,
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
    run_factor_experiment,
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
    "Run the complete financial factor experiment.",
    experiment_id="factor_models",
)
quick = args.quick
clipping = args.clipping
evaluation_structure = args.evaluation_structure
verbose = not args.quiet
paths = prepare_output_paths(
    clipping=clipping,
    evaluation_structure=evaluation_structure,
)
config = load_experiment_config("factor_models", paths.root)
dataset_config = config["dataset"]
model_config = config["models"]
analysis_config = config["analysis"]
stress_config = config["validation_stress"]
execution_config = config["execution"]
FACTOR_COLUMNS = tuple(str(value) for value in dataset_config["features"])
ASSETS = tuple(str(value) for value in dataset_config["assets"])
MAIN_MODELS = tuple(str(value) for value in model_config["main"])
ORIENTATION_MODELS = tuple(str(value) for value in model_config["orientation"])
EXTREME_ROBUSTNESS_MODELS = tuple(
    str(value) for value in model_config["extreme_robustness"]
)
RESIDUAL_COVARIANCE_MODELS = tuple(
    str(value) for value in model_config["residual_covariance"]
)
PREDICTION_MODELS = tuple(str(value) for value in model_config["prediction"])
STABILITY_CUTOFFS = tuple(
    str(value) for value in analysis_config["stability_cutoffs"]
)
STABILITY_ASSET = str(analysis_config["stability_asset"])
PREDICTION_ASSETS = tuple(
    str(value) for value in analysis_config["prediction_assets"]
)
PARAMETER_GRIDS = parameter_grid_overrides(config)
data = load_or_build_processed_data(paths)
result = run_factor_experiment(
    data.factor_frames,
    assets=ASSETS,
    features=FACTOR_COLUMNS,
    quick=quick,
    clipping=clipping,
    evaluation_structure=evaluation_structure,
    model_names=MAIN_MODELS,
    random_state=int(execution_config["random_state"]),
    parameter_grids=PARAMETER_GRIDS,
    verbose=verbose,
)
artifacts: dict[str, Path] = {}

capacity_shapley_rows: list[dict[str, object]] = []
capacity_interaction_rows: list[pd.DataFrame] = []
capacity_matrices: dict[tuple[str, str], pd.DataFrame] = {}
for (asset, model), fitted in result.fitted_models.items():
    if not str(model).startswith("Choquet"):
        continue
    values, matrix = capacity_summary(fitted, list(FACTOR_COLUMNS))
    capacity_matrices[(asset, model)] = matrix
    capacity_shapley_rows.extend(
        {
            "asset": asset,
            "model": model,
            "feature": feature,
            "Shapley importance": float(value),
        }
        for feature, value in values.items()
    )
    interactions = top_interactions(
        matrix,
        n=len(FACTOR_COLUMNS) * (len(FACTOR_COLUMNS) - 1) // 2,
    )
    interactions.insert(0, "model", model)
    interactions.insert(0, "asset", asset)
    capacity_interaction_rows.append(interactions)

capacity_shapley_panel = (
    pd.DataFrame(capacity_shapley_rows)
    .set_index(["asset", "model", "feature"])
    .sort_index()
    if capacity_shapley_rows
    else pd.DataFrame()
)
capacity_interaction_panel = (
    pd.concat(capacity_interaction_rows, ignore_index=True)
    .set_index(["asset", "model", "first", "second"])
    .sort_index()
    if capacity_interaction_rows
    else pd.DataFrame()
)

residual_rows: list[dict[str, float | str]] = []
for model, residuals in result.residuals.items():
    covariance = residuals.cov()
    off_diagonal = covariance.to_numpy()[
        ~np.eye(len(covariance), dtype=bool)
    ]
    residual_rows.append(
        {
            "model": model,
            "trace": float(np.trace(covariance)),
            "Frobenius norm": float(np.linalg.norm(covariance.to_numpy())),
            "mean absolute off-diagonal": float(np.mean(np.abs(off_diagonal))),
        }
    )
residual_summary = (
    pd.DataFrame(residual_rows)
    .set_index("model")
    .sort_values("Frobenius norm")
)
in_sample_residual_rows: list[dict[str, float | str]] = []
for model, residuals in result.in_sample_residuals.items():
    covariance = residuals.cov()
    off_diagonal = covariance.to_numpy()[
        ~np.eye(len(covariance), dtype=bool)
    ]
    in_sample_residual_rows.append(
        {
            "model": model,
            "trace": float(np.trace(covariance)),
            "Frobenius norm": float(np.linalg.norm(covariance.to_numpy())),
            "mean absolute off-diagonal": float(
                np.mean(np.abs(off_diagonal))
            ),
        }
    )
in_sample_residual_summary = (
    pd.DataFrame(in_sample_residual_rows)
    .set_index("model")
    .sort_values("Frobenius norm")
)
aggregate = result.metrics.groupby("model")[[
    "RMSE",
    "MAE",
    "OOS R2",
    "Correlation",
    "parameter count",
    "refit runtime seconds",
]].agg(["mean", "std"])
aggregate.columns = [f"{metric} {statistic}" for metric, statistic in aggregate.columns]
mean_metrics = result.metrics.groupby("model")[["RMSE", "MAE", "OOS R2"]].mean()
asset_rmse = result.metrics["RMSE"].unstack("model")
selected_choquet = select_model_family_by_validation(
    result.metrics,
    family_prefix="Choquet",
    score_column="validation RMSE",
)
orientation_ablation = result.metrics[
    result.metrics.index.get_level_values("model").isin(ORIENTATION_MODELS)
]
stress_metric_panels: list[pd.DataFrame] = []
stress_audits: list[pd.Series] = []
clipping_panels: list[pd.DataFrame] = []
estimation_robustness_panels: list[pd.DataFrame] = []
validation_stress_panels: list[pd.DataFrame] = []
validation_stress_summaries: list[pd.DataFrame] = []
validation_stress_samples: list[pd.DataFrame] = []
validation_stress_failures: list[pd.DataFrame] = []
for asset, split in result.splits.items():
    X_fit = pd.concat([split.X_train, split.X_validation])
    y_fit = pd.concat([split.y_train, split.y_validation])
    stress = fit_empirical_stress_definition(X_fit, y_fit)
    fit_extreme_mask = stress.mask(X_fit, y_fit)
    test_stress_mask = stress.mask(split.X_test, split.y_test)

    for sample, X_sample, y_sample in (
        ("estimation", X_fit, y_fit),
        ("test", split.X_test, split.y_test),
    ):
        audit = stress.audit(X_sample, y_sample, sample=sample)
        audit = audit.drop(labels="sample")
        audit.name = (asset, sample)
        stress_audits.append(audit)

    asset_predictions = pd.DataFrame(
        {
            model: predictions[asset]
            for model, predictions in result.predictions.items()
            if asset in predictions
        }
    ).reindex(split.X_test.index)
    stress_metrics = regression_regime_metrics(
        split.y_test,
        asset_predictions,
        test_stress_mask,
    )
    stress_metrics["asset"] = asset
    stress_metric_panels.append(stress_metrics.reset_index())

    clipping_reference = result.fitted_models.get((asset, "OLS"))
    if clipping_reference is not None:
        for sample, X_sample in (
            ("estimation", X_fit),
            ("test", split.X_test),
        ):
            clipping_audit = clipping_diagnostics(
                clipping_reference,
                X_sample,
                sample=sample,
            )
            clipping_audit["asset"] = asset
            clipping_panels.append(
                clipping_audit.reset_index(names="feature")
            )

    robustness_models = {
        model: result.fitted_models[(asset, model)]
        for model in EXTREME_ROBUSTNESS_MODELS
        if (asset, model) in result.fitted_models
    }
    robustness = regression_estimation_robustness(
        robustness_models,
        X_fit,
        y_fit,
        split.X_test,
        split.y_test,
        asset_predictions,
        fit_extreme_mask=fit_extreme_mask,
        test_stress_mask=test_stress_mask,
    )
    robustness["asset"] = asset
    estimation_robustness_panels.append(robustness.reset_index())

    validation_comparison = compare_validation_stress_regimes(
        data.factor_frames[asset][list(FACTOR_COLUMNS)],
        data.factor_frames[asset]["target_excess_loss"],
        model_names=tuple(
            result.metrics.xs(asset, level="asset").index.astype(str)
        ),
        quick=quick,
        clipping=clipping,
        random_state=int(execution_config["random_state"]),
        validation_start=str(stress_config["validation_start"]),
        validation_end=str(stress_config["validation_end"]),
        stress_start=str(stress_config["stress_start"]),
        stress_end=str(stress_config["stress_end"]),
        parameter_grids=PARAMETER_GRIDS,
    )
    validation_metrics = validation_comparison.metrics.reset_index()
    validation_metrics["asset"] = asset
    validation_stress_panels.append(validation_metrics)
    validation_summary = validation_comparison.selection_summary.reset_index()
    validation_summary["asset"] = asset
    validation_stress_summaries.append(validation_summary)
    validation_samples = validation_comparison.sample_summary.reset_index()
    validation_samples["asset"] = asset
    validation_stress_samples.append(validation_samples)
    if not validation_comparison.failures.empty:
        validation_failures = validation_comparison.failures.copy()
        validation_failures["asset"] = asset
        validation_stress_failures.append(validation_failures)

factor_stress_metrics = pd.concat(
    stress_metric_panels,
    ignore_index=True,
).set_index(["asset", "model"])
factor_stress_audit = pd.DataFrame(stress_audits)
factor_stress_audit.index = pd.MultiIndex.from_tuples(
    factor_stress_audit.index,
    names=["asset", "sample"],
)
factor_clipping_audit = pd.concat(
    clipping_panels,
    ignore_index=True,
).set_index(["asset", "sample", "feature"])
factor_estimation_robustness = pd.concat(
    estimation_robustness_panels,
    ignore_index=True,
).set_index(["asset", "model"])
factor_validation_stress = pd.concat(
    validation_stress_panels,
    ignore_index=True,
).set_index(["asset", "regime", "model"])
factor_validation_stress_samples = pd.concat(
    validation_stress_samples,
    ignore_index=True,
).set_index(["asset", "sample"])
factor_validation_stress_summary = pd.concat(
    validation_stress_summaries,
    ignore_index=True,
).set_index(["asset", "regime"])
factor_validation_stress_failures = (
    pd.concat(validation_stress_failures, ignore_index=True)
    if validation_stress_failures
    else pd.DataFrame(columns=["regime", "model", "message", "asset"])
)

artifacts["asset-model metrics"] = save_table(
    result.metrics, "experiment_1_asset_model_metrics.csv", paths
)
artifacts["in-sample asset-model metrics"] = save_table(
    result.in_sample_metrics,
    "experiment_1_in_sample_asset_model_metrics.csv",
    paths,
)
artifacts["aggregate metrics"] = save_table(
    aggregate, "experiment_1_aggregate_metrics.csv", paths
)
artifacts["asset RMSE"] = save_table(
    asset_rmse, "experiment_1_asset_rmse.csv", paths
)
artifacts["residual covariance summary"] = save_table(
    residual_summary,
    "experiment_1_residual_covariance_summary.csv",
    paths,
)
artifacts["in-sample residual covariance summary"] = save_table(
    in_sample_residual_summary,
    "experiment_1_in_sample_residual_covariance_summary.csv",
    paths,
)
artifacts["evaluation folds"] = save_table(
    pd.concat(result.fold_summaries, names=["asset", "fold"]),
    "experiment_1_evaluation_folds.csv",
    paths,
)
artifacts["fold metrics"] = save_table(
    result.fold_metrics,
    "experiment_1_evaluation_fold_metrics.csv",
    paths,
)
artifacts["selected parameters"] = save_table(
    result.selected_parameters,
    "experiment_1_selected_parameters.csv",
    paths,
)
artifacts["validation-selected Choquet"] = save_table(
    selected_choquet,
    "experiment_1_validation_selected_choquet.csv",
    paths,
)
artifacts["orientation ablation"] = save_table(
    orientation_ablation,
    "experiment_1_orientation_ablation.csv",
    paths,
)
artifacts["all final orientations"] = save_table(
    orientation_tables(
        result.fitted_models,
        key_names=("asset", "model"),
    ),
    "experiment_1_orientations.csv",
    paths,
)
artifacts["empirical stress audit"] = save_table(
    factor_stress_audit,
    "experiment_1_empirical_stress_audit.csv",
    paths,
)
artifacts["empirical stress metrics"] = save_table(
    factor_stress_metrics,
    "experiment_1_empirical_stress_metrics.csv",
    paths,
)
artifacts["clipping audit"] = save_table(
    factor_clipping_audit,
    "experiment_1_clipping_audit.csv",
    paths,
)
artifacts["extreme-estimation robustness"] = save_table(
    factor_estimation_robustness,
    "experiment_1_extreme_estimation_robustness.csv",
    paths,
)
artifacts["validation-stress comparison"] = save_table(
    factor_validation_stress,
    "experiment_1_validation_stress_comparison.csv",
    paths,
)
artifacts["validation-stress samples"] = save_table(
    factor_validation_stress_samples,
    "experiment_1_validation_stress_samples.csv",
    paths,
)
artifacts["validation-stress selection summary"] = save_table(
    factor_validation_stress_summary,
    "experiment_1_validation_stress_selection_summary.csv",
    paths,
)
artifacts["validation-stress failures"] = save_table(
    factor_validation_stress_failures,
    "experiment_1_validation_stress_failures.csv",
    paths,
    index=False,
)
artifacts["failures"] = save_table(
    result.failures, "experiment_1_failures.csv", paths, index=False
)
artifacts["all capacity Shapley indices"] = save_table(
    capacity_shapley_panel,
    "experiment_1_all_capacity_shapley.csv",
    paths,
)
artifacts["all capacity interactions"] = save_table(
    capacity_interaction_panel,
    "experiment_1_all_capacity_interactions.csv",
    paths,
)

for model, predictions in result.predictions.items():
    slug = artifact_slug(model)
    artifacts[f"predictions {model}"] = save_table(
        predictions.assign(evaluation_structure=evaluation_structure), f"experiment_1_predictions_{slug}.parquet", paths
    )
for model, residuals in result.residuals.items():
    slug = artifact_slug(model)
    artifacts[f"residuals {model}"] = save_table(
        residuals, f"experiment_1_residuals_{slug}.parquet", paths
    )
for model, residuals in result.in_sample_residuals.items():
    slug = artifact_slug(model)
    artifacts[f"in-sample residuals {model}"] = save_table(
        residuals,
        f"experiment_1_in_sample_residuals_{slug}.parquet",
        paths,
    )
    artifacts[f"in-sample residual covariance {model}"] = save_table(
        residuals.cov(),
        f"experiment_1_in_sample_residual_covariance_{slug}.csv",
        paths,
    )

shapley_panel = pd.DataFrame()
if result.shapley:
    shapley_panel = pd.concat(result.shapley, axis=1).T
    artifacts["Shapley indices"] = save_table(
        shapley_panel, "experiment_1_shapley_indices.csv", paths
    )
aapl_frame = data.factor_frames[STABILITY_ASSET]
reference = str(analysis_config["representative_model"])
stability = expanding_capacity_stability(
    aapl_frame[list(FACTOR_COLUMNS)],
    aapl_frame["target_excess_loss"],
    cutoffs=STABILITY_CUTOFFS,
    task="regression",
    model_name=reference,
    purge=int(analysis_config["stability_purge"]),
    clipping=clipping,
    verbose=verbose,
)
artifacts["AAPL Shapley stability"] = save_table(
    stability.shapley,
    "experiment_1_aapl_shapley_stability.csv",
    paths,
)
artifacts["AAPL stability failures"] = save_table(
    stability.failures,
    "experiment_1_aapl_stability_failures.csv",
    paths,
    index=False,
)

if reference in result.predictions:
    aapl_predictions = pd.concat(
        {
            model: frame[STABILITY_ASSET]
            for model, frame in result.predictions.items()
            if STABILITY_ASSET in frame
        },
        axis=1,
    ).dropna()
    aapl_actual = aapl_frame.loc[
        aapl_predictions.index, "target_excess_loss"
    ]
    artifacts["AAPL HAC comparison"] = save_table(
        hac_model_comparison(
            aapl_actual,
            aapl_predictions,
            reference=reference,
            loss="squared",
            max_lags=int(analysis_config["hac_max_lags"]),
        ),
        "experiment_1_aapl_hac_comparison.csv",
        paths,
    )
chosen = result.fitted_models.get((STABILITY_ASSET, reference))
if chosen is not None:
    artifacts["AAPL orientation"] = save_table(
        orientation_table(chosen),
        "experiment_1_aapl_orientation.csv",
        paths,
    )

axis = plot_matrix(
    aapl_frame[list(FACTOR_COLUMNS)].corr(),
    title="Correlation of ETF factor proxies",
    symmetric=True,
)
artifacts["factor correlation figure"] = save_figure(
    axis.figure, "experiment_1_factor_correlation.png", paths
)

axis = plot_metric_ranking(mean_metrics, "RMSE")
artifacts["mean RMSE figure"] = save_figure(
    axis.figure, "experiment_1_mean_rmse.png", paths
)

figure, axis = plt.subplots(figsize=(12, 5))
image = axis.imshow(asset_rmse.to_numpy(), aspect="auto")
axis.set_xticks(range(asset_rmse.shape[1]), asset_rmse.columns, rotation=90)
axis.set_yticks(range(asset_rmse.shape[0]), asset_rmse.index)
axis.set_title("Test RMSE by asset and model")
figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
figure.tight_layout()
artifacts["asset RMSE figure"] = save_figure(
    figure, "experiment_1_asset_rmse.png", paths
)

for model in (
    "Choquet 2-additive",
    "Choquet 2-additive L1",
    "Choquet 2-additive scaled-q",
    "Choquet 2-additive scaled-q L1",
):
    matrix = capacity_matrices.get((STABILITY_ASSET, model))
    if matrix is None:
        continue
    axis = plot_matrix(
        matrix,
        title=f"{STABILITY_ASSET}: pairwise interactions — {model}",
        symmetric=True,
    )
    artifacts[f"{STABILITY_ASSET} interactions {model}"] = save_figure(
        axis.figure,
        f"experiment_1_{artifact_slug(STABILITY_ASSET)}_interactions_"
        f"{artifact_slug(model)}.png",
        paths,
    )

for model in RESIDUAL_COVARIANCE_MODELS:
    if model in result.residuals:
        axis = plot_matrix(
            result.residual_covariance(model),
            title=f"Residual covariance — {model}",
            symmetric=True,
        )
        artifacts[f"residual covariance figure {model}"] = save_figure(
            axis.figure,
            f"experiment_1_residual_covariance_{artifact_slug(model)}.png",
            paths,
        )

for asset in PREDICTION_ASSETS:
    available = [
        model
        for model in PREDICTION_MODELS
        if model in result.predictions and asset in result.predictions[model]
    ]
    if available:
        predictions = pd.concat(
            {model: result.predictions[model][asset] for model in available},
            axis=1,
        )
        actual = data.factor_frames[asset].loc[
            predictions.index, "target_excess_loss"
        ]
        axis = plot_actual_predictions(
            actual,
            predictions,
            title=f"{asset}: out-of-sample excess-loss fit",
            start="2020-01-01",
        )
        artifacts[f"prediction figure {asset}"] = save_figure(
            axis.figure,
            f"experiment_1_predictions_{artifact_slug(asset)}.png",
            paths,
        )

if not shapley_panel.empty:
    axis = plot_shapley(
        shapley_panel.mean().sort_values(ascending=False),
        title="Average Choquet Shapley importance across assets",
    )
    artifacts["average Shapley figure"] = save_figure(
        axis.figure, "experiment_1_average_shapley.png", paths
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    image = axis.imshow(shapley_panel.to_numpy(), aspect="auto")
    axis.set_xticks(
        range(shapley_panel.shape[1]), shapley_panel.columns, rotation=90
    )
    axis.set_yticks(range(shapley_panel.shape[0]), shapley_panel.index)
    axis.set_title("Shapley importance by asset")
    figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    figure.tight_layout()
    artifacts["Shapley panel figure"] = save_figure(
        figure, "experiment_1_shapley_by_asset.png", paths
    )

axis = stability.shapley.plot(
    figsize=(11, 5), marker="o", title="AAPL factor Shapley stability"
)
axis.set_ylabel("Shapley importance")
axis.grid(alpha=0.25)
axis.figure.tight_layout()
artifacts["AAPL stability figure"] = save_figure(
    axis.figure, "experiment_1_aapl_shapley_stability.png", paths
)
print_artifacts(artifacts)
