"""Run and persist the complete tail-risk classification experiment."""

from __future__ import annotations

# The full pipeline intentionally lives in this executable script.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from mwc_experiments.configuration import (
    load_experiment_config,
    parameter_grid_overrides,
)
from mwc_experiments.data import load_or_build_processed_data
from mwc_experiments.evaluation import (
    orientation_table,
    orientation_tables,
    plot_classifier_discrimination,
    plot_matrix,
    plot_metric_ranking,
    plot_probability_calibration,
    plot_shapley,
    top_interactions,
)
from mwc_experiments.workflows import (
    expanding_capacity_stability,
    run_tail_classification_experiment,
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
    "Run the complete tail-risk experiment.",
    experiment_id="tail_risk",
)
quick = args.quick
verbose = not args.quiet
paths = prepare_output_paths()
config = load_experiment_config("tail_risk", paths.root)
dataset_config = config["dataset"]
model_config = config["models"]
analysis_config = config["analysis"]
walk_forward_config = config["walk_forward"]
calibration_config = config["calibration"]
class_weight_config = config["class_weight"]
aggregation_config = config["aggregation"]
execution_config = config["execution"]
HORIZONS = tuple(int(value) for value in dataset_config["horizons"])
MAIN_RISK_FEATURES = tuple(str(value) for value in dataset_config["features"])
PRIMARY_TAIL_ALPHA = float(dataset_config["primary_alpha"])
ROBUSTNESS_TAIL_ALPHA = float(dataset_config["robustness_alpha"])
MAIN_MODELS = tuple(str(value) for value in model_config["main"])
ORIENTATION_MODELS = tuple(str(value) for value in model_config["orientation"])
ROBUSTNESS_MODELS = tuple(str(value) for value in model_config["robustness"])
REPORT_MODELS = tuple(str(value) for value in model_config["report"])
CALIBRATION_METHODS = tuple(
    str(value) for value in calibration_config["methods"]
)
CLASS_WEIGHT_MODES = tuple(
    str(value) for value in class_weight_config["modes"]
)
AGGREGATION_MODEL = str(aggregation_config["model_name"])
AGGREGATION_BASE_MODELS = tuple(
    str(value) for value in aggregation_config["base_models"]
)
ROBUSTNESS_AGGREGATION_BASE_MODELS = tuple(
    str(value) for value in aggregation_config["robustness_base_models"]
)
STABILITY_CUTOFFS = tuple(
    str(value) for value in analysis_config["stability_cutoffs"]
)
STABILITY_HORIZON = int(analysis_config["stability_horizon"])
PARAMETER_GRIDS = parameter_grid_overrides(config)
dataset = load_or_build_processed_data(paths).equal_weight_dataset
primary = run_tail_classification_experiment(
    dataset,
    features=MAIN_RISK_FEATURES,
    horizons=HORIZONS,
    alpha=PRIMARY_TAIL_ALPHA,
    quick=quick,
    model_names=MAIN_MODELS,
    random_state=int(execution_config["random_state"]),
    oos_start=str(walk_forward_config["oos_start"]),
    training_window_years=int(
        walk_forward_config["training_window_years"]
    ),
    selection_window_months=int(
        walk_forward_config["selection_window_months"]
    ),
    calibration_window_months=int(
        walk_forward_config["calibration_window_months"]
    ),
    oos_block_years=int(walk_forward_config["oos_block_years"]),
    calibration_methods=CALIBRATION_METHODS,
    class_weight_modes=CLASS_WEIGHT_MODES,
    aggregation_model_name=AGGREGATION_MODEL,
    aggregation_base_models=AGGREGATION_BASE_MODELS,
    parameter_grids=PARAMETER_GRIDS,
    verbose=verbose,
)
rare = run_tail_classification_experiment(
    dataset,
    features=MAIN_RISK_FEATURES,
    horizons=HORIZONS,
    alpha=ROBUSTNESS_TAIL_ALPHA,
    quick=quick,
    model_names=ROBUSTNESS_MODELS,
    random_state=int(execution_config["random_state"]),
    oos_start=str(walk_forward_config["oos_start"]),
    training_window_years=int(
        walk_forward_config["training_window_years"]
    ),
    selection_window_months=int(
        walk_forward_config["selection_window_months"]
    ),
    calibration_window_months=int(
        walk_forward_config["calibration_window_months"]
    ),
    oos_block_years=int(walk_forward_config["oos_block_years"]),
    calibration_methods=CALIBRATION_METHODS,
    class_weight_modes=CLASS_WEIGHT_MODES,
    aggregation_model_name=AGGREGATION_MODEL,
    aggregation_base_models=ROBUSTNESS_AGGREGATION_BASE_MODELS,
    parameter_grids=PARAMETER_GRIDS,
    verbose=verbose,
)
stability = expanding_capacity_stability(
    dataset[list(MAIN_RISK_FEATURES)],
    dataset[
        f"tail_event_h{STABILITY_HORIZON}_a"
        f"{str(PRIMARY_TAIL_ALPHA).replace('.', 'p')}"
    ].astype(float),
    cutoffs=STABILITY_CUTOFFS,
    task="classification",
    purge=int(analysis_config["stability_purge"]),
    verbose=verbose,
)
artifacts: dict[str, Path] = {}

audit_rows: list[dict[str, float | int]] = []
for alpha in (PRIMARY_TAIL_ALPHA, ROBUSTNESS_TAIL_ALPHA):
    label = str(alpha).replace(".", "p")
    for horizon in HORIZONS:
        valid = dataset[f"tail_event_h{horizon}_a{label}"].dropna().astype(int)
        audit_rows.append(
            {
                "alpha": alpha,
                "horizon": horizon,
                "observations": len(valid),
                "events": int(valid.sum()),
                "event rate": float(valid.mean()),
            }
        )
artifacts["tail-label audit"] = save_table(
    pd.DataFrame(audit_rows).set_index(["alpha", "horizon"]),
    "experiment_2b_tail_label_audit.csv",
    paths,
)

for horizon, horizon_result in primary.items():
    artifacts[f"metrics h{horizon}"] = save_table(
        horizon_result.metrics,
        f"experiment_2b_classification_metrics_h{horizon}_a095.csv",
        paths,
    )
    artifacts[f"discrimination h{horizon}"] = save_table(
        horizon_result.discrimination_metrics,
        f"experiment_2b_discrimination_h{horizon}_a095.csv",
        paths,
    )
    artifacts[f"calibration h{horizon}"] = save_table(
        horizon_result.calibration_metrics,
        f"experiment_2b_calibration_h{horizon}_a095.csv",
        paths,
    )
    artifacts[f"calibration sample h{horizon}"] = save_table(
        horizon_result.calibration_sample_summary,
        f"experiment_2b_calibration_sample_h{horizon}_a095.csv",
        paths,
    )
    artifacts[f"probabilities h{horizon}"] = save_table(
        horizon_result.probabilities,
        f"experiment_2b_probabilities_h{horizon}_a095.parquet",
        paths,
    )
    artifacts[f"thresholds h{horizon}"] = save_table(
        horizon_result.thresholds,
        f"experiment_2b_thresholds_h{horizon}_a095.csv",
        paths,
    )
    artifacts[f"selected parameters h{horizon}"] = save_table(
        horizon_result.selected_parameters,
        f"experiment_2b_selected_parameters_h{horizon}_a095.csv",
        paths,
    )
    artifacts[f"walk-forward folds h{horizon}"] = save_table(
        horizon_result.fold_summary,
        f"experiment_2b_walk_forward_folds_h{horizon}_a095.csv",
        paths,
    )
    artifacts[f"walk-forward metrics h{horizon}"] = save_table(
        horizon_result.fold_metrics,
        f"experiment_2b_walk_forward_metrics_h{horizon}_a095.csv",
        paths,
    )
    artifacts[f"orientation history h{horizon}"] = save_table(
        horizon_result.orientation_history,
        f"experiment_2b_orientation_history_h{horizon}_a095.csv",
        paths,
    )
    artifacts[f"Shapley history h{horizon}"] = save_table(
        horizon_result.shapley_history,
        f"experiment_2b_shapley_history_h{horizon}_a095.csv",
        paths,
    )
    artifacts[f"orientation ablation h{horizon}"] = save_table(
        horizon_result.metrics[
            horizon_result.metrics["base model"].isin(ORIENTATION_MODELS)
            & horizon_result.metrics["probability calibration"].eq(
                "uncalibrated"
            )
        ],
        f"experiment_2b_orientation_ablation_h{horizon}_a095.csv",
        paths,
    )
    artifacts[f"final orientations h{horizon}"] = save_table(
        orientation_tables(
            horizon_result.fitted_models,
            key_names=("model",),
        ),
        f"experiment_2b_orientations_h{horizon}_a095.csv",
        paths,
    )
    artifacts[f"failures h{horizon}"] = save_table(
        horizon_result.failures,
        f"experiment_2b_failures_h{horizon}_a095.csv",
        paths,
        index=False,
    )
    for model, shapley in horizon_result.shapley.items():
        base_model = str(horizon_result.metrics.loc[model, "base model"])
        weight_mode = str(horizon_result.metrics.loc[model, "class weight"])
        weight_suffix = "" if weight_mode == "balanced" else "_unweighted"
        table_slug = base_model.lower().replace(" ", "_") + weight_suffix
        figure_slug = artifact_slug(base_model) + weight_suffix
        artifacts[f"Shapley h{horizon} {model}"] = save_table(
            shapley,
            f"experiment_2b_shapley_h{horizon}_{table_slug}.csv",
            paths,
        )
        artifacts[f"interactions h{horizon} {model}"] = save_table(
            horizon_result.interactions[model],
            f"experiment_2b_interactions_h{horizon}_{table_slug}.csv",
            paths,
        )
        artifacts[f"top interactions h{horizon} {model}"] = save_table(
            top_interactions(
                horizon_result.interactions[model],
                n=int(analysis_config["top_interactions"]),
            ),
            f"experiment_2b_top_interactions_h{horizon}_{figure_slug}.csv",
            paths,
            index=False,
        )

    axis = plot_metric_ranking(
        horizon_result.metrics,
        "PR AUC",
        lower_is_better=False,
    )
    artifacts[f"PR AUC ranking figure h{horizon}"] = save_figure(
        axis.figure,
        f"experiment_2b_pr_auc_ranking_h{horizon}_a095.png",
        paths,
    )
    axis = plot_metric_ranking(
        horizon_result.calibration_metrics,
        "Brier",
        lower_is_better=True,
    )
    artifacts[f"Brier ranking figure h{horizon}"] = save_figure(
        axis.figure,
        f"experiment_2b_brier_ranking_h{horizon}_a095.png",
        paths,
    )
    preferred = horizon_result.metrics.index[
        horizon_result.metrics["base model"].isin(
            (*REPORT_MODELS, AGGREGATION_MODEL)
        )
        & horizon_result.metrics["probability calibration"].eq(
            "uncalibrated"
        )
    ].tolist()
    axes = plot_classifier_discrimination(
        horizon_result.split.y_test,
        horizon_result.probabilities,
        models=preferred,
        legend_outside=True,
    )
    axes[0].figure.suptitle(
        f"Classifier discrimination — h={horizon}", y=1.03
    )
    artifacts[f"classifier discrimination h{horizon}"] = save_figure(
        axes[0].figure,
        f"experiment_2b_classifier_discrimination_h{horizon}_a095.png",
        paths,
    )
    calibration_bases = (
        "Logistic",
        "Gradient boosting",
        "MLP",
        "Fuzzy Choquet neural network",
        "Choquistic 2-additive",
        "Choquet linear classifier",
        AGGREGATION_MODEL,
    )
    calibration_models = horizon_result.metrics.index[
        horizon_result.metrics["base model"].isin(calibration_bases)
    ].tolist()
    axis = plot_probability_calibration(
        horizon_result.split.y_test,
        horizon_result.probabilities,
        models=calibration_models,
        legend_outside=True,
    )
    axis.figure.suptitle(
        f"Uncalibrated vs calibrated probabilities — h={horizon}", y=1.03
    )
    artifacts[f"calibration comparison h{horizon}"] = save_figure(
        axis.figure,
        f"experiment_2b_calibration_comparison_h{horizon}_a095.png",
        paths,
    )

    probability_families = (
        "Logistic",
        "Gradient boosting",
        "Choquistic 2-additive",
        "Choquet linear classifier",
        AGGREGATION_MODEL,
    )
    probability_models = horizon_result.metrics.index[
        horizon_result.metrics["base model"].isin(probability_families)
        & horizon_result.metrics["probability calibration"].eq(
            "uncalibrated"
        )
    ].tolist()
    figure, axis = plt.subplots(figsize=(13, 4))
    horizon_result.probabilities[probability_models].plot(
        ax=axis, linewidth=0.9
    )
    events = horizon_result.split.y_test.astype(bool)
    axis.scatter(
        events.index[events],
        np.repeat(1.02, int(events.sum())),
        marker="|",
        s=70,
        label="realized tail event",
    )
    axis.set_ylim(-0.02, 1.08)
    axis.set_title(
        f"Predicted tail probabilities and realized events — h={horizon}"
    )
    axis.set_ylabel("Probability")
    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
    )
    axis.grid(alpha=0.2)
    figure.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    artifacts[f"probability paths figure h{horizon}"] = save_figure(
        figure,
        f"experiment_2b_probability_paths_h{horizon}_a095.png",
        paths,
    )

    confusion_models = probability_models
    figure, axes = plt.subplots(
        1, len(confusion_models), figsize=(5 * len(confusion_models), 4)
    )
    axes = np.atleast_1d(axes)
    for axis, model in zip(axes, confusion_models):
        predicted = (
            horizon_result.probabilities[model]
            >= horizon_result.thresholds[model]
        ).astype(int)
        ConfusionMatrixDisplay.from_predictions(
            horizon_result.split.y_test,
            predicted,
            display_labels=["non-tail", "tail"],
            ax=axis,
            colorbar=False,
        )
        axis.set_title(f"{model} — h={horizon}")
    figure.tight_layout()
    artifacts[f"confusion matrices h{horizon}"] = save_figure(
        figure,
        f"experiment_2b_confusion_matrices_h{horizon}_a095.png",
        paths,
    )

    for model, shapley in horizon_result.shapley.items():
        base_model = str(horizon_result.metrics.loc[model, "base model"])
        weight_mode = str(horizon_result.metrics.loc[model, "class weight"])
        weight_suffix = "" if weight_mode == "balanced" else "_unweighted"
        slug = artifact_slug(base_model) + weight_suffix
        axis = plot_shapley(
            shapley,
            title=f"{model}: Shapley importance, h={horizon}",
        )
        artifacts[f"Shapley figure h{horizon} {model}"] = save_figure(
            axis.figure,
            f"experiment_2b_shapley_h{horizon}_{slug}.png",
            paths,
        )
        axis = plot_matrix(
            horizon_result.interactions[model],
            title=f"{model}: pairwise interaction indices, h={horizon}",
        )
        artifacts[f"interaction figure h{horizon} {model}"] = save_figure(
            axis.figure,
            f"experiment_2b_interactions_h{horizon}_{slug}.png",
            paths,
        )

for horizon, horizon_result in rare.items():
    artifacts[f"rare-event metrics h{horizon}"] = save_table(
        horizon_result.metrics,
        f"experiment_2b_classification_metrics_h{horizon}_a0975.csv",
        paths,
    )
    artifacts[f"rare-event discrimination h{horizon}"] = save_table(
        horizon_result.discrimination_metrics,
        f"experiment_2b_discrimination_h{horizon}_a0975.csv",
        paths,
    )
    artifacts[f"rare-event calibration h{horizon}"] = save_table(
        horizon_result.calibration_metrics,
        f"experiment_2b_calibration_h{horizon}_a0975.csv",
        paths,
    )
    artifacts[f"rare-event calibration sample h{horizon}"] = save_table(
        horizon_result.calibration_sample_summary,
        f"experiment_2b_calibration_sample_h{horizon}_a0975.csv",
        paths,
    )
    artifacts[f"rare-event probabilities h{horizon}"] = save_table(
        horizon_result.probabilities,
        f"experiment_2b_probabilities_h{horizon}_a0975.parquet",
        paths,
    )
    artifacts[f"rare-event thresholds h{horizon}"] = save_table(
        horizon_result.thresholds,
        f"experiment_2b_thresholds_h{horizon}_a0975.csv",
        paths,
    )
    artifacts[f"rare-event walk-forward folds h{horizon}"] = save_table(
        horizon_result.fold_summary,
        f"experiment_2b_walk_forward_folds_h{horizon}_a0975.csv",
        paths,
    )
    artifacts[f"rare-event walk-forward metrics h{horizon}"] = save_table(
        horizon_result.fold_metrics,
        f"experiment_2b_walk_forward_metrics_h{horizon}_a0975.csv",
        paths,
    )
    artifacts[f"rare-event orientation history h{horizon}"] = save_table(
        horizon_result.orientation_history,
        f"experiment_2b_orientation_history_h{horizon}_a0975.csv",
        paths,
    )
    artifacts[f"rare-event Shapley history h{horizon}"] = save_table(
        horizon_result.shapley_history,
        f"experiment_2b_shapley_history_h{horizon}_a0975.csv",
        paths,
    )
    artifacts[f"rare-event orientation ablation h{horizon}"] = save_table(
        horizon_result.metrics[
            horizon_result.metrics["base model"].isin(ORIENTATION_MODELS)
            & horizon_result.metrics["probability calibration"].eq(
                "uncalibrated"
            )
        ],
        f"experiment_2b_orientation_ablation_h{horizon}_a0975.csv",
        paths,
    )
    artifacts[f"rare-event final orientations h{horizon}"] = save_table(
        orientation_tables(
            horizon_result.fitted_models,
            key_names=("model",),
        ),
        f"experiment_2b_orientations_h{horizon}_a0975.csv",
        paths,
    )
    artifacts[f"rare-event failures h{horizon}"] = save_table(
        horizon_result.failures,
        f"experiment_2b_failures_h{horizon}_a0975.csv",
        paths,
        index=False,
    )

artifacts["Shapley stability"] = save_table(
    stability.shapley,
    "experiment_2b_shapley_stability_h5_a095.csv",
    paths,
)
artifacts["interaction stability"] = save_table(
    stability.interaction_long,
    "experiment_2b_interaction_stability_h5_a095_long.csv",
    paths,
    index=False,
)
artifacts["interaction stability summary"] = save_table(
    stability.interaction_stability(),
    "experiment_2b_interaction_stability_h5_a095_summary.csv",
    paths,
)
artifacts["stability failures"] = save_table(
    stability.failures,
    "experiment_2b_stability_failures_h5_a095.csv",
    paths,
    index=False,
)
representative = primary[5].fitted_models.get(
    "Choquistic 2-additive [balanced]"
)
if representative is None:
    representative = primary[5].fitted_models.get("Choquistic 2-additive")
if representative is not None:
    artifacts["orientation"] = save_table(
        orientation_table(representative),
        "experiment_2b_orientation_h5_choquistic_2-additive.csv",
        paths,
    )

figure, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
for axis, horizon in zip(axes, HORIZONS):
    actual = dataset[f"future_loss_h{horizon}"]
    threshold = dataset[f"historical_var_h{horizon}_a0p95"]
    actual.plot(ax=axis, linewidth=0.7, label="future loss")
    threshold.plot(
        ax=axis, linewidth=1.0, label="historical VaR threshold"
    )
    axis.set_title(
        f"Point-in-time tail threshold, h={horizon}, alpha=95%"
    )
    axis.grid(alpha=0.2)
handles, labels = axes[0].get_legend_handles_labels()
figure.legend(
    handles,
    labels,
    loc="center left",
    bbox_to_anchor=(0.84, 0.5),
)
figure.tight_layout(rect=(0.0, 0.0, 0.83, 1.0))
artifacts["tail threshold figure"] = save_figure(
    figure,
    "experiment_2b_tail_thresholds_a095.png",
    paths,
)

axis = stability.shapley.plot(
    figsize=(12, 5),
    marker="o",
    title="5-day tail event: expanding-window Shapley stability",
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
    "experiment_2b_shapley_stability_h5_a095.png",
    paths,
)
print_artifacts(artifacts)
