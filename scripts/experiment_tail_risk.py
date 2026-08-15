"""Run and persist the complete tail-risk classification experiment."""

from __future__ import annotations

# The full pipeline intentionally lives in this executable script.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from mwc_experiments.data import load_or_build_processed_data
from mwc_experiments.evaluation import (
    orientation_table,
    plot_classifier_diagnostics,
    plot_matrix,
    plot_metric_ranking,
    plot_shapley,
    top_interactions,
)
from mwc_experiments.settings import (
    HORIZONS,
    MAIN_RISK_FEATURES,
    PRIMARY_TAIL_ALPHA,
    ROBUSTNESS_TAIL_ALPHA,
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


STABILITY_CUTOFFS = (
    "2018-12-31",
    "2020-12-31",
    "2022-12-30",
    "2024-12-31",
    "2026-07-30",
)
ROBUSTNESS_MODELS = (
    "Prior probability",
    "Logistic",
    "Penalized logistic",
    "Gradient boosting",
    "Choquistic 1-additive",
    "Choquistic 2-additive",
)
REPORT_MODELS = (
    "Prior probability",
    "Logistic",
    "Explicit interactions",
    "Gradient boosting",
    "Choquistic 1-additive",
    "Choquistic 2-additive",
)


args = parse_experiment_args("Run the complete tail-risk experiment.")
quick = args.quick
verbose = not args.quiet
paths = prepare_output_paths()
dataset = load_or_build_processed_data(paths).equal_weight_dataset
primary = run_tail_classification_experiment(
    dataset,
    horizons=HORIZONS,
    alpha=PRIMARY_TAIL_ALPHA,
    quick=quick,
    verbose=verbose,
)
rare = run_tail_classification_experiment(
    dataset,
    horizons=HORIZONS,
    alpha=ROBUSTNESS_TAIL_ALPHA,
    quick=quick,
    model_names=ROBUSTNESS_MODELS,
    verbose=verbose,
)
stability = expanding_capacity_stability(
    dataset[list(MAIN_RISK_FEATURES)],
    dataset["tail_event_h5_a0p95"].astype(float),
    cutoffs=STABILITY_CUTOFFS,
    task="classification",
    purge=5,
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
    artifacts[f"failures h{horizon}"] = save_table(
        horizon_result.failures,
        f"experiment_2b_failures_h{horizon}_a095.csv",
        paths,
        index=False,
    )
    for model, shapley in horizon_result.shapley.items():
        legacy_name = model.lower().replace(" ", "_")
        artifacts[f"Shapley h{horizon} {model}"] = save_table(
            shapley,
            f"experiment_2b_shapley_h{horizon}_{legacy_name}.csv",
            paths,
        )
        artifacts[f"interactions h{horizon} {model}"] = save_table(
            horizon_result.interactions[model],
            f"experiment_2b_interactions_h{horizon}_{legacy_name}.csv",
            paths,
        )
        artifacts[f"top interactions h{horizon} {model}"] = save_table(
            top_interactions(horizon_result.interactions[model], n=12),
            f"experiment_2b_top_interactions_h{horizon}_{artifact_slug(model)}.csv",
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
    preferred = [
        model for model in REPORT_MODELS if model in horizon_result.probabilities
    ]
    axes = plot_classifier_diagnostics(
        horizon_result.split.y_test,
        horizon_result.probabilities,
        models=preferred,
    )
    axes[0].figure.suptitle(
        f"Classifier diagnostics — h={horizon}", y=1.03
    )
    artifacts[f"classifier diagnostics h{horizon}"] = save_figure(
        axes[0].figure,
        f"experiment_2b_classifier_diagnostics_h{horizon}_a095.png",
        paths,
    )

    probability_models = [
        model
        for model in (
            "Logistic",
            "Gradient boosting",
            "Choquistic 2-additive",
        )
        if model in horizon_result.probabilities
    ]
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
    axis.legend(ncol=2)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    artifacts[f"probability paths figure h{horizon}"] = save_figure(
        figure,
        f"experiment_2b_probability_paths_h{horizon}_a095.png",
        paths,
    )

    confusion_models = [
        model
        for model in (
            "Logistic",
            "Gradient boosting",
            "Choquistic 2-additive",
        )
        if model in horizon_result.probabilities
    ]
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
        slug = artifact_slug(model)
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
    axis.legend()
    axis.grid(alpha=0.2)
figure.tight_layout()
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
axis.figure.tight_layout()
artifacts["stability figure"] = save_figure(
    axis.figure,
    "experiment_2b_shapley_stability_h5_a095.png",
    paths,
)
print_artifacts(artifacts)
