"""Plots domain."""

from __future__ import annotations
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

def plot_metric_ranking(metrics: pd.DataFrame, metric: str, *, lower_is_better: bool = True):
    order = metrics[metric].sort_values(ascending=lower_is_better)
    axis = order.plot(kind="bar", figsize=(10, 4), title=f"Model comparison: {metric}")
    axis.set_ylabel(metric)
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return axis


def plot_actual_predictions(
    actual: pd.Series,
    predictions: pd.DataFrame,
    *,
    models: list[str] | None = None,
    title: str = "Actual and predicted outcomes",
    start: str | None = None,
):
    selected = predictions if models is None else predictions[models]
    if start is not None:
        actual = actual.loc[start:]
        selected = selected.loc[start:]
    axis = actual.plot(figsize=(12, 4), label="Observed", linewidth=1.2)
    selected.plot(ax=axis, linewidth=0.9)
    axis.set_title(title)
    axis.legend(ncol=2)
    axis.grid(alpha=0.2)
    plt.tight_layout()
    return axis


def plot_matrix(matrix: pd.DataFrame, *, title: str, symmetric: bool = True):
    values = matrix.to_numpy(dtype=float)
    if symmetric:
        limit = np.nanmax(np.abs(values))
        vmin, vmax = -limit, limit
    else:
        vmin, vmax = np.nanmin(values), np.nanmax(values)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(values, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=90)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return ax


def plot_shapley(shapley: pd.Series, *, title: str):
    axis = shapley.sort_values().plot(kind="barh", figsize=(8, 4), title=title)
    axis.set_xlabel("Shapley importance")
    axis.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    return axis


def plot_classifier_discrimination(
    y_true,
    probabilities: pd.DataFrame,
    *,
    models: list[str] | None = None,
    legend_outside: bool = False,
):
    """Plot ROC and precision-recall curves for classifier discrimination."""
    selected_models = list(probabilities.columns) if models is None else models
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for model in selected_models:
        probability = probabilities[model]
        RocCurveDisplay.from_predictions(y_true, probability, name=model, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_true, probability, name=model, ax=axes[1])
    axes[0].set_title("ROC curves")
    axes[1].set_title("Precision-recall curves")
    if legend_outside:
        handles, labels = axes[1].get_legend_handles_labels()
        for axis in axes:
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.81, 0.5),
            fontsize=8,
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
    else:
        fig.tight_layout()
    return axes


def plot_probability_calibration(
    y_true,
    probabilities: pd.DataFrame,
    *,
    models: list[str] | None = None,
    legend_outside: bool = False,
):
    """Plot reliability curves without mixing in discrimination diagnostics."""
    selected_models = list(probabilities.columns) if models is None else models
    fig, axis = plt.subplots(figsize=(7, 4))
    for model in selected_models:
        observed, predicted = calibration_curve(
            y_true,
            probabilities[model],
            n_bins=10,
            strategy="quantile",
        )
        axis.plot(predicted, observed, marker="o", label=model)
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="perfect calibration",
    )
    axis.set_title("Probability calibration")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed frequency")
    if legend_outside:
        axis.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=8,
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.72, 1.0))
    else:
        axis.legend(fontsize=8)
        fig.tight_layout()
    return axis
