"""Leakage and reporting tests for tail-risk probability calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline

from mwc_experiments.configuration import load_experiment_config
from mwc_experiments.evaluation.metrics import classification_metrics
from mwc_experiments.workflows import run_tail_classification_experiment


def _classification_dataset(*, invert_test_labels: bool = False) -> pd.DataFrame:
    index = pd.bdate_range("2016-01-04", "2021-12-31")
    position = np.arange(len(index))
    event = ((position % 7) == 0).astype(float)
    signal_event = event.copy()
    if invert_test_labels:
        event[index >= pd.Timestamp("2021-01-01")] = (
            1.0 - event[index >= pd.Timestamp("2021-01-01")]
        )
    return pd.DataFrame(
        {
            "signal": np.sin(position / 11.0) + 0.5 * signal_event,
            "trend": position / len(index),
            "tail_event_h1_a0p95": event,
        },
        index=index,
    )


def _run(dataset: pd.DataFrame):
    return run_tail_classification_experiment(
        dataset,
        features=("signal", "trend"),
        horizons=(1,),
        alpha=0.95,
        quick=True,
        model_names=("Logistic",),
        oos_start="2020-01-01",
        training_window_years=4,
        selection_window_months=12,
        calibration_window_months=12,
        oos_block_years=1,
        calibration_methods=("sigmoid", "isotonic"),
        verbose=False,
    )[1]


def test_tail_risk_uses_frozen_full_pipelines_for_temporal_calibration() -> None:
    result = _run(_classification_dataset())

    assert list(result.probabilities) == [
        "Logistic",
        "Logistic [sigmoid]",
        "Logistic [isotonic]",
    ]
    assert set(result.calibrated_models) == {
        "Logistic [sigmoid]",
        "Logistic [isotonic]",
    }
    for calibrated in result.calibrated_models.values():
        assert isinstance(calibrated, CalibratedClassifierCV)
        assert isinstance(calibrated.estimator, FrozenEstimator)
        assert isinstance(calibrated.estimator.estimator, Pipeline)

    summary = result.calibration_sample_summary
    for fold in summary.index.get_level_values("fold").unique():
        fold_summary = summary.xs(fold, level="fold")
        assert fold_summary.loc["train", "end"] < fold_summary.loc[
            "selection_validation", "start"
        ]
        assert fold_summary.loc["selection_validation", "end"] < (
            fold_summary.loc["calibration", "start"]
        )
        assert fold_summary.loc["calibration", "end"] < fold_summary.loc[
            "OOS", "start"
        ]
    assert set(result.discrimination_metrics) >= {"ROC AUC", "PR AUC"}
    assert set(result.calibration_metrics) >= {
        "Brier",
        "Log loss",
        "Mean predicted probability",
        "Observed event prevalence",
        "Calibration gap",
    }


def test_test_labels_never_affect_selection_or_calibration() -> None:
    original = _run(_classification_dataset())
    changed_test = _run(_classification_dataset(invert_test_labels=True))

    pd.testing.assert_frame_equal(
        original.probabilities,
        changed_test.probabilities,
    )
    pd.testing.assert_frame_equal(original.thresholds, changed_test.thresholds)
    pd.testing.assert_frame_equal(
        original.selected_parameters,
        changed_test.selected_parameters,
    )


def test_classification_metrics_compare_prevalence_and_mean_probability() -> None:
    metrics = classification_metrics(
        [0, 0, 1, 1],
        [0.1, 0.3, 0.6, 0.8],
    )

    assert metrics["Observed event prevalence"] == pytest.approx(0.5)
    assert metrics["Mean predicted probability"] == pytest.approx(0.45)
    assert metrics["Calibration gap"] == pytest.approx(-0.05)
    assert metrics["Absolute calibration gap"] == pytest.approx(0.05)
    assert metrics["Observed event rate"] == metrics[
        "Observed event prevalence"
    ]


def test_tail_risk_calibration_is_configurable() -> None:
    config = load_experiment_config("tail_risk")

    assert config["walk_forward"]["training_window_years"] == 5
    assert config["walk_forward"]["selection_window_months"] == 18
    assert config["walk_forward"]["calibration_window_months"] == 24
    assert config["walk_forward"]["oos_block_years"] == 1
    assert config["calibration"]["methods"] == ["sigmoid", "isotonic"]
