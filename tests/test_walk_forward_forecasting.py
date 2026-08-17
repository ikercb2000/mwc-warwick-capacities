"""Rolling walk-forward contracts for the forecasting workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mwc_experiments.configuration import load_experiment_config
from mwc_experiments.workflows import run_future_loss_experiment


def _regression_dataset(*, change_final_targets: bool = False) -> pd.DataFrame:
    index = pd.bdate_range("2014-01-02", "2022-12-30")
    position = np.arange(len(index))
    x1 = np.sin(position / 17.0)
    x2 = np.cos(position / 31.0)
    target = 0.4 * x1 - 0.2 * x2 + 0.001 * position
    if change_final_targets:
        target = target.copy()
        target[index >= pd.Timestamp("2022-01-01")] += 100.0
    return pd.DataFrame(
        {"x1": x1, "x2": x2, "future_loss_h1": target},
        index=index,
    )


def _run(dataset: pd.DataFrame):
    return run_future_loss_experiment(
        dataset,
        features=("x1", "x2"),
        horizons=(1,),
        quick=True,
        model_names=("OLS", "Ridge", "OLS oriented", "Choquet 1-additive"),
        oos_start="2020-01-01",
        training_window_years=5,
        validation_window_months=12,
        oos_block_years=1,
        aggregation_model_name="Choquet model aggregator",
        aggregation_base_models=("OLS", "Ridge"),
        verbose=False,
    ).horizons[1]


def test_future_loss_refits_preprocessing_and_capacity_in_every_fold() -> None:
    result = _run(_regression_dataset())

    assert len(result.fold_summary) == 3
    assert result.metrics["OOS folds"].eq(3).all()
    assert result.predictions.index.min() == pd.Timestamp("2020-01-01")
    assert result.predictions.index.max().year == 2022
    assert result.selected_parameters.index.get_level_values("fold").nunique() == 3
    assert result.orientation_history.index.get_level_values("fold").nunique() == 3
    choquet_shapley = result.shapley_history.xs(
        "Choquet 1-additive",
        level="model",
    )
    assert choquet_shapley.index.get_level_values("fold").nunique() == 3
    assert result.final_split.X_test.index.min().year == 2022
    assert result.split.X_test.index.min().year == 2020
    assert "Choquet model aggregator" in result.predictions
    assert set(result.shapley["Choquet model aggregator"].index) == {
        "OLS",
        "Ridge",
    }


def test_final_oos_targets_cannot_change_walk_forward_predictions() -> None:
    original = _run(_regression_dataset())
    changed = _run(_regression_dataset(change_final_targets=True))

    pd.testing.assert_frame_equal(original.predictions, changed.predictions)
    pd.testing.assert_frame_equal(
        original.selected_parameters,
        changed.selected_parameters,
    )


def test_forecasting_walk_forward_settings_are_configurable() -> None:
    future_loss = load_experiment_config("future_loss")["walk_forward"]
    tail_risk = load_experiment_config("tail_risk")["walk_forward"]

    for config in (future_loss, tail_risk):
        assert config["oos_start"] == "2020-01-01"
        assert config["training_window_years"] == 5
        assert config["oos_block_years"] == 1
