from __future__ import annotations

import numpy as np
import pandas as pd

from capacities_ml_fin.finance import (
    aggregate_returns,
    forward_losses,
    lag_weights,
    portfolio_returns,
)
from mwc_experiments.modeling.splits import (
    chronological_split,
    rolling_walk_forward_splits,
)


def test_forward_log_loss_starts_after_forecast_origin() -> None:
    returns = pd.Series([0.01, 0.02, 0.03, 0.04], index=pd.RangeIndex(4))
    target = forward_losses(returns, horizon=2, method="log")
    assert np.isclose(target.iloc[0], -(0.02 + 0.03))
    assert np.isclose(target.iloc[1], -(0.03 + 0.04))
    assert target.iloc[-1] != target.iloc[-1]  # NaN: insufficient future observations


def test_lagged_weights_are_explicit() -> None:
    returns = pd.DataFrame(
        {"A": [0.1, 0.0], "B": [0.0, 0.2]}, index=pd.RangeIndex(2)
    )
    weights = pd.DataFrame(
        {"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=pd.RangeIndex(2)
    )
    lagged = lag_weights(weights)
    result = portfolio_returns(returns, lagged)
    assert np.isnan(result.iloc[0])
    assert np.isclose(result.iloc[1], 0.0)


def test_return_aggregation_respects_compounding_convention() -> None:
    simple = pd.Series([0.10, -0.05])
    aggregated = aggregate_returns(simple, horizon=2, method="simple")
    assert np.isclose(aggregated.iloc[1], (1.10 * 0.95) - 1.0)


def test_chronological_split_purges_horizon_boundaries() -> None:
    index = pd.bdate_range("2018-01-01", "2021-12-31")
    X = pd.DataFrame({"x": np.arange(len(index))}, index=index)
    y = pd.Series(np.arange(len(index)), index=index)
    split = chronological_split(
        X,
        y,
        train_end="2018-12-31",
        validation_end="2019-12-31",
        horizon=5,
    )
    assert split.X_train.index.max() <= pd.Timestamp("2018-12-24")
    assert split.X_validation.index.max() <= pd.Timestamp("2019-12-24")
    assert split.X_test.index.min() > pd.Timestamp("2019-12-31")


def test_rolling_walk_forward_moves_window_and_purges_every_boundary() -> None:
    index = pd.bdate_range("2014-01-01", "2022-12-30")
    X = pd.DataFrame({"x": np.arange(len(index))}, index=index)
    y = pd.Series(np.arange(len(index)), index=index)
    folds = list(
        rolling_walk_forward_splits(
            X,
            y,
            oos_start="2020-01-01",
            training_window_years=5,
            validation_window_months=12,
            oos_block_years=1,
            horizon=10,
        )
    )

    assert len(folds) == 3
    positions = pd.Series(np.arange(len(index)), index=index)
    for fold in folds:
        split = fold.split
        assert positions[split.X_train.index.max()] + 10 < positions[
            split.X_validation.index.min()
        ]
        assert positions[split.X_validation.index.max()] + 10 < positions[
            split.X_test.index.min()
        ]
        assert split.X_train.index.min() >= fold.window_start
        assert split.X_test.index.min() >= fold.oos_start
        assert split.X_test.index.max() < fold.oos_end
    assert folds[1].window_start == folds[0].window_start + pd.DateOffset(
        years=1
    )
