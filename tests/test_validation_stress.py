from __future__ import annotations

import numpy as np
import pandas as pd

from mwc_experiments.workflows import compare_validation_stress_regimes


def test_validation_stress_comparison_uses_a_common_post_2021_test() -> None:
    index = pd.bdate_range("2016-01-01", "2023-12-31")
    position = np.arange(len(index), dtype=float)
    X = pd.DataFrame(
        {
            "first": np.sin(position / 30.0),
            "second": np.cos(position / 50.0),
        },
        index=index,
    )
    y = pd.Series(2.0 * X["first"] - 0.5 * X["second"], index=index)

    result = compare_validation_stress_regimes(
        X,
        y,
        model_names=("OLS", "Ridge"),
        quick=True,
    )

    assert result.sample_summary.loc["test", "start"] > pd.Timestamp(
        "2021-12-31"
    )
    assert (
        result.sample_summary.loc["walk-forward validation", "observations"]
        > result.sample_summary.loc[
            "validation excluding 2020-2021", "observations"
        ]
    )
    assert (
        result.sample_summary.loc["final refit", "observations"]
        > result.sample_summary.loc["initial training", "observations"]
    )
    assert result.metrics["validation folds"].eq(4).all()
    assert result.metrics.groupby(level="regime")[
        "selected by validation"
    ].sum().eq(1).all()
    assert (result.selection_summary["selection regret"] >= 0.0).all()
    assert set(result.selection_summary.index) == {
        "including stress",
        "excluding 2020-2021",
    }
