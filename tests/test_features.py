from __future__ import annotations

import numpy as np
import pandas as pd

from mwc_experiments.data.features import _portfolio_liquidity


def test_amihud_scaling_uses_only_strictly_prior_observations() -> None:
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    amihud = pd.DataFrame(
        {
            "first": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
            "second": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
        },
        index=index,
    )
    weights = pd.Series({"first": 0.5, "second": 0.5})

    liquidity = _portfolio_liquidity(amihud, weights)
    revised_future = amihud.copy()
    revised_future.iloc[-1] = 10_000.0
    revised_liquidity = _portfolio_liquidity(revised_future, weights)

    assert np.isnan(liquidity.iloc[0])
    assert liquidity.iloc[1] == np.log1p(2.0 / 1.0)
    assert liquidity.iloc[-1] == np.log1p(100.0 / 3.0)
    pd.testing.assert_series_equal(
        liquidity.iloc[:-1],
        revised_liquidity.iloc[:-1],
    )
