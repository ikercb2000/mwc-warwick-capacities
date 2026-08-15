"""Construct financial features and reusable prepared market panels."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from mwc_experiments.settings import (
    ANNUALIZATION_FACTOR,
    EQUITY_TICKERS,
    SECTOR_ETF_BY_ASSET,
    SECTOR_ETFS,
)
from mwc_experiments.data.types import PreparedMarketData, RawMarketData
from capacities_ml_fin.finance import (
    amihud_illiquidity,
    drawdown,
    equal_weights,
    lag_weights,
    market_cap_weights,
    momentum,
    portfolio_returns,
    price_returns,
    realized_volatility,
    to_losses,
)


def _average_pairwise_rolling_correlation(
    returns: pd.DataFrame,
    window: int,
) -> pd.Series:
    """Compute the mean rolling correlation across all return-series pairs."""
    pairwise = [
        returns[first].rolling(window, min_periods=window).corr(returns[second])
        for first, second in combinations(returns.columns, 2)
    ]
    return pd.concat(pairwise, axis=1).mean(axis=1).rename(
        f"average_correlation_{window}d"
    )


def _portfolio_liquidity(
    amihud: pd.DataFrame,
    weights: pd.Series | pd.DataFrame,
) -> pd.Series:
    """Aggregate Amihud illiquidity into a stable portfolio stress measure."""
    if isinstance(weights, pd.Series):
        values = amihud.mul(weights.reindex(amihud.columns), axis=1).sum(axis=1)
    else:
        values = (amihud * weights.reindex(index=amihud.index, columns=amihud.columns)).sum(axis=1)
    # Amihud values are tiny and very skewed. log(1+x) after scaling retains order and
    # produces a numerically stable stress indicator.
    positive = values.where(values > 0.0)
    scale = positive.median(skipna=True)
    scale = 1.0 if not np.isfinite(scale) or scale <= 0.0 else scale
    return np.log1p(values / scale).rename("liquidity_stress_20d")


def _portfolio_feature_table(
    *,
    portfolio_returns_series: pd.Series,
    portfolio_losses_series: pd.Series,
    liquidity_stress: pd.Series,
    equity_returns: pd.DataFrame,
    etf_returns: pd.DataFrame,
    fred: pd.DataFrame,
) -> pd.DataFrame:
    """Build the common predictor table for one portfolio weighting scheme."""
    features = pd.DataFrame(index=portfolio_returns_series.index)

    features["market_loss_1d"] = -etf_returns["SPY"]
    features["small_cap_relative_loss_1d"] = -(
        etf_returns["IWM"] - etf_returns["SPY"]
    )
    features["vix"] = fred["vix"]
    features["vix_change_1d"] = fred["vix"].diff()

    features["realized_volatility_20d"] = realized_volatility(
        portfolio_returns_series,
        window=20,
        annualize=True,
        periods_per_year=ANNUALIZATION_FACTOR,
    )
    features["realized_volatility_60d"] = realized_volatility(
        portfolio_returns_series,
        window=60,
        annualize=True,
        periods_per_year=ANNUALIZATION_FACTOR,
    )
    features["drawdown_severity"] = -drawdown(
        portfolio_returns_series,
        method="log",
    )
    features["negative_momentum_21d"] = -momentum(
        portfolio_returns_series,
        lookback=21,
        skip=0,
        method="log",
    )
    features["lagged_portfolio_loss_1d"] = portfolio_losses_series.shift(1)
    features["lagged_portfolio_loss_5d"] = (
        portfolio_losses_series.shift(1).rolling(5, min_periods=5).sum()
    )
    features["liquidity_stress_20d"] = liquidity_stress

    features["baa_credit_spread_pct"] = fred["baa_credit_spread_pct"]
    features["baa_credit_spread_change_5d_bps"] = (
        fred["baa_credit_spread_pct"].diff(5) * 100.0
    )
    features["term_spread_10y_2y_pct"] = (
        fred["treasury_10y_yield_pct"] - fred["treasury_2y_yield_pct"]
    )
    features["treasury_2y_change_5d_bps"] = (
        fred["treasury_2y_yield_pct"].diff(5) * 100.0
    )
    features["fed_funds_effective_pct"] = fred["fed_funds_effective_pct"]

    features["average_equity_correlation_60d"] = (
        _average_pairwise_rolling_correlation(equity_returns, 60)
    )
    sector_returns = etf_returns[list(SECTOR_ETFS)]
    features["sector_return_dispersion_20d"] = (
        sector_returns.std(axis=1).rolling(20, min_periods=20).mean()
    )
    features["equity_return_dispersion_20d"] = (
        equity_returns.std(axis=1).rolling(20, min_periods=20).mean()
    )
    return features.replace([np.inf, -np.inf], np.nan)


def prepare_market_data(raw: RawMarketData) -> PreparedMarketData:
    """Construct all reusable return, portfolio, liquidity and stress panels."""
    equity_tri = raw.equity_fields["total_return_index"]
    etf_tri = raw.etf_fields["total_return_index"]
    equity_returns = price_returns(equity_tri, method="log")
    etf_returns = price_returns(etf_tri, method="log")
    equity_returns.columns = list(EQUITY_TICKERS)
    equity_losses = to_losses(equity_returns)
    etf_losses = to_losses(etf_returns)

    annual_3m = raw.fred["treasury_3m_yield_pct"] / 100.0
    risk_free_simple = (1.0 + annual_3m).pow(1.0 / ANNUALIZATION_FACTOR) - 1.0
    risk_free_log = np.log1p(risk_free_simple).rename("risk_free_log")

    ew = equal_weights(EQUITY_TICKERS)
    equal_weight_returns = portfolio_returns(equity_returns, ew).rename(
        "equal_weight_return"
    )
    equal_weight_losses = (-equal_weight_returns).rename("equal_weight_loss")

    contemporaneous_cap_weights = market_cap_weights(raw.equity_fields["market_cap"])
    lagged_cap_weights = lag_weights(contemporaneous_cap_weights, periods=1)
    cap_weight_returns = portfolio_returns(equity_returns, lagged_cap_weights).rename(
        "cap_weight_return"
    )
    cap_weight_losses = (-cap_weight_returns).rename("cap_weight_loss")

    dollar_volume = raw.equity_fields["px_last"] * raw.equity_fields["volume"]
    equity_amihud_20d = amihud_illiquidity(
        equity_returns,
        dollar_volume,
        window=20,
    )
    equal_liquidity = _portfolio_liquidity(equity_amihud_20d, ew)
    cap_liquidity = _portfolio_liquidity(equity_amihud_20d, lagged_cap_weights)

    equal_features = _portfolio_feature_table(
        portfolio_returns_series=equal_weight_returns,
        portfolio_losses_series=equal_weight_losses,
        liquidity_stress=equal_liquidity,
        equity_returns=equity_returns,
        etf_returns=etf_returns,
        fred=raw.fred,
    )
    cap_features = _portfolio_feature_table(
        portfolio_returns_series=cap_weight_returns,
        portfolio_losses_series=cap_weight_losses,
        liquidity_stress=cap_liquidity,
        equity_returns=equity_returns,
        etf_returns=etf_returns,
        fred=raw.fred,
    )

    return PreparedMarketData(
        equity_returns=equity_returns,
        equity_losses=equity_losses,
        etf_returns=etf_returns,
        etf_losses=etf_losses,
        fred=raw.fred,
        risk_free_log=risk_free_log,
        equal_weight_returns=equal_weight_returns,
        equal_weight_losses=equal_weight_losses,
        cap_weight_returns=cap_weight_returns,
        cap_weight_losses=cap_weight_losses,
        lagged_cap_weights=lagged_cap_weights,
        equity_amihud_20d=equity_amihud_20d,
        portfolio_features_equal=equal_features,
        portfolio_features_cap=cap_features,
    )


def factor_frame(data: PreparedMarketData, asset: str) -> pd.DataFrame:
    """Build the same-day factor model frame for one equity."""
    if asset not in EQUITY_TICKERS:
        raise KeyError(f"Unknown asset {asset!r}.")
    sector = SECTOR_ETF_BY_ASSET[asset]
    etf = data.etf_returns
    rf = data.risk_free_log

    frame = pd.DataFrame(index=etf.index)
    frame["market_excess_return"] = etf["SPY"] - rf
    frame["sector_excess_return"] = etf[sector] - rf
    frame["size_spread"] = etf["IWM"] - etf["SPY"]
    frame["value_growth_spread"] = etf["IVE"] - etf["IVW"]
    frame["momentum_spread"] = etf["MTUM"] - etf["SPY"]
    frame["quality_spread"] = etf["QUAL"] - etf["SPY"]
    frame["low_volatility_spread"] = etf["USMV"] - etf["SPY"]
    frame["target_excess_loss"] = -(
        data.equity_returns[asset] - rf
    )
    return frame.dropna()
