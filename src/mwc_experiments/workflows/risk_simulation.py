from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from capacities_ml_fin.risk import (
    DistortedCapacity,
    ExpectedShortfallDistortion,
    IdentityDistortion,
    ProbabilityCapacity,
    ProportionalHazardsDistortion,
    RollingRiskEstimator,
    ValueAtRiskDistortion,
    capital_backtest,
    christoffersen_independence_test,
    distortion_risk_measure,
    diversification_benefit,
    expected_shortfall,
    kupiec_coverage_test,
    risk_contributions,
    stress_backtest,
    value_at_risk,
)


def simulate_regime_switching_losses(
    n_observations: int = 4_000,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Simulate three heavy-tailed asset-loss series with persistent stress regimes.

    The simulation is deliberately transparent rather than calibrated to a particular
    market. It creates calm and stress states, correlated Student-t shocks and a sharper
    increase in common dependence during stress. The resulting sample is useful for
    illustrating distortion risk measures, capital backtests and diversification.
    """
    if n_observations < 300:
        raise ValueError("n_observations must be at least 300.")
    rng = np.random.default_rng(random_state)

    stress = np.zeros(n_observations, dtype=bool)
    for t in range(1, n_observations):
        if stress[t - 1]:
            stress[t] = rng.random() > 0.12  # persistent stress state
        else:
            stress[t] = rng.random() < 0.018

    calm_correlation = np.array(
        [[1.0, 0.25, 0.15], [0.25, 1.0, 0.20], [0.15, 0.20, 1.0]]
    )
    stress_correlation = np.array(
        [[1.0, 0.72, 0.58], [0.72, 1.0, 0.64], [0.58, 0.64, 1.0]]
    )
    calm_scale = np.array([0.010, 0.008, 0.009])
    stress_scale = np.array([0.030, 0.025, 0.028])
    calm_cholesky = np.linalg.cholesky(calm_correlation)
    stress_cholesky = np.linalg.cholesky(stress_correlation)

    losses = np.empty((n_observations, 3), dtype=float)
    degrees_of_freedom = 5.0
    for t in range(n_observations):
        normal = rng.standard_normal(3)
        radial = np.sqrt(rng.chisquare(degrees_of_freedom) / degrees_of_freedom)
        if stress[t]:
            shock = stress_cholesky @ normal / radial
            losses[t] = 0.0018 + stress_scale * shock
        else:
            shock = calm_cholesky @ normal / radial
            losses[t] = -0.0002 + calm_scale * shock

    # A few rare common jumps make tail comparisons visible without determining them.
    jump = rng.random(n_observations) < 0.006
    losses[jump] += rng.lognormal(mean=-3.3, sigma=0.45, size=(jump.sum(), 1))

    index = pd.bdate_range("2008-01-02", periods=n_observations)
    frame = pd.DataFrame(
        losses,
        index=index,
        columns=["asset_a_loss", "asset_b_loss", "asset_c_loss"],
    )
    frame["portfolio_loss"] = frame[
        ["asset_a_loss", "asset_b_loss", "asset_c_loss"]
    ].mean(axis=1)
    frame["stress"] = stress
    return frame


def static_risk_table(
    losses: pd.Series | np.ndarray,
    *,
    var_levels: tuple[float, ...] = (0.95, 0.975, 0.99),
    ph_gammas: tuple[float, ...] = (0.85, 0.70, 0.50),
) -> pd.DataFrame:
    """Return a comparison of empirical VaR, ES and proportional-hazards capital."""
    values = np.asarray(losses, dtype=float)
    rows: list[dict[str, float | str]] = [
        {
            "measure": "Identity / mean loss",
            "parameter": "-",
            "capital": distortion_risk_measure(values, IdentityDistortion()),
        }
    ]
    for alpha in var_levels:
        rows.extend(
            [
                {
                    "measure": "Value-at-Risk",
                    "parameter": f"alpha={alpha:g}",
                    "capital": value_at_risk(values, alpha),
                },
                {
                    "measure": "Expected Shortfall",
                    "parameter": f"alpha={alpha:g}",
                    "capital": expected_shortfall(values, alpha),
                },
            ]
        )
    for gamma in ph_gammas:
        rows.append(
            {
                "measure": "Proportional hazards",
                "parameter": f"gamma={gamma:g}",
                "capital": distortion_risk_measure(
                    values, ProportionalHazardsDistortion(gamma)
                ),
            }
        )
    return pd.DataFrame(rows).set_index(["measure", "parameter"]).sort_values("capital")


def rolling_capital_panel(
    losses: pd.Series,
    *,
    window: int = 252,
) -> pd.DataFrame:
    """Estimate one-step-ahead capital from rolling historical windows."""
    measures = {
        "VaR 99%": ValueAtRiskDistortion(0.99),
        "ES 97.5%": ExpectedShortfallDistortion(0.975),
        "PH gamma=0.70": ProportionalHazardsDistortion(0.70),
        "PH gamma=0.50": ProportionalHazardsDistortion(0.50),
    }
    capital = {}
    for name, measure in measures.items():
        estimator = RollingRiskEstimator(
            measure,
            window=window,
            window_type="rolling",
            min_periods=window,
            horizon=1,
        ).fit(losses)
        capital[name] = estimator.predict_in_sample()
    return pd.DataFrame(capital, index=losses.index)


def capital_backtest_table(
    losses: pd.Series,
    capital: pd.DataFrame,
    *,
    stress_mask: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Summarise overall and stress-period performance for each capital series."""
    rows: list[dict[str, object]] = []
    for name in capital:
        aligned = pd.concat([losses.rename("loss"), capital[name].rename("capital")], axis=1).dropna()
        overall = capital_backtest(aligned["loss"], aligned["capital"])
        row: dict[str, object] = {"model": name, "sample": "all", **asdict(overall)}
        rows.append(row)
        if stress_mask is not None:
            mask = pd.Series(stress_mask, index=losses.index).reindex(aligned.index).fillna(False)
            stressed = stress_backtest(
                aligned["loss"].to_numpy(),
                aligned["capital"].to_numpy(),
                mask.to_numpy(dtype=bool),
            )
            rows.append({"model": name, "sample": "stress", **asdict(stressed)})
    return pd.DataFrame(rows).set_index(["model", "sample"])


def coverage_test_table(losses: pd.Series, capital: pd.DataFrame) -> pd.DataFrame:
    """Run coverage/independence tests where a nominal VaR level is defined."""
    rows = []
    nominal = {"VaR 99%": 0.99}
    for name, alpha in nominal.items():
        aligned = pd.concat([losses.rename("loss"), capital[name].rename("capital")], axis=1).dropna()
        kupiec = kupiec_coverage_test(aligned["loss"], aligned["capital"], alpha)
        independence = christoffersen_independence_test(aligned["loss"], aligned["capital"])
        rows.extend(
            [
                {
                    "model": name,
                    "test": "Kupiec unconditional coverage",
                    "statistic": kupiec.statistic,
                    "p_value": kupiec.p_value,
                },
                {
                    "model": name,
                    "test": "Christoffersen independence",
                    "statistic": independence.statistic,
                    "p_value": independence.p_value,
                },
            ]
        )
    return pd.DataFrame(rows).set_index(["model", "test"])


def ordered_risk_contributions(
    losses: pd.Series | np.ndarray,
    *,
    distortion=None,
) -> pd.DataFrame:
    """Return the ordered Choquet increment decomposition for a distortion."""
    values = np.asarray(losses, dtype=float)
    if distortion is None:
        distortion = ExpectedShortfallDistortion(0.975)
    capacity = DistortedCapacity(
        ProbabilityCapacity(np.ones(values.size)),
        distortion,
    )
    return risk_contributions(values, capacity)


def diversification_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare diversification benefits for several risk measures."""
    first = frame["asset_a_loss"].to_numpy()
    second = frame["asset_b_loss"].to_numpy()
    measures = {
        "VaR 99%": lambda x: value_at_risk(x, 0.99),
        "ES 97.5%": lambda x: expected_shortfall(x, 0.975),
        "PH gamma=0.70": lambda x: distortion_risk_measure(
            x, ProportionalHazardsDistortion(0.70)
        ),
        "PH gamma=0.50": lambda x: distortion_risk_measure(
            x, ProportionalHazardsDistortion(0.50)
        ),
    }
    rows = []
    for name, measure in measures.items():
        rows.append(
            {
                "measure": name,
                "rho(A)": measure(first),
                "rho(B)": measure(second),
                "rho(A+B)": measure(first + second),
                "diversification benefit": diversification_benefit(
                    first, second, measure
                ),
            }
        )
    return pd.DataFrame(rows).set_index("measure")
