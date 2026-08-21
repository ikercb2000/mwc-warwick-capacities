from __future__ import annotations

import numpy as np
import pandas as pd

from mwc_experiments.evaluation import (
    clipping_diagnostics,
    fit_empirical_stress_definition,
    regression_estimation_robustness,
    regression_regime_metrics,
)
from mwc_experiments.modeling.registries import regression_candidates


def _sample() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2020-01-01", periods=60, freq="D")
    first = np.linspace(-1.0, 1.0, len(index))
    second = np.sin(np.linspace(0.0, 4.0, len(index)))
    X = pd.DataFrame({"first": first, "second": second}, index=index)
    y = pd.Series(1.5 * first - 0.4 * second, index=index, name="loss")
    return X, y


def test_empirical_stress_thresholds_are_reused_out_of_sample() -> None:
    X, y = _sample()
    definition = fit_empirical_stress_definition(X.iloc[:40], y.iloc[:40])
    X_test = X.iloc[40:].copy()
    y_test = y.iloc[40:].copy()
    X_test.iloc[0, 0] = 100.0

    mask = definition.mask(X_test, y_test)
    audit = definition.audit(X_test, y_test, sample="test")

    assert bool(mask.iloc[0])
    assert audit["sample"] == "test"
    assert audit["stress observations"] == int(mask.sum())
    assert definition.feature_upper["first"] < 100.0


def test_clipping_and_estimation_robustness_use_raw_extreme_dates() -> None:
    X, y = _sample()
    X_fit, y_fit = X.iloc[:45], y.iloc[:45]
    X_test, y_test = X.iloc[45:], y.iloc[45:]
    candidate = regression_candidates(
        2,
        clipping=True,
        include_mlp=False,
        include_dummy=False,
        include_regularized_choquet=False,
    )["OLS"].estimator
    fitted = candidate.fit(X_fit, y_fit)
    predictions = pd.DataFrame(
        {"OLS": fitted.predict(X_test)},
        index=X_test.index,
    )
    definition = fit_empirical_stress_definition(X_fit, y_fit)
    fit_mask = definition.mask(X_fit, y_fit)
    test_mask = definition.mask(X_test, y_test)

    clipping = clipping_diagnostics(fitted, X_test, sample="test")
    regime = regression_regime_metrics(y_test, predictions, test_mask)
    robustness = regression_estimation_robustness(
        {"OLS": fitted},
        X_fit,
        y_fit,
        X_test,
        y_test,
        predictions,
        fit_extreme_mask=fit_mask,
        test_stress_mask=test_mask,
    )

    assert (clipping["sample"] == "test").all()
    assert clipping["clipping enabled"].all()
    assert clipping["clipped observations"].sum() > 0
    assert regime.index.name == "model"
    assert regime.loc["OLS", "stress observations"] == int(test_mask.sum())
    assert robustness.loc["OLS", "failure"] == ""
    assert robustness.loc["OLS", "extreme fit observations removed"] == int(
        fit_mask.sum()
    )


def test_clipping_is_disabled_by_default() -> None:
    X, y = _sample()
    fitted = regression_candidates(
        2,
        include_mlp=False,
        include_dummy=False,
        include_regularized_choquet=False,
    )["OLS"].estimator.fit(X.iloc[:45], y.iloc[:45])

    audit = clipping_diagnostics(fitted, X.iloc[45:], sample="test")

    assert not audit["clipping enabled"].any()
    assert audit["clipped observations"].sum() == 0
