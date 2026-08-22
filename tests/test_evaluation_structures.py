"""Contracts for fixed and rolling_5y predictive evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mwc_experiments.configuration import load_experiment_config
from mwc_experiments.modeling.splits import evaluation_splits


ROOT = Path(__file__).resolve().parents[1]


def _sample() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.bdate_range("2014-01-01", "2022-12-30")
    values = np.arange(len(index), dtype=float)
    return pd.DataFrame({"x": values}, index=index), pd.Series(values, index=index)


def test_fixed_uses_exact_calendar_partitions_and_purges() -> None:
    X, y = _sample()
    folds = evaluation_splits(X, y, evaluation_structure="fixed", horizon=10)
    assert len(folds) == 1
    split = folds[0].split
    positions = pd.Series(np.arange(len(X.index)), index=X.index)
    assert split.X_train.index.min() == X.index.min()
    assert positions[split.X_train.index.max()] + 10 < positions[
        split.X_validation.index.min()
    ]
    assert split.X_validation.index.min() == pd.Timestamp("2019-01-01")
    assert positions[split.X_validation.index.max()] + 10 < positions[
        split.X_test.index.min()
    ]
    assert split.X_test.index.min() == pd.Timestamp("2020-01-01")
    assert split.X_test.index.max() == X.index.max()


def test_rolling_5y_has_five_year_lookbacks_and_annual_steps() -> None:
    X, y = _sample()
    folds = evaluation_splits(
        X, y, evaluation_structure="rolling_5y", horizon=5
    )
    assert len(folds) == 3
    for fold in folds:
        assert fold.window_start == fold.oos_start - pd.DateOffset(years=5)
        assert fold.validation_start == fold.oos_start - pd.DateOffset(months=12)
    for previous, current in zip(folds, folds[1:]):
        assert current.window_start == previous.window_start + pd.DateOffset(years=1)
        assert current.oos_start == previous.oos_start + pd.DateOffset(years=1)


def test_predictive_defaults_and_full_wrappers_cover_four_variants() -> None:
    for experiment in ("factor_models", "future_loss", "tail_risk"):
        config = load_experiment_config(experiment)
        assert config["evaluation"]["structure"] == "rolling_5y"
        assert config["preprocessing"]["clipping_enabled"] is False

    windows = (ROOT / "scripts" / "run_all_experiments.ps1").read_text(
        encoding="utf-8"
    )
    unix = (ROOT / "scripts" / "run_all_experiments.sh").read_text(
        encoding="utf-8"
    )
    for source in (windows, unix):
        assert "fixed" in source and "rolling_5y" in source
        assert "--evaluation-structure" in source
        assert "--with-clipping" in source
        assert "--without-clipping" in source
    assert "evaluation_structure" not in (
        ROOT / "scripts" / "experiment_distortion_risk.py"
    ).read_text(encoding="utf-8")