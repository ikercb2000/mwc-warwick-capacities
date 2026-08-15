"""Chronological data splitting utilities for empirical evaluation."""

from __future__ import annotations

from typing import Iterator

import pandas as pd

from mwc_experiments.settings import TRAIN_END, VALIDATION_END
from mwc_experiments.modeling.types import TemporalSplit


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    train_end: str = TRAIN_END,
    validation_end: str = VALIDATION_END,
    horizon: int = 0,
) -> TemporalSplit:
    """Create chronological train, validation and test samples with purging."""
    common = X.join(y.rename("__target__"), how="inner").dropna()
    X_clean = common[X.columns]
    y_clean = common["__target__"]

    train_mask = X_clean.index <= pd.Timestamp(train_end)
    validation_mask = (
        (X_clean.index > pd.Timestamp(train_end))
        & (X_clean.index <= pd.Timestamp(validation_end))
    )
    test_mask = X_clean.index > pd.Timestamp(validation_end)

    X_train, y_train = X_clean.loc[train_mask], y_clean.loc[train_mask]
    X_validation = X_clean.loc[validation_mask]
    y_validation = y_clean.loc[validation_mask]
    X_test, y_test = X_clean.loc[test_mask], y_clean.loc[test_mask]

    if horizon > 0:
        if len(X_train) <= horizon or len(X_validation) <= horizon:
            raise ValueError("Not enough observations to purge the requested horizon.")
        X_train, y_train = X_train.iloc[:-horizon], y_train.iloc[:-horizon]
        X_validation = X_validation.iloc[:-horizon]
        y_validation = y_validation.iloc[:-horizon]

    if min(len(X_train), len(X_validation), len(X_test)) == 0:
        raise ValueError("One temporal split is empty; inspect dates and missing values.")

    return TemporalSplit(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
    )


def expanding_test_blocks(
    index: pd.DatetimeIndex,
    *,
    start: str,
    step: int = 21,
) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Yield expanding training indices and non-overlapping test blocks."""
    test_positions = [i for i, date in enumerate(index) if date >= pd.Timestamp(start)]
    if not test_positions:
        raise ValueError("No observations on or after the requested start date.")
    first = test_positions[0]
    for block_start in range(first, len(index), step):
        block_end = min(block_start + step, len(index))
        yield index[:block_start], index[block_start:block_end]