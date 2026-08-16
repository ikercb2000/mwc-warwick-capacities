"""Splits domain."""

from __future__ import annotations
from typing import Iterator
import pandas as pd
from mwc_experiments.settings import TRAIN_END, VALIDATION_END
from mwc_experiments.modeling.types import TemporalSplit, WalkForwardFold


def _purge_partition(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Drop targets whose forward horizon crosses the next boundary."""
    if horizon == 0:
        return X, y
    if len(X) <= horizon:
        raise ValueError("Not enough observations to purge the requested horizon.")
    return X.iloc[:-horizon], y.iloc[:-horizon]

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


def rolling_walk_forward_splits(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    oos_start: str,
    training_window_years: int = 5,
    validation_window_months: int = 12,
    oos_block_years: int = 1,
    horizon: int = 0,
) -> Iterator[WalkForwardFold]:
    """Yield rolling windows with internal validation and purged boundaries.

    The lookback window ends immediately before each OOS block. Its final
    ``validation_window_months`` are held out for hyperparameter selection;
    the remaining earlier observations form the inner training partition.
    Both boundaries are purged by the forecast horizon.
    """
    integer_settings = {
        "training_window_years": training_window_years,
        "validation_window_months": validation_window_months,
        "oos_block_years": oos_block_years,
    }
    for name, value in integer_settings.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if horizon < 0:
        raise ValueError("horizon must be non-negative.")
    if validation_window_months >= 12 * training_window_years:
        raise ValueError(
            "Validation must be shorter than the complete training window."
        )
    if not isinstance(X.index, pd.DatetimeIndex):
        raise TypeError("Walk-forward evaluation requires a DatetimeIndex.")

    common = X.join(y.rename("__target__"), how="inner").dropna().sort_index()
    X_clean = common[X.columns]
    y_clean = common["__target__"]
    first_oos = pd.Timestamp(oos_start)
    if first_oos > X_clean.index.max():
        raise ValueError("oos_start is after the available sample.")

    fold_number = 0
    block_start = first_oos
    while block_start <= X_clean.index.max():
        block_end = block_start + pd.DateOffset(years=oos_block_years)
        window_start = block_start - pd.DateOffset(years=training_window_years)
        validation_start = block_start - pd.DateOffset(
            months=validation_window_months
        )
        train_mask = (
            (X_clean.index >= window_start)
            & (X_clean.index < validation_start)
        )
        validation_mask = (
            (X_clean.index >= validation_start)
            & (X_clean.index < block_start)
        )
        test_mask = (
            (X_clean.index >= block_start)
            & (X_clean.index < block_end)
        )
        X_test = X_clean.loc[test_mask]
        if X_test.empty:
            block_start = block_end
            continue
        X_train, y_train = _purge_partition(
            X_clean.loc[train_mask],
            y_clean.loc[train_mask],
            horizon,
        )
        X_validation, y_validation = _purge_partition(
            X_clean.loc[validation_mask],
            y_clean.loc[validation_mask],
            horizon,
        )
        if min(len(X_train), len(X_validation)) == 0:
            raise ValueError(
                f"Walk-forward fold {fold_number} contains an empty partition."
            )
        yield WalkForwardFold(
            fold=fold_number,
            window_start=window_start,
            validation_start=validation_start,
            oos_start=block_start,
            oos_end=block_end,
            split=TemporalSplit(
                X_train=X_train,
                y_train=y_train,
                X_validation=X_validation,
                y_validation=y_validation,
                X_test=X_test,
                y_test=y_clean.loc[test_mask],
            ),
        )
        fold_number += 1
        block_start = block_end


def aggregate_walk_forward_split(
    folds: list[WalkForwardFold],
) -> TemporalSplit:
    """Expose the final estimation window and concatenated OOS sample."""
    if not folds:
        raise ValueError("At least one walk-forward fold is required.")
    final = folds[-1].split
    return TemporalSplit(
        X_train=final.X_train,
        y_train=final.y_train,
        X_validation=final.X_validation,
        y_validation=final.y_validation,
        X_test=pd.concat([fold.split.X_test for fold in folds]).sort_index(),
        y_test=pd.concat([fold.split.y_test for fold in folds]).sort_index(),
    )


def walk_forward_fold_summary(
    folds: list[WalkForwardFold],
) -> pd.DataFrame:
    """Return auditable dates and observation counts for every rolling fold."""
    rows: list[dict[str, object]] = []
    for fold in folds:
        split = fold.split
        rows.append(
            {
                "fold": fold.fold,
                "window start": fold.window_start,
                "train start": split.X_train.index.min(),
                "train end": split.X_train.index.max(),
                "train observations": len(split.X_train),
                "validation start": split.X_validation.index.min(),
                "validation end": split.X_validation.index.max(),
                "validation observations": len(split.X_validation),
                "OOS start": split.X_test.index.min(),
                "OOS end": split.X_test.index.max(),
                "OOS observations": len(split.X_test),
            }
        )
    return pd.DataFrame(rows).set_index("fold")


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
