from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DAILY_SERIES,
    MACRO_SERIES,
    PROCESSED_DIR,
    RAW_ALFRED_DIR,
    RAW_FRED_DIR,
    SAMPLE_END,
    SAMPLE_START,
)


def _load_daily_series(series_id: str, variable_name: str) -> pd.DataFrame:
    path = RAW_FRED_DIR / f"{series_id}_{variable_name}.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df.rename(columns={"value": variable_name})


def _load_macro_initial(series_id: str, variable_name: str) -> pd.DataFrame:
    path = RAW_ALFRED_DIR / f"{series_id}_{variable_name}_initial.csv"
    df = pd.read_csv(
        path,
        parse_dates=["observation_date", "available_date"],
    )
    return df.rename(columns={"value": variable_name})


def _default_business_calendar() -> pd.DatetimeIndex:
    return pd.date_range(SAMPLE_START, SAMPLE_END, freq="B")


def _calendar_from_file(path: Path, date_column: str) -> pd.DatetimeIndex:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path, columns=[date_column])
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, usecols=[date_column])
    else:
        raise ValueError("Calendar file must be .parquet or .csv")

    dates = pd.to_datetime(df[date_column]).dropna().drop_duplicates().sort_values()
    dates = dates[(dates >= SAMPLE_START) & (dates <= SAMPLE_END)]
    return pd.DatetimeIndex(dates)


def _align_market_data(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    panel = pd.DataFrame({"date": calendar}).sort_values("date")

    for series_id, variable_name in DAILY_SERIES.items():
        df = _load_daily_series(series_id, variable_name).dropna(subset=[variable_name])
        df = df.sort_values("date")

        panel = pd.merge_asof(
            panel,
            df[["date", variable_name]],
            on="date",
            direction="backward",
        )

    return panel


def _align_macro_initial(
    panel: pd.DataFrame,
    series_id: str,
    variable_name: str,
) -> pd.DataFrame:
    macro = _load_macro_initial(series_id, variable_name)
    macro = macro.dropna(subset=[variable_name, "available_date"])
    macro = macro.sort_values("available_date")

    macro = macro[
        ["available_date", "observation_date", variable_name]
    ].rename(
        columns={
            "observation_date": f"{series_id.lower()}_observation_date",
            "available_date": f"{series_id.lower()}_available_date",
        }
    )

    return pd.merge_asof(
        panel.sort_values("date"),
        macro.sort_values(f"{series_id.lower()}_available_date"),
        left_on="date",
        right_on=f"{series_id.lower()}_available_date",
        direction="backward",
    )


def build_features(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    df = _align_market_data(calendar)

    for series_id, variable_name in MACRO_SERIES.items():
        df = _align_macro_initial(df, series_id, variable_name)

    # Yield-curve and rate features.
    df["term_spread_10y_2y_pct"] = (
        df["treasury_10y_yield_pct"] - df["treasury_2y_yield_pct"]
    )

    df["risk_free_daily_simple"] = (
        (1.0 + df["treasury_3m_yield_pct"] / 100.0) ** (1.0 / 252.0) - 1.0
    )
    df["risk_free_daily_log"] = np.log1p(df["risk_free_daily_simple"])

    # Daily changes.
    df["vix_change_1d"] = df["vix"].diff()
    df["treasury_2y_change_1d_bps"] = df["treasury_2y_yield_pct"].diff() * 100.0
    df["treasury_10y_change_1d_bps"] = df["treasury_10y_yield_pct"].diff() * 100.0
    df["term_spread_change_1d_bps"] = df["term_spread_10y_2y_pct"].diff() * 100.0
    df["baa_credit_spread_change_1d_bps"] = df["baa_credit_spread_pct"].diff() * 100.0
    df["fed_funds_change_1d_bps"] = df["fed_funds_effective_pct"].diff() * 100.0

    # Point-in-time macro transformations.
    # Because the monthly values have already been aligned according to their
    # publication/availability dates, these changes only become visible after release.
    cpi_monthly = (
        df[
            ["cpiaucsl_observation_date", "cpi_index"]
        ]
        .dropna()
        .drop_duplicates("cpiaucsl_observation_date")
        .sort_values("cpiaucsl_observation_date")
    )
    cpi_monthly["inflation_yoy_pct"] = (
        cpi_monthly["cpi_index"].pct_change(12, fill_method=None) * 100.0
    )
    df = df.merge(
        cpi_monthly[["cpiaucsl_observation_date", "inflation_yoy_pct"]],
        on="cpiaucsl_observation_date",
        how="left",
    )

    unemployment_monthly = (
        df[
            ["unrate_observation_date", "unemployment_rate_pct"]
        ]
        .dropna()
        .drop_duplicates("unrate_observation_date")
        .sort_values("unrate_observation_date")
    )
    unemployment_monthly["unemployment_change_1m_pp"] = (
        unemployment_monthly["unemployment_rate_pct"].diff()
    )
    df = df.merge(
        unemployment_monthly[
            ["unrate_observation_date", "unemployment_change_1m_pp"]
        ],
        on="unrate_observation_date",
        how="left",
    )

    indpro_monthly = (
        df[
            ["indpro_observation_date", "industrial_production_index"]
        ]
        .dropna()
        .drop_duplicates("indpro_observation_date")
        .sort_values("indpro_observation_date")
    )
    indpro_monthly["industrial_production_yoy_pct"] = (
        indpro_monthly["industrial_production_index"]
        .pct_change(12, fill_method=None)
        * 100.0
    )
    indpro_monthly["industrial_production_mom_pct"] = (
        indpro_monthly["industrial_production_index"]
        .pct_change(fill_method=None)
        * 100.0
    )
    df = df.merge(
        indpro_monthly[
            [
                "indpro_observation_date",
                "industrial_production_yoy_pct",
                "industrial_production_mom_pct",
            ]
        ],
        on="indpro_observation_date",
        how="left",
    )

    # Re-propagate transformed macro values until the next release.
    macro_feature_cols = [
        "inflation_yoy_pct",
        "unemployment_change_1m_pp",
        "industrial_production_yoy_pct",
        "industrial_production_mom_pct",
    ]
    df[macro_feature_cols] = df[macro_feature_cols].ffill()

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calendar",
        type=Path,
        default=None,
        help=(
            "Optional .parquet/.csv file whose date column defines the exact "
            "equity trading calendar."
        ),
    )
    parser.add_argument(
        "--calendar-date-column",
        default="date",
    )
    args = parser.parse_args()

    if args.calendar is None:
        calendar = _default_business_calendar()
        print("Using a Monday-Friday business-day calendar.")
    else:
        calendar = _calendar_from_file(args.calendar, args.calendar_date_column)
        print(f"Using trading calendar from {args.calendar}")

    df = build_features(calendar)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "fred_daily_features.parquet"
    df.to_parquet(out, index=False)

    print(f"Saved {len(df):,} rows -> {out}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")


if __name__ == "__main__":
    main()
