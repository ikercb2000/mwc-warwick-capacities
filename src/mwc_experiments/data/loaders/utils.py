"""Loaders domain."""

from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from mwc_experiments.settings import (
    SAMPLE_END,
    SAMPLE_START,
    EQUITY_TICKERS,
    ETF_TICKERS,
    FRED_SERIES,
)
from mwc_experiments.data.types import RawMarketData
from mwc_experiments.paths import RepoPaths
from .mappings import COLUMN_RENAMES

def _header_row(path: Path) -> int:
    """Locate the spreadsheet row containing Bloomberg's date header."""
    preview = pd.read_excel(path, header=None, nrows=20)
    first_column = preview.iloc[:, 0].astype(str).str.strip().str.lower()
    matches = np.flatnonzero(first_column.eq("date").to_numpy())
    if matches.size == 0:
        raise ValueError(f"Could not find a Date header in {path}.")
    return int(matches[0])


def _parse_dates(values: pd.Series) -> pd.DatetimeIndex:
    """Parse Bloomberg dates from datetimes, strings or Excel serial values."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))

    numeric = pd.to_numeric(values, errors="coerce")
    numeric_share = float(numeric.notna().mean())
    finite = numeric.dropna()
    # Datetime objects converted to integers are nanoseconds and must not be
    # interpreted with an Excel origin. We try to detect excel serials by
    # checking that the median is in the range of 10k-100k days since 1899-12-30
    looks_like_excel_serial = (
        numeric_share > 0.9
        and not finite.empty
        and finite.median() > 10_000
        and finite.median() < 100_000
    )
    if looks_like_excel_serial:
        parsed = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
    else:
        parsed = pd.to_datetime(values, errors="coerce")
    return pd.DatetimeIndex(parsed)


def read_bloomberg_workbook(path: str | Path) -> pd.DataFrame:
    """Read one Bloomberg workbook and standardise field names and dates."""
    path = Path(path)
    frame = pd.read_excel(path, header=_header_row(path))
    frame = frame.dropna(how="all").copy()
    if "Date" not in frame.columns:
        raise ValueError(f"Workbook {path} has no Date column after parsing.")
    frame["Date"] = _parse_dates(frame["Date"])
    frame = frame.dropna(subset=["Date"]).set_index("Date").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.rename(columns=COLUMN_RENAMES)
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.index.name = "date"
    return frame


def _stack_fields(frames: dict[str, pd.DataFrame], tickers: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Combine per-ticker frames into one aligned panel for each field."""
    fields = sorted({column for ticker in tickers for column in frames[ticker].columns})
    result: dict[str, pd.DataFrame] = {}
    for field in fields:
        result[field] = pd.concat(
            {ticker: frames[ticker][field] for ticker in tickers if field in frames[ticker]},
            axis=1,
        ).sort_index()
        result[field].columns.name = "ticker"
    return result


def _load_fred(paths: RepoPaths, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Load FRED series and carry observations forward onto the equity calendar."""
    columns: dict[str, pd.Series] = {}
    for series_id, variable_name in FRED_SERIES.items():
        matches = sorted(paths.fred_raw.glob(f"{series_id}_*.csv"))
        if not matches:
            raise FileNotFoundError(
                f"Missing FRED series {series_id}. Expected a CSV in {paths.fred_raw}."
            )
        raw = pd.read_csv(matches[0])
        if not {"date", "value"}.issubset(raw.columns):
            raise ValueError(f"Unexpected FRED CSV format in {matches[0]}.")
        series = pd.Series(
            pd.to_numeric(raw["value"], errors="coerce").to_numpy(),
            index=pd.to_datetime(raw["date"], errors="coerce"),
            name=variable_name,
        ).sort_index()
        series = series[~series.index.duplicated(keep="last")]
        expanded = series.reindex(series.index.union(calendar)).sort_index().ffill()
        columns[variable_name] = expanded.reindex(calendar)
    panel = pd.concat(columns.values(), axis=1)
    panel.index.name = "date"
    return panel


def load_raw_market_data(
    paths: RepoPaths | None = None,
    *,
    sample_start: str = SAMPLE_START,
    sample_end: str = SAMPLE_END,
) -> RawMarketData:
    """Load and align all usable Bloomberg and FRED data in the repository."""
    paths = RepoPaths.discover() if paths is None else paths

    equity_dir = paths.bloomberg_raw / "market_data" / "equities" / "daily"
    etf_dir = paths.bloomberg_raw / "market_data" / "etfs" / "daily"

    equity_frames: dict[str, pd.DataFrame] = {}
    for ticker in EQUITY_TICKERS:
        path = equity_dir / f"{ticker}_daily_market_data.xlsx"
        if not path.exists():
            raise FileNotFoundError(path)
        equity_frames[ticker] = read_bloomberg_workbook(path)

    etf_frames: dict[str, pd.DataFrame] = {}
    for ticker in ETF_TICKERS:
        matches = sorted(etf_dir.rglob(f"{ticker}_daily_etf_data.xlsx"))
        if not matches:
            raise FileNotFoundError(f"Could not locate {ticker}_daily_etf_data.xlsx")
        etf_frames[ticker] = read_bloomberg_workbook(matches[0])

    equity_fields = _stack_fields(equity_frames, EQUITY_TICKERS)
    etf_fields = _stack_fields(etf_frames, ETF_TICKERS)

    calendar = equity_fields["total_return_index"].dropna(how="any").index
    calendar = calendar[(calendar >= sample_start) & (calendar <= sample_end)]

    equity_fields = {key: value.reindex(calendar) for key, value in equity_fields.items()}
    etf_fields = {key: value.reindex(calendar) for key, value in etf_fields.items()}
    fred = _load_fred(paths, calendar)

    return RawMarketData(
        equity_fields=equity_fields,
        etf_fields=etf_fields,
        fred=fred,
        calendar=calendar,
    )
