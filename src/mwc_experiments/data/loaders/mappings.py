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

COLUMN_RENAMES: dict[str, str] = {
    "TOT_RETURN_INDEX_GROSS_DVDS": "total_return_index",
    "PX_LAST": "px_last",
    "PX_OPEN": "px_open",
    "PX_HIGH": "px_high",
    "PX_LOW": "px_low",
    "VOLUME": "volume",
    "CUR_MKT_CAP": "market_cap",
}
