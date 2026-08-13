from pathlib import Path

# This utility folder is expected to live directly in the repository root:
#
# repo/
# ├── fred_data_download/
# ├── data/
# ├── src/
# └── ...
#
UTILITY_DIR = Path(__file__).resolve().parent
REPO_ROOT = UTILITY_DIR.parent

# Final empirical sample used in the dissertation.
SAMPLE_START = "2014-01-01"
SAMPLE_END = "2026-07-30"

# Extra history is downloaded so lagged / YoY macro features are available
# at the beginning of the empirical sample.
DOWNLOAD_START = "2012-01-01"
DOWNLOAD_END = SAMPLE_END

RAW_FRED_DIR = REPO_ROOT / "data" / "raw" / "fred"
RAW_ALFRED_DIR = REPO_ROOT / "data" / "raw" / "alfred" / "initial_releases"
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "fred"

# Daily / market series.
DAILY_SERIES = {
    "VIXCLS": "vix",
    "DGS3MO": "treasury_3m_yield_pct",
    "DGS2": "treasury_2y_yield_pct",
    "DGS10": "treasury_10y_yield_pct",
    "BAA10Y": "baa_credit_spread_pct",
    "DFF": "fed_funds_effective_pct",
}

# Lower-frequency macro series.
# These are downloaded as initial releases via ALFRED/FRED real-time parameters
# to reduce look-ahead/revision bias in forecasting experiments.
MACRO_SERIES = {
    "CPIAUCSL": "cpi_index",
    "UNRATE": "unemployment_rate_pct",
    "INDPRO": "industrial_production_index",
}
