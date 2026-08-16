"""Scalar and sequence settings loaded from TOML."""

from __future__ import annotations

from .mappings import (
    _ASSETS,
    _COMPUTATION,
    _DATES,
    _FEATURES,
    _FORECASTING,
    _TAIL_RISK,
    _VALIDATION_STRESS,
)

SAMPLE_START = str(_DATES["sample_start"])

TRAIN_END = str(_DATES["train_end"])

CLASSIFICATION_TRAIN_END = str(_DATES["classification_train_end"])

VALIDATION_END = str(_DATES["validation_end"])

SAMPLE_END = str(_DATES["sample_end"])

VALIDATION_STRESS_VALIDATION_START = str(
    _VALIDATION_STRESS["validation_start"]
)

VALIDATION_STRESS_END = str(_VALIDATION_STRESS["validation_end"])

VALIDATION_STRESS_START = str(_VALIDATION_STRESS["stress_start"])

VALIDATION_STRESS_PERIOD_END = str(_VALIDATION_STRESS["stress_end"])

HORIZONS = tuple(int(value) for value in _FORECASTING["horizons"])

ANNUALIZATION_FACTOR = int(_FORECASTING["annualization_factor"])

PRIMARY_TAIL_ALPHA = float(_TAIL_RISK["primary_alpha"])

ROBUSTNESS_TAIL_ALPHA = float(_TAIL_RISK["robustness_alpha"])

TAIL_WINDOW = int(_TAIL_RISK["rolling_window"])

MIN_TAIL_HISTORY = int(_TAIL_RISK["minimum_history"])

RANDOM_STATE = int(_COMPUTATION["random_state"])

QUICK_MODE_DEFAULT = bool(_COMPUTATION["quick_mode_default"])

EQUITY_TICKERS = tuple(str(value) for value in _ASSETS["equities"])

MARKET_ETFS = tuple(str(value) for value in _ASSETS["market_etfs"])

STYLE_ETFS = tuple(str(value) for value in _ASSETS["style_etfs"])

SECTOR_ETFS = tuple(str(value) for value in _ASSETS["sector_etfs"])

ETF_TICKERS = MARKET_ETFS + STYLE_ETFS + SECTOR_ETFS

FACTOR_COLUMNS = tuple(str(value) for value in _FEATURES["factor_columns"])

MAIN_RISK_FEATURES = tuple(str(value) for value in _FEATURES["main_risk"])

EXTENDED_RISK_FEATURES = tuple(str(value) for value in _FEATURES["extended_risk"])
