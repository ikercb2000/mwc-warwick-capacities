"""Load typed experiment settings from the repository TOML file."""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any


def _find_settings_path() -> Path:
    """Locate the settings file in a source checkout or installed distribution."""
    module_path = Path(__file__).resolve()
    candidates = (
        module_path.parents[2] / "configs" / "experiment_settings.toml",
        module_path.parents[1] / "configs" / "experiment_settings.toml",
        *(parent / "configs" / "experiment_settings.toml" for parent in (Path.cwd(), *Path.cwd().parents)),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not locate configs/experiment_settings.toml.")


SETTINGS_PATH = _find_settings_path()


def _load_settings(path: Path = SETTINGS_PATH) -> dict[str, Any]:
    """Read and parse the experiment settings TOML file."""
    with path.open("rb") as settings_file:
        return tomllib.load(settings_file)


_SETTINGS = _load_settings()

_DATES = _SETTINGS["dates"]
SAMPLE_START = str(_DATES["sample_start"])
TRAIN_END = str(_DATES["train_end"])
CLASSIFICATION_TRAIN_END = str(_DATES["classification_train_end"])
VALIDATION_END = str(_DATES["validation_end"])
SAMPLE_END = str(_DATES["sample_end"])

_FORECASTING = _SETTINGS["forecasting"]
HORIZONS = tuple(int(value) for value in _FORECASTING["horizons"])
ANNUALIZATION_FACTOR = int(_FORECASTING["annualization_factor"])

_TAIL_RISK = _SETTINGS["tail_risk"]
PRIMARY_TAIL_ALPHA = float(_TAIL_RISK["primary_alpha"])
ROBUSTNESS_TAIL_ALPHA = float(_TAIL_RISK["robustness_alpha"])
TAIL_WINDOW = int(_TAIL_RISK["rolling_window"])
MIN_TAIL_HISTORY = int(_TAIL_RISK["minimum_history"])

_COMPUTATION = _SETTINGS["computation"]
RANDOM_STATE = int(_COMPUTATION["random_state"])
QUICK_MODE_DEFAULT = bool(_COMPUTATION["quick_mode_default"])

_ASSETS = _SETTINGS["assets"]
EQUITY_TICKERS = tuple(str(value) for value in _ASSETS["equities"])
MARKET_ETFS = tuple(str(value) for value in _ASSETS["market_etfs"])
STYLE_ETFS = tuple(str(value) for value in _ASSETS["style_etfs"])
SECTOR_ETFS = tuple(str(value) for value in _ASSETS["sector_etfs"])
ETF_TICKERS = MARKET_ETFS + STYLE_ETFS + SECTOR_ETFS
SECTOR_ETF_BY_ASSET = {
    str(asset): str(etf) for asset, etf in _ASSETS["sector_etf_by_asset"].items()
}

FRED_SERIES = {
    str(series): str(variable)
    for series, variable in _SETTINGS["fred"]["series"].items()
}

_FEATURES = _SETTINGS["features"]
FACTOR_COLUMNS = tuple(str(value) for value in _FEATURES["factor_columns"])
MAIN_RISK_FEATURES = tuple(str(value) for value in _FEATURES["main_risk"])
EXTENDED_RISK_FEATURES = tuple(str(value) for value in _FEATURES["extended_risk"])