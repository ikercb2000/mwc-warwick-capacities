"""Dictionary mappings derived from shared settings."""

from __future__ import annotations

from .utils import _find_settings_path, _load_settings

SETTINGS_PATH = _find_settings_path()

_SETTINGS = _load_settings(SETTINGS_PATH)
_DATES = _SETTINGS["dates"]
_VALIDATION_STRESS = _SETTINGS["validation_stress"]
_FORECASTING = _SETTINGS["forecasting"]
_TAIL_RISK = _SETTINGS["tail_risk"]
_COMPUTATION = _SETTINGS["computation"]
_ASSETS = _SETTINGS["assets"]
_FEATURES = _SETTINGS["features"]

SECTOR_ETF_BY_ASSET = {
    str(asset): str(etf) for asset, etf in _ASSETS["sector_etf_by_asset"].items()
}

FRED_SERIES = {
    str(series): str(variable)
    for series, variable in _SETTINGS["fred"]["series"].items()
}
