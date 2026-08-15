from __future__ import annotations

import tomllib

from mwc_experiments import settings


def test_runtime_settings_are_loaded_from_the_repository_toml() -> None:
    """Ensure exported settings preserve values declared in the TOML source."""
    with settings.SETTINGS_PATH.open("rb") as settings_file:
        raw = tomllib.load(settings_file)

    assert settings.SAMPLE_START == raw["dates"]["sample_start"]
    assert settings.HORIZONS == tuple(raw["forecasting"]["horizons"])
    assert settings.EQUITY_TICKERS == tuple(raw["assets"]["equities"])
    assert settings.FRED_SERIES == raw["fred"]["series"]
    assert settings.MAIN_RISK_FEATURES == tuple(raw["features"]["main_risk"])
    assert settings.VALIDATION_STRESS_START == raw["validation_stress"][
        "stress_start"
    ]
    assert settings.VALIDATION_STRESS_PERIOD_END == raw[
        "validation_stress"
    ]["stress_end"]
