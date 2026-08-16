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
    assert settings.ORIENTATION_MINIMUM_ABSOLUTE_CORRELATION == 0.2
    assert settings.ORIENTATION_MINIMUM_ABSOLUTE_CORRELATION == raw[
        "orientation"
    ]["minimum_absolute_correlation"]
    assert settings.ORIENTATION_STABILITY_SUBPERIODS == raw["orientation"][
        "stability_subperiods"
    ]
    assert settings.ORIENTATION_REQUIRE_SIGN_STABILITY == raw["orientation"][
        "require_sign_stability"
    ]
    assert settings.VALIDATION_STRESS_START == raw["validation_stress"][
        "stress_start"
    ]
    assert settings.VALIDATION_STRESS_PERIOD_END == raw[
        "validation_stress"
    ]["stress_end"]
