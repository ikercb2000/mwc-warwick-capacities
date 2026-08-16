from __future__ import annotations

from pathlib import Path

import pandas as pd

from mwc_experiments.settings import EQUITY_TICKERS, ETF_TICKERS, SAMPLE_END


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "data" / "raw" / "bloomberg_requirements.csv"


def test_bloomberg_requirements_match_configured_universe() -> None:
    """Keep the redistributable acquisition manifest aligned with settings."""
    requirements = pd.read_csv(REQUIREMENTS, dtype=str)

    assert requirements["ticker"].is_unique
    assert set(requirements.loc[requirements["asset_type"] == "equity", "ticker"]) == set(
        EQUITY_TICKERS
    )
    assert set(requirements.loc[requirements["asset_type"] == "etf", "ticker"]) == set(
        ETF_TICKERS
    )
    assert set(requirements["end_date"]) == {SAMPLE_END}
    assert set(requirements["period"]) == {"D"}
    assert set(requirements["currency"]) == {"USD"}


def test_bloomberg_requirements_encode_loader_fields_and_filenames() -> None:
    """Protect the exact field and filename contract consumed by the loader."""
    requirements = pd.read_csv(REQUIREMENTS, dtype=str)
    equities = requirements[requirements["asset_type"] == "equity"]
    etfs = requirements[requirements["asset_type"] == "etf"]

    assert equities["required_fields"].str.split(";").map(set).eq(
        {
            "TOT_RETURN_INDEX_GROSS_DVDS",
            "PX_LAST",
            "VOLUME",
            "CUR_MKT_CAP",
        }
    ).all()
    assert etfs["required_fields"].eq("TOT_RETURN_INDEX_GROSS_DVDS").all()
    assert all(
        path == f"market_data/equities/daily/{ticker}_daily_market_data.xlsx"
        for ticker, path in zip(equities["ticker"], equities["relative_path"])
    )
    assert all(
        path.endswith(f"/{ticker}_daily_etf_data.xlsx")
        for ticker, path in zip(etfs["ticker"], etfs["relative_path"])
    )
