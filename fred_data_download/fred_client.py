from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests


BASE_URL = "https://api.stlouisfed.org/fred"


def get_api_key() -> str:
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Copy .env.example to .env, add your key, "
            "and load it before running the scripts."
        )
    return key


def _request(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    request_params = {
        **params,
        "api_key": get_api_key(),
        "file_type": "json",
    }
    response = requests.get(
        f"{BASE_URL}/{endpoint}",
        params=request_params,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def download_standard_series(
    series_id: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    payload = _request(
        "series/observations",
        {
            "series_id": series_id,
            "observation_start": start,
            "observation_end": end,
        },
    )

    df = pd.DataFrame(payload["observations"])
    df = df[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def download_initial_release_series(
    series_id: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download the first published value for each observation.

    FRED/ALFRED output_type=4 requests initial-release observations.
    realtime_start is retained as the date from which the value was available.
    """
    payload = _request(
        "series/observations",
        {
            "series_id": series_id,
            "observation_start": start,
            "observation_end": end,
            "output_type": 4,
        },
    )

    df = pd.DataFrame(payload["observations"])
    keep = ["date", "realtime_start", "value"]
    df = df[keep].copy()

    df["date"] = pd.to_datetime(df["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df.rename(
        columns={
            "date": "observation_date",
            "realtime_start": "available_date",
        }
    )
