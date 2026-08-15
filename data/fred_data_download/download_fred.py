from __future__ import annotations

from dotenv import load_dotenv

from config import (
    DAILY_SERIES,
    DOWNLOAD_END,
    DOWNLOAD_START,
    MACRO_SERIES,
    RAW_ALFRED_DIR,
    RAW_FRED_DIR,
    REPO_ROOT,
)
from fred_client import (
    download_initial_release_series,
    download_standard_series,
)


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")

    RAW_FRED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ALFRED_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading daily FRED series...")
    for series_id, variable_name in DAILY_SERIES.items():
        df = download_standard_series(
            series_id=series_id,
            start=DOWNLOAD_START,
            end=DOWNLOAD_END,
        )
        out = RAW_FRED_DIR / f"{series_id}_{variable_name}.csv"
        df.to_csv(out, index=False)
        print(f"  {series_id}: {len(df):,} rows -> {out}")

    print("\nDownloading initial-release macro observations...")
    for series_id, variable_name in MACRO_SERIES.items():
        df = download_initial_release_series(
            series_id=series_id,
            start=DOWNLOAD_START,
            end=DOWNLOAD_END,
        )
        out = RAW_ALFRED_DIR / f"{series_id}_{variable_name}_initial.csv"
        df.to_csv(out, index=False)
        print(f"  {series_id}: {len(df):,} rows -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
