from dotenv import load_dotenv

import download_fred
import build_daily_features


def main() -> None:
    load_dotenv()
    download_fred.main()

    # Runs with a generic Mon-Fri calendar.
    # For the final dissertation dataset, prefer:
    #
    # python build_daily_features.py \
    #   --calendar ../data/processed/equities_daily.parquet
    #
    build_daily_features.main()


if __name__ == "__main__":
    main()
