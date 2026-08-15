from dotenv import load_dotenv

import download_fred
import build_daily_features
from config import REPO_ROOT


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    download_fred.main()

    # Runs with a generic Mon-Fri calendar.
    # For the final dissertation dataset, prefer:
    #
    # python build_daily_features.py \
    #   --calendar ../experiments/portfolio_equal_weight.parquet
    #
    build_daily_features.main()


if __name__ == "__main__":
    main()
