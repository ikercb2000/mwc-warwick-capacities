"""Datasets domain."""

from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4
import pandas as pd
from capacities_ml_fin.finance import forward_losses
from mwc_experiments.configuration import load_experiment_config
from mwc_experiments.settings import (
    EQUITY_TICKERS,
    HORIZONS,
    MIN_TAIL_HISTORY,
    PRIMARY_TAIL_ALPHA,
    ROBUSTNESS_TAIL_ALPHA,
    TAIL_WINDOW,
)
from mwc_experiments.data.features import (
    factor_frame,
    prepare_market_data,
)
from mwc_experiments.data.loaders import load_raw_market_data
from mwc_experiments.data.types import (
    ExperimentData,
    PreparedMarketData,
    ProcessedExperimentData,
)
from mwc_experiments.paths import RepoPaths
from mwc_experiments.runs import atomic_write_json, sha256_file

class StaleProcessedDataError(RuntimeError):
    """Indicate that cached datasets do not match their current inputs."""
