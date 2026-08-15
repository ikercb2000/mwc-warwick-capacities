"""Load persisted experiment artifacts for lightweight notebook reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mwc_experiments.paths import RepoPaths


def load_result_table(
    filename: str,
    *,
    paths: RepoPaths | None = None,
    **read_options: Any,
) -> pd.DataFrame:
    """Load a CSV or Parquet result table for display without recomputation."""
    resolved = RepoPaths.discover() if paths is None else paths
    target = resolved.tables / filename
    if not target.is_file():
        raise FileNotFoundError(
            f"Missing result table {target}. Run the corresponding experiment script first."
        )
    if target.suffix == ".csv":
        return pd.read_csv(target, **read_options)
    if target.suffix == ".parquet":
        return pd.read_parquet(target, **read_options)
    raise ValueError(f"Unsupported result-table format: {target.suffix}")


def result_figure_path(
    filename: str,
    *,
    paths: RepoPaths | None = None,
) -> str:
    """Return a validated figure filename suitable for IPython Image display."""
    resolved = RepoPaths.discover() if paths is None else paths
    target = resolved.figures / filename
    if not target.is_file():
        raise FileNotFoundError(
            f"Missing result figure {target}. Run the corresponding experiment script first."
        )
    return str(target)


def experiment_data_path(
    filename: str,
    *,
    paths: RepoPaths | None = None,
) -> Path:
    """Return a validated path to a persisted model-ready experiment dataset."""
    resolved = RepoPaths.discover() if paths is None else paths
    target = resolved.experiments / filename
    if not target.is_file():
        raise FileNotFoundError(
            f"Missing processed dataset {target}. Run scripts/build_experiment_data.py first."
        )
    return target