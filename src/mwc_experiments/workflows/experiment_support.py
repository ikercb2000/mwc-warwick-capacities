"""Shared command-line and artifact helpers for complete experiments."""

from __future__ import annotations

# These helpers are shared by the executable experiment scripts.

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

from mwc_experiments.paths import RepoPaths
from mwc_experiments.settings import QUICK_MODE_DEFAULT


def parse_experiment_args(description: str) -> argparse.Namespace:
    """Parse the common quick/full and verbosity switches for an experiment."""
    parser = argparse.ArgumentParser(description=description)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        dest="quick",
        help="Use one representative hyperparameter configuration per model family.",
    )
    mode.add_argument(
        "--full",
        action="store_false",
        dest="quick",
        help="Use the complete validation grids.",
    )
    parser.set_defaults(quick=QUICK_MODE_DEFAULT)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress model-by-model progress messages.",
    )
    return parser.parse_args()


def prepare_output_paths(paths: RepoPaths | None = None) -> RepoPaths:
    """Resolve repository paths and ensure all artifact directories exist."""
    resolved = RepoPaths.discover() if paths is None else paths
    resolved.ensure_output_dirs()
    return resolved


def artifact_slug(value: str) -> str:
    """Convert a model or panel label into a stable filename fragment."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not slug:
        raise ValueError("Artifact labels must contain at least one letter or digit.")
    return slug


def save_table(
    data: pd.DataFrame | pd.Series,
    filename: str,
    paths: RepoPaths,
    *,
    index: bool = True,
) -> Path:
    """Persist a DataFrame or Series under the experiment tables directory."""
    target = paths.tables / filename
    if target.suffix == ".csv":
        data.to_csv(target, index=index)
    elif target.suffix == ".parquet":
        frame = data.to_frame() if isinstance(data, pd.Series) else data
        frame.to_parquet(target, index=index)
    else:
        raise ValueError(f"Unsupported table format: {target.suffix}")
    return target


def save_figure(figure: Figure, filename: str, paths: RepoPaths) -> Path:
    """Persist and close a Matplotlib figure under the results figure directory."""
    target = paths.figures / filename
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return target


def print_artifacts(artifacts: dict[str, Path]) -> None:
    """Print a compact catalogue of artifacts produced by an experiment."""
    print(f"Generated {len(artifacts)} artifacts:")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")
