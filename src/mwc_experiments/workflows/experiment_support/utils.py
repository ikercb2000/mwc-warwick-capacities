"""Experiment Support domain."""

from __future__ import annotations
import argparse
import atexit
from pathlib import Path
import re
import sys
from typing import TextIO
from uuid import uuid4
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from mwc_experiments.configuration import load_experiment_config
from mwc_experiments.paths import RepoPaths
from mwc_experiments.runs import (
    ExperimentRunPaths,
    create_experiment_run,
    finish_experiment_run,
    infer_experiment_id,
)
from mwc_experiments.settings import QUICK_MODE_DEFAULT
from .types import _Tee

matplotlib.use("Agg")


_ACTIVE_RUN: ExperimentRunPaths | None = None


_RUN_FINISHED = False


_LOG_STREAM: TextIO | None = None


_ORIGINAL_STDOUT: TextIO | None = None


_ORIGINAL_STDERR: TextIO | None = None


def parse_experiment_args(
    description: str,
    *,
    experiment_id: str | None = None,
) -> argparse.Namespace:
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
    quick_default = QUICK_MODE_DEFAULT
    if experiment_id is not None:
        config = load_experiment_config(experiment_id)
        quick_default = bool(
            config.get("execution", {}).get(
                "quick_mode_default",
                quick_default,
            )
        )
    parser.set_defaults(quick=quick_default)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress model-by-model progress messages.",
    )
    return parser.parse_args()


def prepare_output_paths(
    paths: RepoPaths | None = None,
    *,
    experiment_id: str | None = None,
) -> ExperimentRunPaths:
    """Create an immutable output directory for this script execution."""
    global _ACTIVE_RUN, _RUN_FINISHED
    global _LOG_STREAM, _ORIGINAL_STDOUT, _ORIGINAL_STDERR
    resolved = RepoPaths.discover() if paths is None else paths
    resolved_experiment_id = experiment_id or infer_experiment_id()
    if "--full" in sys.argv:
        mode = "full"
    elif "--quick" in sys.argv:
        mode = "quick"
    elif resolved_experiment_id in {"factor_models", "future_loss", "tail_risk"}:
        config = load_experiment_config(resolved_experiment_id, resolved.root)
        quick_default = bool(
            config.get("execution", {}).get(
                "quick_mode_default",
                QUICK_MODE_DEFAULT,
            )
        )
        mode = "quick" if quick_default else "full"
    else:
        mode = "standard"
    _ACTIVE_RUN = create_experiment_run(
        resolved,
        experiment_id=resolved_experiment_id,
        mode=mode,
    )
    _RUN_FINISHED = False
    _ORIGINAL_STDOUT = sys.stdout
    _ORIGINAL_STDERR = sys.stderr
    _LOG_STREAM = (_ACTIVE_RUN.logs / "run.log").open(
        "a",
        encoding="utf-8",
        buffering=1,
    )
    sys.stdout = _Tee(_ORIGINAL_STDOUT, _LOG_STREAM)
    sys.stderr = _Tee(_ORIGINAL_STDERR, _LOG_STREAM)

    def mark_failed_at_exit() -> None:
        if _ACTIVE_RUN is not None and not _RUN_FINISHED:
            if _LOG_STREAM is not None:
                _LOG_STREAM.flush()
            finish_experiment_run(
                _ACTIVE_RUN,
                (_ACTIVE_RUN.logs / "run.log",),
                status="failed",
                message="Process exited before artifact publication completed.",
            )

    atexit.register(mark_failed_at_exit)
    return _ACTIVE_RUN


def artifact_slug(value: str) -> str:
    """Convert a model or panel label into a stable filename fragment."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not slug:
        raise ValueError("Artifact labels must contain at least one letter or digit.")
    return slug


def save_table(
    data: pd.DataFrame | pd.Series,
    filename: str,
    paths: RepoPaths | ExperimentRunPaths,
    *,
    index: bool = True,
) -> Path:
    """Persist a DataFrame or Series under the experiment tables directory."""
    target = paths.tables / filename
    temporary = target.with_name(
        f".{target.stem}.{uuid4().hex}{target.suffix}"
    )
    if target.suffix == ".csv":
        data.to_csv(temporary, index=index)
    elif target.suffix == ".parquet":
        frame = data.to_frame() if isinstance(data, pd.Series) else data
        frame.to_parquet(temporary, index=index)
    else:
        raise ValueError(f"Unsupported table format: {target.suffix}")
    temporary.replace(target)
    return target


def save_figure(
    figure: Figure,
    filename: str,
    paths: RepoPaths | ExperimentRunPaths,
) -> Path:
    """Persist and close a Matplotlib figure under the results figure directory."""
    target = paths.figures / filename
    temporary = target.with_name(
        f".{target.stem}.{uuid4().hex}{target.suffix}"
    )
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    temporary.replace(target)
    plt.close(figure)
    return target


def print_artifacts(artifacts: dict[str, Path]) -> None:
    """Publish a complete run and print its compact artifact catalogue."""
    global _RUN_FINISHED, _LOG_STREAM
    print(f"Generated {len(artifacts)} artifacts:")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")
    if _ACTIVE_RUN is not None:
        if _LOG_STREAM is not None:
            _LOG_STREAM.flush()
        finish_experiment_run(
            _ACTIVE_RUN,
            [*artifacts.values(), _ACTIVE_RUN.logs / "run.log"],
            status="success",
        )
        _RUN_FINISHED = True
        sys.stdout = _ORIGINAL_STDOUT or sys.__stdout__
        sys.stderr = _ORIGINAL_STDERR or sys.__stderr__
        if _LOG_STREAM is not None:
            _LOG_STREAM.close()
            _LOG_STREAM = None
