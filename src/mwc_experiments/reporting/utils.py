"""Reporting domain."""

from __future__ import annotations
from pathlib import Path
import json
import os
from typing import Any
import pandas as pd
from mwc_experiments.paths import RepoPaths
from mwc_experiments.runs import infer_artifact_experiment, sha256_file

def result_artifact_path(
    filename: str,
    *,
    kind: str,
    paths: RepoPaths,
    experiment: str | None,
    run_id: str | None,
) -> Path:
    """Resolve an explicit or latest-successful run, with legacy fallback."""
    experiment_id = experiment or infer_artifact_experiment(filename)
    selected_run = run_id
    if experiment_id is not None and selected_run is None:
        environment_name = f"MWC_RUN_{experiment_id.upper()}"
        selected_run = os.environ.get(environment_name)
    if experiment_id is not None and selected_run is None:
        pointer = paths.latest / f"{experiment_id}.json"
        if pointer.is_file():
            selected_run = json.loads(pointer.read_text(encoding="utf-8"))[
                "run_id"
            ]
    if selected_run is not None:
        if experiment_id is None:
            raise ValueError("experiment is required when run_id cannot be inferred.")
        run = paths.runs / experiment_id / selected_run
        manifest = run / "manifest.json"
        success = run / "SUCCESS"
        if not manifest.is_file() or not success.is_file():
            raise FileNotFoundError(
                f"Run {experiment_id}/{selected_run} is not complete."
            )
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        status = payload.get("status")
        if status != "success":
            raise RuntimeError(
                f"Run {experiment_id}/{selected_run} has status {status!r}."
            )
        target = run / kind / filename
        records = {
            (paths.root / record["path"]).resolve(): record["sha256"]
            for record in payload.get("artifacts", [])
        }
        if target.resolve() not in records:
            raise RuntimeError(
                f"Artifact {filename} is not registered in the run manifest."
            )
        if target.is_file() and sha256_file(target) != records[target.resolve()]:
            raise RuntimeError(f"Artifact checksum mismatch: {target}")
        return target
    legacy = paths.tables if kind == "tables" else paths.figures
    return legacy / filename


def successful_runs_by_mode(
    experiment: str,
    *,
    paths: RepoPaths | None = None,
) -> dict[str, str]:
    """Return the newest successful immutable run for every execution mode."""
    resolved = RepoPaths.discover() if paths is None else paths
    selected: dict[str, tuple[str, str]] = {}
    for manifest_path in resolved.runs.glob(f"{experiment}/*/manifest.json"):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "success":
            continue
        run_id = str(payload["run_id"])
        mode = str(payload["mode"])
        started = str(payload.get("started_at_utc", ""))
        if mode not in selected or started > selected[mode][0]:
            selected[mode] = (started, run_id)
    return {mode: item[1] for mode, item in selected.items()}


def load_result_comparison(
    filename: str,
    *,
    experiment: str | None = None,
    modes: tuple[str, ...] = (
        "full_fixed_clipping",
        "full_fixed_no_clipping",
        "full_rolling_5y_clipping",
        "full_rolling_5y_no_clipping",
    ),
    paths: RepoPaths | None = None,
    **read_options: Any,
) -> pd.DataFrame:
    """Load comparable modes without ever pooling them as one test sample."""
    resolved = RepoPaths.discover() if paths is None else paths
    experiment_id = experiment or infer_artifact_experiment(filename)
    if experiment_id is None:
        raise ValueError("Could not infer the experiment for comparison.")
    available = successful_runs_by_mode(experiment_id, paths=resolved)
    frames: list[pd.DataFrame] = []
    for mode in modes:
        run_id = available.get(mode)
        if run_id is None:
            continue
        frame = load_result_table(
            filename,
            paths=resolved,
            experiment=experiment_id,
            run_id=run_id,
            **read_options,
        ).reset_index()
        if "evaluation_structure" not in frame:
            frame["evaluation_structure"] = (
                "rolling_5y" if "rolling_5y" in mode else "fixed"
            )
        frame["clipping"] = not mode.endswith("no_clipping")
        frame["run_mode"] = mode
        frame["run_id"] = run_id
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No successful comparison runs are available for {experiment_id}."
        )
    return pd.concat(frames, ignore_index=True, sort=False)

def load_result_table(
    filename: str,
    *,
    paths: RepoPaths | None = None,
    experiment: str | None = None,
    run_id: str | None = None,
    **read_options: Any,
) -> pd.DataFrame:
    """Load a CSV or Parquet result table for display without recomputation."""
    resolved = RepoPaths.discover() if paths is None else paths
    target = result_artifact_path(
        filename,
        kind="tables",
        paths=resolved,
        experiment=experiment,
        run_id=run_id,
    )
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
    experiment: str | None = None,
    run_id: str | None = None,
) -> str:
    """Return a validated figure filename suitable for IPython Image display."""
    resolved = RepoPaths.discover() if paths is None else paths
    target = result_artifact_path(
        filename,
        kind="figures",
        paths=resolved,
        experiment=experiment,
        run_id=run_id,
    )
    if not target.is_file():
        raise FileNotFoundError(
            f"Missing result figure {target}. Run the corresponding experiment script first."
        )
    return str(target)


def audit_notebook_artifacts(
    *,
    paths: RepoPaths | None = None,
) -> pd.DataFrame:
    """Check every catalogued notebook reference against published results."""
    from mwc_experiments.runs import (
        experiment_catalog,
        notebook_artifact_references,
    )

    resolved = RepoPaths.discover() if paths is None else paths
    rows: list[dict[str, object]] = []
    for experiment_id, specification in experiment_catalog(
        resolved.root
    ).items():
        notebook = resolved.root / specification["notebook"]
        for filename in sorted(notebook_artifact_references(notebook)):
            kind = "figures" if Path(filename).suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".svg",
            } else "tables"
            try:
                target = result_artifact_path(
                    filename,
                    kind=kind,
                    paths=resolved,
                    experiment=experiment_id,
                    run_id=None,
                )
                exists = target.is_file()
                message = "" if exists else f"Missing {target}"
            except Exception as error:
                target = None
                exists = False
                message = f"{type(error).__name__}: {error}"
            rows.append(
                {
                    "experiment": experiment_id,
                    "notebook": specification["notebook"],
                    "artifact": filename,
                    "exists": exists,
                    "resolved path": str(target) if target else "",
                    "message": message,
                }
            )
    return pd.DataFrame(rows)


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
