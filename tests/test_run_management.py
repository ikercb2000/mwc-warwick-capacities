from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mwc_experiments.data import StaleProcessedDataError, load_processed_data
from mwc_experiments.paths import RepoPaths
from mwc_experiments.reporting import load_result_table
from mwc_experiments.runs import (
    create_experiment_run,
    finish_experiment_run,
    publish_run_snapshot,
)
from mwc_experiments.settings import EQUITY_TICKERS


def _repository(tmp_path: Path, notebook_source: str) -> RepoPaths:
    (tmp_path / "data").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "experiments").mkdir()
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    (tmp_path / "configs" / "experiment_settings.toml").write_text(
        "[computation]\nrandom_state=42\n"
    )
    (tmp_path / "configs" / "experiments.toml").write_text(
        "[experiments.demo]\n"
        'config="configs/experiments/demo.toml"\n'
    )
    (tmp_path / "configs" / "experiments" / "demo.toml").write_text(
        "[experiment]\n"
        'id="demo"\n'
        'script="scripts/demo.py"\n'
        'notebook="notebooks/demo.ipynb"\n'
        'description="demo"\n'
    )
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": notebook_source,
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (tmp_path / "notebooks" / "demo.ipynb").write_text(
        json.dumps(notebook),
        encoding="utf-8",
    )
    return RepoPaths(tmp_path)


def test_successful_run_is_published_and_loaded_by_checksum(
    tmp_path: Path,
) -> None:
    paths = _repository(
        tmp_path,
        'load_result_table("demo_table.csv")',
    )
    run = create_experiment_run(paths, experiment_id="demo", mode="full")
    table = run.tables / "demo_table.csv"
    pd.DataFrame({"value": [1.0]}).to_csv(table, index=False)
    finish_experiment_run(run, [table], status="success")

    loaded = load_result_table(
        "demo_table.csv",
        paths=paths,
        experiment="demo",
    )

    assert loaded.loc[0, "value"] == 1.0
    assert (run.run / "SUCCESS").is_file()
    assert (run.run / "shared_config.toml").is_file()
    assert (run.run / "experiment_config.toml").is_file()
    assert (paths.latest / "demo.json").is_file()
    publication = publish_run_snapshot(paths, experiment_id="demo")
    assert len(publication["artifacts"]) == 1
    assert (paths.tables / "demo_table.csv").is_file()
    assert (paths.published / "demo.json").is_file()
    table.write_text("value\n2.0\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum"):
        load_result_table(
            "demo_table.csv",
            paths=paths,
            experiment="demo",
        )


def test_run_contract_rejects_missing_notebook_artifact(
    tmp_path: Path,
) -> None:
    paths = _repository(
        tmp_path,
        'load_result_table("required.csv")',
    )
    run = create_experiment_run(paths, experiment_id="demo", mode="quick")

    with pytest.raises(RuntimeError, match="required.csv"):
        finish_experiment_run(run, [], status="success")
    assert not (paths.latest / "demo.json").exists()


def test_processed_data_without_manifest_is_rejected(tmp_path: Path) -> None:
    paths = RepoPaths(tmp_path)
    paths.ensure_output_dirs()
    factor_dir = paths.experiments / "factor_models"
    factor_dir.mkdir()
    frame = pd.DataFrame({"value": [1.0]})
    frame.to_parquet(paths.experiments / "portfolio_equal_weight.parquet")
    frame.to_parquet(paths.experiments / "portfolio_cap_weight.parquet")
    for asset in EQUITY_TICKERS:
        frame.to_parquet(factor_dir / f"{asset}_factor_frame.parquet")

    with pytest.raises(StaleProcessedDataError, match="manifest"):
        load_processed_data(paths)
