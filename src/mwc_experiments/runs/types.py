"""Runs domain."""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import ast
import tomllib
from typing import Iterable
from uuid import uuid4
from mwc_experiments.configuration import (
    experiment_config_path,
    load_experiment_config,
)
from mwc_experiments.paths import RepoPaths

@dataclass(frozen=True, slots=True)
class ExperimentRunPaths:
    """Expose shared inputs and run-specific immutable output directories."""

    repository: RepoPaths
    experiment_id: str
    run_id: str
    mode: str

    @property
    def root(self) -> Path:
        return self.repository.root

    @property
    def run(self) -> Path:
        return self.repository.runs / self.experiment_id / self.run_id

    @property
    def tables(self) -> Path:
        return self.run / "tables"

    @property
    def figures(self) -> Path:
        return self.run / "figures"

    @property
    def models(self) -> Path:
        return self.run / "models"

    @property
    def logs(self) -> Path:
        return self.run / "logs"

    @property
    def results(self) -> Path:
        return self.run

    @property
    def experiments(self) -> Path:
        return self.repository.experiments

    @property
    def bloomberg_raw(self) -> Path:
        return self.repository.bloomberg_raw

    @property
    def fred_raw(self) -> Path:
        return self.repository.fred_raw

    @property
    def manifest(self) -> Path:
        return self.run / "manifest.json"

    def ensure_output_dirs(self) -> None:
        self.repository.ensure_output_dirs()
        for path in (self.tables, self.figures, self.models, self.logs):
            path.mkdir(parents=True, exist_ok=True)
