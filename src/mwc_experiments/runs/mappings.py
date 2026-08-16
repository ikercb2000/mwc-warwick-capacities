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

SCRIPT_EXPERIMENTS = {
    "build_experiment_data": "data_preparation",
    "experiment_factors": "factor_models",
    "experiment_predict_loss": "future_loss",
    "experiment_tail_risk": "tail_risk",
    "experiment_distortion_risk": "distortion_risk",
    "experiment_autoregression": "autoregression",
}

ARTIFACT_EXPERIMENTS = {
    "data_": "data_preparation",
    "experiment_1_": "factor_models",
    "experiment_2a_": "future_loss",
    "experiment_2b_": "tail_risk",
    "experiment_3_": "distortion_risk",
    "robustness_ar_": "autoregression",
}
