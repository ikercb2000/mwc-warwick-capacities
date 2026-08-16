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

class _Tee:
    """Mirror terminal output into a persistent run log."""

    def __init__(self, terminal: TextIO, log: TextIO) -> None:
        self.terminal = terminal
        self.log = log

    def write(self, value: str) -> int:
        self.terminal.write(value)
        self.log.write(value)
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()
