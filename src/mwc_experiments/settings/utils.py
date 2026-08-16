"""Shared settings domain."""

from __future__ import annotations
from pathlib import Path
import tomllib
from typing import Any

def _find_settings_path() -> Path:
    """Locate the settings file in a source checkout or installed distribution."""
    module_path = Path(__file__).resolve()
    candidates = (
        module_path.parents[3] / "configs" / "experiment_settings.toml",
        module_path.parents[2] / "configs" / "experiment_settings.toml",
        module_path.parents[1] / "configs" / "experiment_settings.toml",
        *(parent / "configs" / "experiment_settings.toml" for parent in (Path.cwd(), *Path.cwd().parents)),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not locate configs/experiment_settings.toml.")


def _load_settings(path: Path | None = None) -> dict[str, Any]:
    """Read and parse the experiment settings TOML file."""
    path = _find_settings_path() if path is None else path
    with path.open("rb") as settings_file:
        return tomllib.load(settings_file)
