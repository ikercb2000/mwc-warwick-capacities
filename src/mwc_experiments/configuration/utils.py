"""Configuration domain."""

from __future__ import annotations
from pathlib import Path
import tomllib
from typing import Any

def _repository_root(root: Path | None = None) -> Path:
    """Find the checkout containing the experiment configuration directory."""
    if root is not None:
        return Path(root).resolve()
    module_path = Path(__file__).resolve()
    candidates = (
        module_path.parents[3],
        module_path.parents[2],
        module_path.parents[1],
        Path.cwd(),
        *Path.cwd().parents,
    )
    for candidate in candidates:
        if (candidate / "configs" / "experiments").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate configs/experiments/.")


def experiment_config_path(
    experiment_id: str,
    root: Path | None = None,
) -> Path:
    """Return the canonical TOML path for one experiment."""
    return _repository_root(root) / "configs" / "experiments" / f"{experiment_id}.toml"


def load_experiment_config(
    experiment_id: str,
    root: Path | None = None,
) -> dict[str, Any]:
    """Load one experiment TOML and reject mismatched identifiers."""
    path = experiment_config_path(experiment_id, root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment configuration: {path}")
    with path.open("rb") as stream:
        payload = tomllib.load(stream)
    metadata = payload.get("experiment")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} has no [experiment] table.")
    configured_id = metadata.get("id")
    if configured_id != experiment_id:
        raise ValueError(
            f"{path} declares experiment id {configured_id!r}, expected {experiment_id!r}."
        )
    for field in ("script", "notebook", "description"):
        if not isinstance(metadata.get(field), str) or not metadata[field]:
            raise ValueError(f"{path} requires experiment.{field}.")
    return payload


def parameter_grid_overrides(
    config: dict[str, Any],
) -> dict[str, dict[str, list[Any]]]:
    """Return TOML grids with explicit sentinels converted to Python values."""
    raw_grids = config.get("parameter_grids", {})
    if not isinstance(raw_grids, dict):
        raise TypeError("parameter_grids must be a TOML table.")

    def normalize(value: Any) -> Any:
        if value == "__none__":
            return None
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    grids: dict[str, dict[str, list[Any]]] = {}
    for model, raw_grid in raw_grids.items():
        if not isinstance(raw_grid, dict):
            raise TypeError(f"Grid for {model!r} must be a TOML table.")
        grids[str(model)] = {}
        for parameter, values in raw_grid.items():
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"Grid {model!r}/{parameter!r} must be a non-empty array."
                )
            grids[str(model)][str(parameter)] = normalize(values)
    return grids
