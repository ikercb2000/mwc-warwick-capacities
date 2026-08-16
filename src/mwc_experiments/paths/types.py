"""Paths domain."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True, slots=True)
class RepoPaths:
    """Repository paths used by the dissertation experiment code."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    @classmethod
    def discover(cls, start: str | Path | None = None) -> "RepoPaths":
        current = Path.cwd() if start is None else Path(start)
        current = current.resolve()
        candidates = (current, *current.parents)
        for candidate in candidates:
            if (candidate / "pyproject.toml").exists() and (candidate / "data").exists():
                return cls(candidate)
        raise FileNotFoundError(
            "Could not locate the repository root. Run from inside the repository "
            "or pass a path beneath a directory containing pyproject.toml and data/."
        )

    @property
    def bloomberg_raw(self) -> Path:
        return self.root / "data" / "raw" / "bloomberg"

    @property
    def fred_raw(self) -> Path:
        return self.root / "data" / "raw" / "fred"

    @property
    def experiments(self) -> Path:
        return self.root / "data" / "experiments"

    @property
    def results(self) -> Path:
        return self.root / "data" / "results"

    @property
    def figures(self) -> Path:
        return self.results / "figures"

    @property
    def tables(self) -> Path:
        return self.results / "tables"

    @property
    def models(self) -> Path:
        return self.results / "models"

    @property
    def runs(self) -> Path:
        return self.results / "runs"

    @property
    def latest(self) -> Path:
        return self.results / "latest"

    @property
    def published(self) -> Path:
        return self.results / "published"

    def ensure_output_dirs(self) -> None:
        for path in (
            self.experiments,
            self.figures,
            self.tables,
            self.models,
            self.runs,
            self.latest,
            self.published,
        ):
            path.mkdir(parents=True, exist_ok=True)
