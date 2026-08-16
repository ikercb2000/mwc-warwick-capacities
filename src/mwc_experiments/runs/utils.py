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
from .types import ExperimentRunPaths
from .mappings import SCRIPT_EXPERIMENTS, ARTIFACT_EXPERIMENTS

def sha256_file(path: Path) -> str:
    """Hash one file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    """Atomically replace a JSON document in its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def infer_experiment_id(script: str | Path | None = None) -> str:
    """Infer a stable experiment identifier from the executable script name."""
    source = Path(sys.argv[0] if script is None else script).stem
    return SCRIPT_EXPERIMENTS.get(source, source.replace("experiment_", ""))


def infer_artifact_experiment(filename: str) -> str | None:
    """Map a stable artifact filename to the experiment that produces it."""
    for prefix, experiment_id in ARTIFACT_EXPERIMENTS.items():
        if filename.startswith(prefix):
            return experiment_id
    return None


def experiment_catalog(root: Path) -> dict[str, dict[str, str]]:
    """Load the index and resolve metadata from each experiment TOML."""
    path = root / "configs" / "experiments.toml"
    with path.open("rb") as stream:
        payload = tomllib.load(stream)
    catalog: dict[str, dict[str, str]] = {}
    for experiment_id, index_entry in payload["experiments"].items():
        if "config" not in index_entry:
            # Compatibility with manifests/tests created before split configs.
            catalog[experiment_id] = dict(index_entry)
            continue
        config = load_experiment_config(experiment_id, root)
        metadata = {
            key: str(value)
            for key, value in config["experiment"].items()
        }
        metadata["config"] = str(index_entry["config"])
        catalog[experiment_id] = metadata
    return catalog


def notebook_artifact_references(path: Path) -> set[str]:
    """Extract literal result filenames requested by one reporting notebook."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    references: set[str] = set()
    for cell in payload["cells"]:
        if cell.get("cell_type") != "code":
            continue
        tree = ast.parse("".join(cell.get("source", [])))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"load_result_table", "result_figure_path"}
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                references.add(node.args[0].value)
    return references


def _git_value(root: Path, arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _package_versions() -> dict[str, str]:
    packages = (
        "msc-warwick-capacities",
        "capacities-ml-fin",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "pyarrow",
    )
    result: dict[str, str] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            continue
    return result


def create_experiment_run(
    repository: RepoPaths,
    *,
    experiment_id: str,
    mode: str,
) -> ExperimentRunPaths:
    """Create one unique run directory and its initial running manifest."""
    repository.ensure_output_dirs()
    commit = _git_value(repository.root, ["rev-parse", "HEAD"])
    short_commit = commit[:7] if commit else "nogit"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{short_commit}_{mode}_{os.getpid()}"
    paths = ExperimentRunPaths(repository, experiment_id, run_id, mode)
    paths.ensure_output_dirs()
    shared_configuration = repository.root / "configs" / "experiment_settings.toml"
    experiment_configuration = experiment_config_path(experiment_id, repository.root)
    configuration_records: dict[str, dict[str, str]] = {}
    for name, configuration, snapshot_name in (
        ("shared", shared_configuration, "shared_config.toml"),
        ("experiment", experiment_configuration, "experiment_config.toml"),
    ):
        if configuration.is_file():
            shutil.copy2(configuration, paths.run / snapshot_name)
            configuration_records[name] = {
                "path": str(configuration.relative_to(repository.root)),
                "sha256": sha256_file(configuration),
                "snapshot": snapshot_name,
            }
    configuration_digest = hashlib.sha256()
    for name, record in sorted(configuration_records.items()):
        configuration_digest.update(name.encode("utf-8"))
        configuration_digest.update(record["sha256"].encode("ascii"))
    dirty = _git_value(repository.root, ["status", "--porcelain"])
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "mode": mode,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "git": {
            "commit": commit,
            "dirty": bool(dirty),
            "changed_paths": dirty.splitlines() if dirty else [],
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "configuration": configuration_records,
        "configuration_sha256": (
            configuration_digest.hexdigest() if configuration_records else None
        ),
        "artifacts": [],
    }
    atomic_write_json(paths.manifest, manifest)
    return paths


def finish_experiment_run(
    paths: ExperimentRunPaths,
    artifacts: Iterable[Path],
    *,
    status: str,
    message: str | None = None,
) -> None:
    """Finalize a run manifest and publish only successful executions as latest."""
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    artifact_paths = [Path(artifact).resolve() for artifact in artifacts]
    if status == "success":
        catalog = experiment_catalog(paths.root)
        specification = catalog.get(paths.experiment_id)
        if specification is not None:
            notebook = paths.root / specification["notebook"]
            expected = notebook_artifact_references(notebook)
            produced = {artifact.name for artifact in artifact_paths}
            missing = sorted(expected.difference(produced))
            if missing:
                raise RuntimeError(
                    "Run cannot be published because notebook artifacts are "
                    "missing: " + ", ".join(missing)
                )
    records = []
    for target in artifact_paths:
        if target.is_file():
            records.append(
                {
                    "path": str(target.relative_to(paths.root)),
                    "bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                }
            )
    manifest.update(
        {
            "status": status,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "artifacts": records,
        }
    )
    started = datetime.fromisoformat(manifest["started_at_utc"])
    finished = datetime.fromisoformat(manifest["finished_at_utc"])
    manifest["duration_seconds"] = (finished - started).total_seconds()
    data_manifest = paths.experiments / "manifest.json"
    if data_manifest.is_file():
        data_payload = json.loads(data_manifest.read_text(encoding="utf-8"))
        manifest["processed_data"] = {
            "manifest": str(data_manifest.relative_to(paths.root)),
            "input_fingerprint": data_payload.get("input_fingerprint"),
            "manifest_sha256": sha256_file(data_manifest),
        }
    atomic_write_json(paths.manifest, manifest)
    if status != "success":
        return
    (paths.run / "SUCCESS").write_text("success\n", encoding="utf-8")
    pointer = {
        "experiment_id": paths.experiment_id,
        "run_id": paths.run_id,
        "mode": paths.mode,
        "manifest": str(paths.manifest.relative_to(paths.root)),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(
        paths.repository.latest / f"{paths.experiment_id}.json",
        pointer,
    )


def publish_run_snapshot(
    repository: RepoPaths,
    *,
    experiment_id: str,
    run_id: str | None = None,
) -> dict[str, object]:
    """Promote one verified run into the Git-trackable flat result snapshot."""
    if run_id is None:
        pointer_path = repository.latest / f"{experiment_id}.json"
        if not pointer_path.is_file():
            raise FileNotFoundError(
                f"No latest successful run exists for {experiment_id}."
            )
        run_id = json.loads(pointer_path.read_text(encoding="utf-8"))["run_id"]
    run = repository.runs / experiment_id / run_id
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file() or not (run / "SUCCESS").is_file():
        raise FileNotFoundError(f"Run {experiment_id}/{run_id} is incomplete.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "success":
        raise RuntimeError(f"Run status is {manifest.get('status')!r}.")
    copied: list[dict[str, str]] = []
    for record in manifest.get("artifacts", []):
        source = (repository.root / record["path"]).resolve()
        if source.parent.name not in {"tables", "figures"}:
            continue
        if sha256_file(source) != record["sha256"]:
            raise RuntimeError(f"Artifact checksum mismatch: {source}")
        destination_dir = (
            repository.tables
            if source.parent.name == "tables"
            else repository.figures
        )
        destination = destination_dir / source.name
        temporary = destination.with_name(
            f".{destination.name}.{uuid4().hex}.tmp"
        )
        shutil.copy2(source, temporary)
        temporary.replace(destination)
        copied.append(
            {
                "source": str(source.relative_to(repository.root)),
                "destination": str(destination.relative_to(repository.root)),
                "sha256": record["sha256"],
            }
        )
    publication = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_manifest_sha256": sha256_file(manifest_path),
        "artifacts": copied,
    }
    atomic_write_json(
        repository.published / f"{experiment_id}.json",
        publication,
    )
    return publication
