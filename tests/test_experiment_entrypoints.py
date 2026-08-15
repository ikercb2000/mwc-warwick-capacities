"""Protect the separation between experiment execution and notebook reporting."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_SCRIPTS = (
    "build_experiment_data.py",
    "experiment_factors.py",
    "experiment_predict_loss.py",
    "experiment_tail_risk.py",
    "experiment_distortion_risk.py",
    "experiment_autoregression.py",
    "run_smoke_experiments.py",
)
PROHIBITED_SCRIPT_DEFINITIONS = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)


def test_experiment_pipelines_are_written_in_the_scripts() -> None:
    """Keep complete pipelines visible without wrapping them in main functions."""
    for filename in EXPERIMENT_SCRIPTS:
        tree = ast.parse((ROOT / "scripts" / filename).read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, PROHIBITED_SCRIPT_DEFINITIONS)
            for node in ast.walk(tree)
        ), filename
        main_guards = [
            node
            for node in tree.body
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
            )
        ]
        assert not main_guards, filename
        assert any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "paths"
                for target in node.targets
            )
            for node in tree.body
        ), filename
        assert "mwc_experiments.experiments" not in ast.unparse(tree), filename


def test_notebooks_only_load_persisted_report_artifacts() -> None:
    """Ensure notebooks do not define or execute experiment-domain logic."""
    prohibited_nodes = PROHIBITED_SCRIPT_DEFINITIONS + (
        ast.Lambda,
        ast.Assign,
        ast.AnnAssign,
        ast.AugAssign,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.Try,
        ast.With,
        ast.AsyncWith,
    )
    prohibited_imports = (
        "mwc_experiments.data",
        "mwc_experiments.modeling",
        "mwc_experiments.workflows",
    )
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", ""))
            if isinstance(cell.get("source", ""), list)
            else cell.get("source", "")
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        tree = ast.parse(source)
        assert not any(isinstance(node, prohibited_nodes) for node in ast.walk(tree)), path.name
        assert not any(name in source for name in prohibited_imports), path.name
        assert "mwc_experiments.reporting" in source, path.name


def test_only_reusable_experiment_support_lives_in_workflows() -> None:
    """Keep shared helpers reusable without hiding complete pipelines in src."""
    workflow_dir = ROOT / "src" / "mwc_experiments" / "workflows"
    assert (workflow_dir / "experiment_support.py").is_file()
    assert not (workflow_dir / "data_preparation.py").exists()
    assert not (ROOT / "src" / "mwc_experiments" / "experiments").exists()
