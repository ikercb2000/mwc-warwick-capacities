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
    support = workflow_dir / "experiment_support"
    assert support.is_dir()
    assert (support / "types.py").is_file()
    assert (support / "utils.py").is_file()
    assert (support / "__init__.py").is_file()
    assert not (workflow_dir / "data_preparation.py").exists()
    assert not (ROOT / "src" / "mwc_experiments" / "experiments").exists()


def test_domain_modules_follow_the_types_utils_mappings_convention() -> None:
    """Prevent classes, functions and dictionary mappings being mixed again."""
    source_root = ROOT / "src" / "mwc_experiments"
    permitted_modules = {
        "__init__.py",
        "types.py",
        "utils.py",
        "mappings.py",
        "constants.py",
    }
    for path in source_root.rglob("*.py"):
        assert path.name in permitted_modules, path.relative_to(source_root)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        top_level_classes = [
            node for node in tree.body if isinstance(node, ast.ClassDef)
        ]
        top_level_functions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        literal_mappings = [
            node
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, (ast.Dict, ast.DictComp))
        ]
        if path.name == "types.py":
            assert not top_level_functions, path.relative_to(source_root)
            assert not literal_mappings, path.relative_to(source_root)
        elif path.name == "utils.py":
            assert not top_level_classes, path.relative_to(source_root)
            assert not literal_mappings, path.relative_to(source_root)
        elif path.name == "mappings.py":
            assert not top_level_classes, path.relative_to(source_root)
            assert not top_level_functions, path.relative_to(source_root)


def test_complete_runner_renders_notebook_reports() -> None:
    """Keep HTML reporting connected on Windows, macOS and Linux."""
    scripts = ROOT / "scripts"
    windows_runner = (scripts / "run_all_experiments.ps1").read_text(
        encoding="utf-8"
    )
    unix_runner = (scripts / "run_all_experiments.sh").read_text(
        encoding="utf-8"
    )
    windows_renderer = scripts / "render_notebooks.ps1"
    unix_renderer = scripts / "render_notebooks.sh"

    assert windows_renderer.is_file()
    assert unix_renderer.is_file()
    assert "render_notebooks.ps1" in windows_runner
    assert "render_notebooks.sh" in unix_runner
    for runner in (windows_runner, unix_runner):
        assert "--quick" in runner
        assert "--full" in runner
        assert "--no-render" in runner
        assert "audit_results.py" in runner
    assert "--execute" in windows_renderer.read_text(encoding="utf-8")
    assert "--execute" in unix_renderer.read_text(encoding="utf-8")


def test_scale_dominant_series_have_inclusive_and_exclusive_figures() -> None:
    """Keep NVDA and VIX from obscuring the scale of the remaining series."""
    artifact_names = (
        "data_equity_wealth_with_nvda.png",
        "data_equity_wealth_without_nvda.png",
        "data_raw_market_panels_with_vix.png",
        "data_raw_market_panels_without_vix.png",
    )
    data_script = (ROOT / "scripts" / "build_experiment_data.py").read_text(
        encoding="utf-8"
    )
    notebook = json.loads(
        (ROOT / "notebooks" / "00_data_preparation_and_eda.ipynb").read_text(
            encoding="utf-8"
        )
    )
    notebook_source = "\n".join(
        "".join(cell.get("source", ""))
        if isinstance(cell.get("source", ""), list)
        else cell.get("source", "")
        for cell in notebook["cells"]
    )

    for artifact_name in artifact_names:
        assert artifact_name in data_script
        assert artifact_name in notebook_source
    for obsolete_name in (
        "data_equity_wealth.png",
        "data_raw_market_panels.png",
    ):
        assert f'"{obsolete_name}"' not in data_script
        assert f'"{obsolete_name}"' not in notebook_source


def test_experiments_persist_complete_orientation_diagnostics() -> None:
    """Keep final training correlations and orientations in result artifacts."""
    expected = {
        "experiment_factors.py": "experiment_1_orientations.csv",
        "experiment_predict_loss.py": "experiment_2a_orientations_h{horizon}.csv",
        "experiment_tail_risk.py": "experiment_2b_orientations_h{horizon}_a095.csv",
    }
    for script_name, artifact_name in expected.items():
        source = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        assert "orientation_tables(" in source
        assert artifact_name in source


def test_tail_risk_persists_separate_probability_evaluations() -> None:
    """Keep discrimination, calibration and temporal samples auditable."""
    source = (ROOT / "scripts" / "experiment_tail_risk.py").read_text(
        encoding="utf-8"
    )

    for artifact_name in (
        "experiment_2b_discrimination_h{horizon}_a095.csv",
        "experiment_2b_calibration_h{horizon}_a095.csv",
        "experiment_2b_calibration_sample_h{horizon}_a095.csv",
    ):
        assert artifact_name in source

    assert "experiment_2b_classifier_discrimination_h{horizon}_a095.png" in source
    assert "experiment_2b_calibration_comparison_h{horizon}_a095.png" in source
    assert "experiment_2b_classifier_diagnostics_h{horizon}_a095.png" not in source


def test_forecasting_experiments_persist_walk_forward_audits() -> None:
    expected = {
        "experiment_predict_loss.py": (
            "experiment_2a_walk_forward_folds_h{horizon}.csv",
            "experiment_2a_walk_forward_metrics_h{horizon}.csv",
            "experiment_2a_orientation_history_h{horizon}.csv",
            "experiment_2a_shapley_history_h{horizon}.csv",
        ),
        "experiment_tail_risk.py": (
            "experiment_2b_walk_forward_folds_h{horizon}_a095.csv",
            "experiment_2b_walk_forward_metrics_h{horizon}_a095.csv",
            "experiment_2b_orientation_history_h{horizon}_a095.csv",
            "experiment_2b_shapley_history_h{horizon}_a095.csv",
        ),
    }
    for script_name, artifacts in expected.items():
        source = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        assert "walk_forward_config" in source
        for artifact in artifacts:
            assert artifact in source
