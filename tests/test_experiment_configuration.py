from __future__ import annotations

from pathlib import Path

import pytest

from mwc_experiments.configuration import (
    load_experiment_config,
    parameter_grid_overrides,
)
from mwc_experiments.data.datasets import _processed_input_fingerprint
from mwc_experiments.modeling.registries import (
    apply_parameter_grid_overrides,
    classification_candidates,
    regression_candidates,
)
from mwc_experiments.paths import RepoPaths
from mwc_experiments.runs import experiment_catalog


def test_catalog_resolves_metadata_from_individual_tomls() -> None:
    paths = RepoPaths.discover()
    catalog = experiment_catalog(paths.root)

    assert set(catalog) == {
        "data_preparation",
        "factor_models",
        "future_loss",
        "tail_risk",
        "distortion_risk",
        "autoregression",
    }
    for experiment_id, metadata in catalog.items():
        assert metadata["id"] == experiment_id
        assert metadata["config"] == (
            f"configs/experiments/{experiment_id}.toml"
        )
        assert (paths.root / metadata["script"]).is_file()
        assert (paths.root / metadata["notebook"]).is_file()


def test_distortion_risk_real_data_configuration_is_coherent() -> None:
    config = load_experiment_config("distortion_risk")
    real_data = config["real_data"]

    assets = set(real_data["assets"])
    assert len(assets) >= 3
    assert set(real_data["correlation_assets"]).issubset(assets)
    assert len(real_data["diversification_assets"]) == 2
    assert set(real_data["diversification_assets"]).issubset(assets)
    assert float(real_data["stress_vix_threshold"]) > 0.0


def test_forecasting_aggregators_only_use_non_choquet_sources() -> None:
    for experiment_id in ("future_loss", "tail_risk"):
        config = load_experiment_config(experiment_id)
        aggregation = config["aggregation"]
        source_groups = [aggregation["base_models"]]
        if "robustness_base_models" in aggregation:
            source_groups.append(aggregation["robustness_base_models"])
        for sources in source_groups:
            assert sources
            assert all("choquet" not in name.casefold() for name in sources)
            assert all("choquistic" not in name.casefold() for name in sources)

        primary_sources = set(aggregation["base_models"])
        assert "MLP" in primary_sources
        assert "Fuzzy Choquet neural network" not in primary_sources


@pytest.mark.parametrize(
    "experiment_id",
    ["factor_models", "future_loss", "tail_risk"],
)
def test_predictive_experiments_disable_clipping_by_default(
    experiment_id: str,
) -> None:
    config = load_experiment_config(experiment_id)

    assert config["preprocessing"]["clipping_enabled"] is False


def test_factor_experiment_configures_requested_choquet_families() -> None:
    config = load_experiment_config("factor_models")
    allowed = {
        "OLS",
        "OLS oriented",
        "Monotone linear",
        "Ridge",
        "Lasso",
        "Elastic net",
        "Choquet 1-additive",
        "Choquet 1-additive scaled-q",
        "Choquet 2-additive",
        "Choquet 2-additive L1",
        "Choquet 2-additive scaled-q",
        "Choquet 2-additive scaled-q L1",
    }

    assert set(config["models"]["main"]) == allowed
    for model_group in (
        "orientation",
        "extreme_robustness",
        "residual_covariance",
        "prediction",
    ):
        assert set(config["models"][model_group]).issubset(allowed)
    assert config["analysis"]["representative_model"] == "Choquet 1-additive"
    assert set(parameter_grid_overrides(config)).issubset(allowed)


def test_future_loss_keeps_scaled_and_unscaled_choquet_variants() -> None:
    config = load_experiment_config("future_loss")
    required = {
        "Choquet 1-additive",
        "Choquet 1-additive scaled-q",
        "Choquet 2-additive",
        "Choquet 2-additive L1",
        "Choquet 2-additive scaled-q",
        "Choquet 2-additive scaled-q L1",
    }

    for group in ("main", "cap_weight", "orientation", "extreme_robustness"):
        assert required.issubset(config["models"][group])
    assert {
        "Choquet 2-additive L1",
        "Choquet 2-additive scaled-q L1",
    }.issubset(parameter_grid_overrides(config))


@pytest.mark.parametrize(
    ("experiment_id", "task", "n_features"),
    [
        ("factor_models", "regression", 7),
        ("future_loss", "regression", 10),
        ("tail_risk", "classification", 10),
    ],
)
def test_toml_parameter_grids_match_real_estimators(
    experiment_id: str,
    task: str,
    n_features: int,
) -> None:
    config = load_experiment_config(experiment_id)
    overrides = parameter_grid_overrides(config)
    candidates = (
        regression_candidates(n_features)
        if task == "regression"
        else classification_candidates(n_features)
    )

    configured = apply_parameter_grid_overrides(
        candidates,
        overrides,
        n_features=n_features,
    )

    assert set(overrides).issubset(configured)
    for model, grid in overrides.items():
        assert set(configured[model].param_grid) == set(grid)


def test_experiment_id_must_match_filename(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs" / "experiments"
    config_dir.mkdir(parents=True)
    (config_dir / "demo.toml").write_text(
        "[experiment]\n"
        'id="wrong"\n'
        'script="scripts/demo.py"\n'
        'notebook="notebooks/demo.ipynb"\n'
        'description="demo"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected 'demo'"):
        load_experiment_config("demo", tmp_path)


def test_data_fingerprint_tracks_data_preparation_toml(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs" / "experiments"
    config_dir.mkdir(parents=True)
    data_config = config_dir / "data_preparation.toml"
    data_config.write_text("[dataset]\nhorizons=[1,5]\n", encoding="utf-8")
    paths = RepoPaths(tmp_path)

    first, inputs = _processed_input_fingerprint(paths)
    data_config.write_text("[dataset]\nhorizons=[1,5,10]\n", encoding="utf-8")
    second, _ = _processed_input_fingerprint(paths)

    assert any(
        item["path"] == "configs/experiments/data_preparation.toml"
        for item in inputs
    )
    assert first != second
