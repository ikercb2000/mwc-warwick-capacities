"""Datasets domain."""

from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4
import pandas as pd
from capacities_ml_fin.finance import forward_losses
from mwc_experiments.configuration import load_experiment_config
from mwc_experiments.settings import (
    EQUITY_TICKERS,
    HORIZONS,
    MIN_TAIL_HISTORY,
    PRIMARY_TAIL_ALPHA,
    ROBUSTNESS_TAIL_ALPHA,
    TAIL_WINDOW,
)
from mwc_experiments.data.features import (
    factor_frame,
    prepare_market_data,
)
from mwc_experiments.data.loaders import load_raw_market_data
from mwc_experiments.data.types import (
    ExperimentData,
    PreparedMarketData,
    ProcessedExperimentData,
)
from mwc_experiments.paths import RepoPaths
from mwc_experiments.runs import atomic_write_json, sha256_file
from .types import StaleProcessedDataError

DATA_MANIFEST_SCHEMA_VERSION = 1


if __name__ == "__main__":
    main()


def _processed_input_fingerprint(paths: RepoPaths) -> tuple[str, list[dict[str, str]]]:
    """Hash raw inputs, data-building code and configuration deterministically."""
    roots = (
        (
            paths.root / "data" / "raw",
            {".csv", ".xls", ".xlsx", ".parquet"},
        ),
        (paths.root / "src" / "mwc_experiments" / "data", {".py"}),
    )
    files = [
        path
        for root, suffixes in roots
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in suffixes
        and "__pycache__" not in path.parts
    ]
    settings = paths.root / "configs" / "experiment_settings.toml"
    if settings.is_file():
        files.append(settings)
    data_config = paths.root / "configs" / "experiments" / "data_preparation.toml"
    if data_config.is_file():
        files.append(data_config)
    digest = hashlib.sha256()
    records: list[dict[str, str]] = []
    for path in sorted(set(files)):
        relative = path.relative_to(paths.root).as_posix()
        checksum = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(checksum.encode("ascii"))
        records.append({"path": relative, "sha256": checksum})
    return digest.hexdigest(), records


def _atomic_parquet(frame: pd.DataFrame, target: Path) -> None:
    """Write a Parquet dataset without exposing partially written files."""
    temporary = target.with_name(
        f".{target.stem}.{uuid4().hex}{target.suffix}"
    )
    frame.to_parquet(temporary)
    temporary.replace(target)


def build_portfolio_dataset(
    data: PreparedMarketData,
    *,
    portfolio: str = "equal",
    horizons: tuple[int, ...] = HORIZONS,
    tail_alphas: tuple[float, ...] = (
        PRIMARY_TAIL_ALPHA,
        ROBUSTNESS_TAIL_ALPHA,
    ),
    tail_window: int = TAIL_WINDOW,
    min_tail_history: int = MIN_TAIL_HISTORY,
) -> pd.DataFrame:
    """Build predictors, future-loss targets and point-in-time tail labels."""
    if portfolio == "equal":
        features = data.portfolio_features_equal.copy()
        returns = data.equal_weight_returns
        observed_loss = data.equal_weight_losses
    elif portfolio == "cap":
        features = data.portfolio_features_cap.copy()
        returns = data.cap_weight_returns
        observed_loss = data.cap_weight_losses
    else:
        raise ValueError("portfolio must be 'equal' or 'cap'.")

    dataset = features.copy()
    dataset["portfolio_loss_1d"] = observed_loss

    for horizon in horizons:
        target_name = f"future_loss_h{horizon}"
        target = forward_losses(returns, horizon=horizon, method="log").rename(target_name)
        dataset[target_name] = target

        # A target at origin s is only observable after s+h. Shifting by h before
        # estimating the rolling quantile prevents the tail threshold at t from using
        # outcomes whose future window has not yet completed by t so we avoid look-ahead bias
        available_history = target.shift(horizon)
        for alpha in tail_alphas:
            alpha_label = str(alpha).replace(".", "p")
            quantile_name = f"historical_var_h{horizon}_a{alpha_label}"
            label_name = f"tail_event_h{horizon}_a{alpha_label}"
            threshold = available_history.rolling(
                tail_window,
                min_periods=min_tail_history,
            ).quantile(alpha)
            dataset[quantile_name] = threshold
            dataset[label_name] = (target > threshold).where(
                target.notna() & threshold.notna()
            ).astype("Int64")

    dataset.index.name = "date"
    return dataset


def build_experiment_data(
    paths: RepoPaths | None = None,
    *,
    horizons: tuple[int, ...] | None = None,
    tail_alphas: tuple[float, ...] | None = None,
    tail_window: int | None = None,
    min_tail_history: int | None = None,
) -> ExperimentData:
    """Build the complete raw, prepared and model-ready experiment bundle."""
    paths = RepoPaths.discover() if paths is None else paths
    dataset_config = load_experiment_config(
        "data_preparation",
        paths.root,
    )["dataset"]
    resolved_horizons = (
        tuple(int(value) for value in dataset_config["horizons"])
        if horizons is None
        else horizons
    )
    resolved_tail_alphas = (
        tuple(float(value) for value in dataset_config["tail_alphas"])
        if tail_alphas is None
        else tail_alphas
    )
    resolved_tail_window = (
        int(dataset_config["tail_window"])
        if tail_window is None
        else tail_window
    )
    resolved_min_tail_history = (
        int(dataset_config["minimum_tail_history"])
        if min_tail_history is None
        else min_tail_history
    )
    raw = load_raw_market_data(paths)
    prepared = prepare_market_data(raw)
    factors = {asset: factor_frame(prepared, asset) for asset in EQUITY_TICKERS}
    equal_dataset = build_portfolio_dataset(
        prepared,
        portfolio="equal",
        horizons=resolved_horizons,
        tail_alphas=resolved_tail_alphas,
        tail_window=resolved_tail_window,
        min_tail_history=resolved_min_tail_history,
    )
    cap_dataset = build_portfolio_dataset(
        prepared,
        portfolio="cap",
        horizons=resolved_horizons,
        tail_alphas=resolved_tail_alphas,
        tail_window=resolved_tail_window,
        min_tail_history=resolved_min_tail_history,
    )
    return ExperimentData(
        raw=raw,
        prepared=prepared,
        factor_frames=factors,
        equal_weight_dataset=equal_dataset,
        cap_weight_dataset=cap_dataset,
    )


def save_processed_data(
    experiment_data: ExperimentData,
    paths: RepoPaths | None = None,
) -> dict[str, object]:
    """Persist portfolio and factor datasets under the experiment data path."""
    paths = RepoPaths.discover() if paths is None else paths
    paths.ensure_output_dirs()

    outputs: dict[str, object] = {}
    equal_path = paths.experiments / "portfolio_equal_weight.parquet"
    cap_path = paths.experiments / "portfolio_cap_weight.parquet"
    _atomic_parquet(experiment_data.equal_weight_dataset, equal_path)
    _atomic_parquet(experiment_data.cap_weight_dataset, cap_path)
    outputs["equal_weight"] = equal_path
    outputs["cap_weight"] = cap_path

    factor_dir = paths.experiments / "factor_models"
    factor_dir.mkdir(parents=True, exist_ok=True)
    factor_paths: dict[str, object] = {}
    for asset, frame in experiment_data.factor_frames.items():
        path = factor_dir / f"{asset}_factor_frame.parquet"
        _atomic_parquet(frame, path)
        factor_paths[asset] = path
    outputs["factor_frames"] = factor_paths
    fingerprint, inputs = _processed_input_fingerprint(paths)
    dataset_frames = {
        equal_path: experiment_data.equal_weight_dataset,
        cap_path: experiment_data.cap_weight_dataset,
        **{
            Path(path): experiment_data.factor_frames[asset]
            for asset, path in factor_paths.items()
        },
    }
    atomic_write_json(
        paths.experiments / "manifest.json",
        {
            "schema_version": DATA_MANIFEST_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_fingerprint": fingerprint,
            "inputs": inputs,
            "datasets": [
                {
                    "path": str(Path(path).relative_to(paths.root)),
                    "bytes": Path(path).stat().st_size,
                    "sha256": sha256_file(Path(path)),
                    "rows": len(frame),
                    "columns": list(frame.columns),
                    "index_start": frame.index.min(),
                    "index_end": frame.index.max(),
                }
                for path, frame in dataset_frames.items()
            ],
        },
    )
    return outputs


def load_processed_data(
    paths: RepoPaths | None = None,
) -> ProcessedExperimentData:
    """Load the persisted portfolio and factor datasets without rebuilding raw data."""
    paths = RepoPaths.discover() if paths is None else paths
    equal_path = paths.experiments / "portfolio_equal_weight.parquet"
    cap_path = paths.experiments / "portfolio_cap_weight.parquet"
    factor_dir = paths.experiments / "factor_models"
    required = [
        equal_path,
        cap_path,
        *(factor_dir / f"{asset}_factor_frame.parquet" for asset in EQUITY_TICKERS),
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Processed experiment data are incomplete. Missing files:\n" + formatted
        )
    manifest_path = paths.experiments / "manifest.json"
    if not manifest_path.is_file():
        raise StaleProcessedDataError(
            "Processed datasets have no provenance manifest and must be rebuilt."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint, _ = _processed_input_fingerprint(paths)
    if (
        manifest.get("schema_version") != DATA_MANIFEST_SCHEMA_VERSION
        or manifest.get("input_fingerprint") != fingerprint
    ):
        raise StaleProcessedDataError(
            "Processed datasets are stale relative to raw data, configuration, "
            "or data-building code."
        )
    recorded = {
        paths.root / item["path"]: item["sha256"]
        for item in manifest.get("datasets", [])
    }
    corrupt = [
        path
        for path in required
        if path not in recorded or sha256_file(path) != recorded[path]
    ]
    if corrupt:
        raise StaleProcessedDataError(
            "Processed dataset checksums do not match the manifest: "
            + ", ".join(str(path) for path in corrupt)
        )
    return ProcessedExperimentData(
        factor_frames={
            asset: pd.read_parquet(factor_dir / f"{asset}_factor_frame.parquet")
            for asset in EQUITY_TICKERS
        },
        equal_weight_dataset=pd.read_parquet(equal_path),
        cap_weight_dataset=pd.read_parquet(cap_path),
    )


def load_or_build_processed_data(
    paths: RepoPaths | None = None,
) -> ProcessedExperimentData:
    """Load processed datasets, building and persisting them when absent."""
    paths = RepoPaths.discover() if paths is None else paths
    try:
        return load_processed_data(paths)
    except (FileNotFoundError, StaleProcessedDataError):
        save_processed_data(build_experiment_data(paths), paths)
        return load_processed_data(paths)


def main() -> None:
    """Build, save and audit all experiment datasets from the command line."""
    paths = RepoPaths.discover()
    data = build_experiment_data(paths)
    outputs = save_processed_data(data, paths)
    print(data.raw.audit().to_string(index=False))
    print("\nSaved processed datasets:")
    for key, value in outputs.items():
        print(f"- {key}: {value}")
