"""Assemble and persist model-ready experiment datasets."""

from __future__ import annotations

import pandas as pd
from capacities_ml_fin.finance import forward_losses

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


def build_experiment_data(paths: RepoPaths | None = None) -> ExperimentData:
    """Build the complete raw, prepared and model-ready experiment bundle."""
    paths = RepoPaths.discover() if paths is None else paths
    raw = load_raw_market_data(paths)
    prepared = prepare_market_data(raw)
    factors = {asset: factor_frame(prepared, asset) for asset in EQUITY_TICKERS}
    equal_dataset = build_portfolio_dataset(prepared, portfolio="equal")
    cap_dataset = build_portfolio_dataset(prepared, portfolio="cap")
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
    experiment_data.equal_weight_dataset.to_parquet(equal_path)
    experiment_data.cap_weight_dataset.to_parquet(cap_path)
    outputs["equal_weight"] = equal_path
    outputs["cap_weight"] = cap_path

    factor_dir = paths.experiments / "factor_models"
    factor_dir.mkdir(parents=True, exist_ok=True)
    factor_paths: dict[str, object] = {}
    for asset, frame in experiment_data.factor_frames.items():
        path = factor_dir / f"{asset}_factor_frame.parquet"
        frame.to_parquet(path)
        factor_paths[asset] = path
    outputs["factor_frames"] = factor_paths
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
    except FileNotFoundError:
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


if __name__ == "__main__":
    main()
