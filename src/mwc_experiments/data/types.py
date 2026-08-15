# imports
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass(slots=True)
class RawMarketData:
    """Hold aligned Bloomberg and FRED panels before feature construction."""

    equity_fields: dict[str, pd.DataFrame]
    etf_fields: dict[str, pd.DataFrame]
    fred: pd.DataFrame
    calendar: pd.DatetimeIndex

    def audit(self) -> pd.DataFrame:
        """Summarise coverage and missingness for every loaded raw panel."""
        rows: list[dict[str, object]] = []
        for group, fields in (("equity", self.equity_fields), ("etf", self.etf_fields)):
            for field, frame in fields.items():
                rows.append(
                    {
                        "group": group,
                        "field": field,
                        "rows": len(frame),
                        "columns": frame.shape[1],
                        "start": frame.index.min(),
                        "end": frame.index.max(),
                        "missing_fraction": float(frame.isna().mean().mean()),
                    }
                )
        rows.append(
            {
                "group": "fred",
                "field": "daily_panel",
                "rows": len(self.fred),
                "columns": self.fred.shape[1],
                "start": self.fred.index.min(),
                "end": self.fred.index.max(),
                "missing_fraction": float(self.fred.isna().mean().mean()),
            }
        )
        return pd.DataFrame(rows)


@dataclass(slots=True)
class PreparedMarketData:
    """Hold reusable return, loss, portfolio and predictor panels."""

    equity_returns: pd.DataFrame
    equity_losses: pd.DataFrame
    etf_returns: pd.DataFrame
    etf_losses: pd.DataFrame
    fred: pd.DataFrame
    risk_free_log: pd.Series
    equal_weight_returns: pd.Series
    equal_weight_losses: pd.Series
    cap_weight_returns: pd.Series
    cap_weight_losses: pd.Series
    lagged_cap_weights: pd.DataFrame
    equity_amihud_20d: pd.DataFrame
    portfolio_features_equal: pd.DataFrame
    portfolio_features_cap: pd.DataFrame


@dataclass(slots=True)
class ExperimentData:
    """Bundle raw, prepared and model-ready datasets for all experiments."""

    raw: RawMarketData
    prepared: PreparedMarketData
    factor_frames: dict[str, pd.DataFrame]
    equal_weight_dataset: pd.DataFrame
    cap_weight_dataset: pd.DataFrame


@dataclass(slots=True)
class ProcessedExperimentData:
    """Hold the persisted model-ready datasets used by experiment runners."""

    factor_frames: dict[str, pd.DataFrame]
    equal_weight_dataset: pd.DataFrame
    cap_weight_dataset: pd.DataFrame