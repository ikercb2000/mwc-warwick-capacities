"""Stability domain."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import clone
from mwc_experiments.evaluation.interpretation import capacity_summary
from mwc_experiments.modeling.registries import (
    classification_candidates,
    regression_candidates,
)

@dataclass(slots=True)
class CapacityStabilityResult:
    shapley: pd.DataFrame
    interactions: dict[pd.Timestamp, pd.DataFrame]
    interaction_long: pd.DataFrame
    failures: pd.DataFrame

    def interaction_stability(self) -> pd.DataFrame:
        """Summarise magnitude, dispersion and sign persistence by predictor pair."""
        if self.interaction_long.empty:
            return pd.DataFrame()
        grouped = self.interaction_long.groupby(["first", "second"])["interaction"]
        summary = grouped.agg(["mean", "std", "min", "max", "count"])
        sign_persistence = grouped.apply(
            lambda values: float(
                max((values > 0).mean(), (values < 0).mean(), (values == 0).mean())
            )
        )
        summary["sign persistence"] = sign_persistence
        summary["mean absolute interaction"] = grouped.apply(
            lambda values: float(np.mean(np.abs(values)))
        )
        return summary.sort_values("mean absolute interaction", ascending=False)
