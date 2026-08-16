"""Robustness domain."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from mwc_experiments.evaluation.metrics import regression_metrics

@dataclass(frozen=True, slots=True)
class EmpiricalStressDefinition:
    """Training-sample thresholds defining one common empirical stress scenario."""

    feature_lower: pd.Series
    feature_upper: pd.Series
    loss_threshold: float

    def mask(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Flag raw observations with extreme predictors or an extreme loss."""
        missing = self.feature_lower.index.difference(X.columns)
        if not missing.empty:
            raise KeyError(f"Missing stress features: {missing.tolist()}")
        aligned_y = y.reindex(X.index)
        feature_frame = X.loc[:, self.feature_lower.index]
        feature_extreme = (
            feature_frame.lt(self.feature_lower, axis="columns")
            | feature_frame.gt(self.feature_upper, axis="columns")
        ).any(axis=1)
        loss_extreme = aligned_y >= self.loss_threshold
        return (feature_extreme | loss_extreme).fillna(False).rename("stress")

    def audit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample: str,
    ) -> pd.Series:
        """Summarise why observations enter the empirical stress scenario."""
        aligned_y = y.reindex(X.index)
        feature_frame = X.loc[:, self.feature_lower.index]
        feature_extreme = (
            feature_frame.lt(self.feature_lower, axis="columns")
            | feature_frame.gt(self.feature_upper, axis="columns")
        ).any(axis=1)
        loss_extreme = aligned_y >= self.loss_threshold
        combined = (feature_extreme | loss_extreme).fillna(False)
        observations = len(X)
        return pd.Series(
            {
                "sample": sample,
                "observations": observations,
                "feature-extreme observations": int(feature_extreme.sum()),
                "loss-extreme observations": int(loss_extreme.sum()),
                "stress observations": int(combined.sum()),
                "stress rate": (
                    float(combined.mean()) if observations else float("nan")
                ),
                "loss threshold": self.loss_threshold,
            }
        )
