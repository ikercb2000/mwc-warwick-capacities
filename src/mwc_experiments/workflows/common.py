"""Helpers shared by experiment workflows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mwc_experiments.modeling.inspection import unwrap_fitted_estimator
from mwc_experiments.modeling.types import Candidate


def quick_candidates(candidates: dict[str, Candidate]) -> dict[str, Candidate]:
    """Retain one representative hyperparameter combination per model."""
    result: dict[str, Candidate] = {}
    for name, candidate in candidates.items():
        compact = {
            key: [values[len(values) // 2]]
            for key, values in candidate.param_grid.items()
        }
        result[name] = Candidate(candidate.estimator, compact)
    return result


def select_model_family_by_validation(
    metrics: pd.DataFrame,
    *,
    family_prefix: str,
    score_column: str,
    model_level: str = "model",
) -> pd.DataFrame:
    """Select the best specification in a model family using validation only."""
    if score_column not in metrics:
        raise KeyError(f"Missing validation score column: {score_column}")
    index_names = [name for name in metrics.index.names if name is not None]
    if model_level not in index_names:
        raise ValueError(f"Metrics index has no {model_level!r} level.")

    frame = metrics.reset_index()
    family = frame[
        frame[model_level].astype(str).str.startswith(family_prefix)
    ].copy()
    if family.empty:
        raise ValueError(f"No models start with {family_prefix!r}.")

    group_levels = [name for name in index_names if name != model_level]
    if not group_levels:
        return (
            family.nsmallest(1, score_column)
            .set_index(model_level)
            .sort_index()
        )

    selected_rows = family.groupby(
        group_levels,
        sort=False,
    )[score_column].idxmin()
    return family.loc[selected_rows].set_index(group_levels).sort_index()


def model_parameter_count(estimator: Any) -> int | float:
    """Return a transparent parameter count when a meaningful count is available."""
    inner = unwrap_fitted_estimator(estimator)
    if hasattr(inner, "result_"):
        return int(np.asarray(inner.result_.parameters).size)
    if hasattr(inner, "coef_"):
        count = int(np.asarray(inner.coef_).size)
        if hasattr(inner, "intercept_"):
            count += int(np.asarray(inner.intercept_).size)
        return count
    return float("nan")
