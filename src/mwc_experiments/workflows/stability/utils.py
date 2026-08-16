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
from .types import CapacityStabilityResult

def expanding_capacity_stability(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cutoffs: tuple[str, ...],
    task: str = "regression",
    model_name: str | None = None,
    purge: int = 0,
    random_state: int = 42,
    verbose: bool = True,
) -> CapacityStabilityResult:
    """Refit a capacity model on expanding samples and track intrinsic interpretation."""
    if task not in {"regression", "classification"}:
        raise ValueError("task must be 'regression' or 'classification'.")
    if purge < 0:
        raise ValueError("purge must be non-negative.")
    features = tuple(X.columns)
    if task == "regression":
        name = "Choquet 2-additive" if model_name is None else model_name
        candidate = regression_candidates(
            len(features),
            random_state=random_state,
            include_mlp=False,
            include_dummy=False,
            include_regularized_choquet=False,
        )[name]
    else:
        name = "Choquistic 2-additive" if model_name is None else model_name
        candidate = classification_candidates(
            len(features),
            random_state=random_state,
            include_mlp=False,
        )[name]

    combined = X.join(y.rename("__target__"), how="inner").dropna()
    shapley_rows: dict[pd.Timestamp, pd.Series] = {}
    matrices: dict[pd.Timestamp, pd.DataFrame] = {}
    long_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for cutoff_string in cutoffs:
        cutoff = pd.Timestamp(cutoff_string)
        if verbose:
            print(f"[capacity stability: {task}] cutoff={cutoff.date()}", flush=True)
        sample = combined.loc[:cutoff]
        if purge:
            sample = sample.iloc[:-purge]
        if len(sample) < max(100, 10 * len(features)):
            failures.append({"cutoff": cutoff, "message": "insufficient observations"})
            continue
        target = sample["__target__"]
        if task == "classification" and target.nunique() < 2:
            failures.append({"cutoff": cutoff, "message": "only one class observed"})
            continue
        estimator = clone(candidate.estimator)
        try:
            estimator.fit(sample[list(features)], target.astype(int) if task == "classification" else target)
            shapley, interaction = capacity_summary(estimator, list(features))
        except Exception as error:
            failures.append(
                {
                    "cutoff": cutoff,
                    "message": f"{type(error).__name__}: {error}",
                }
            )
            continue
        shapley_rows[cutoff] = shapley
        matrices[cutoff] = interaction
        for i, first in enumerate(features):
            for j in range(i + 1, len(features)):
                second = features[j]
                long_rows.append(
                    {
                        "cutoff": cutoff,
                        "first": first,
                        "second": second,
                        "interaction": float(interaction.loc[first, second]),
                    }
                )

    shapley_panel = (
        pd.DataFrame(shapley_rows).T.sort_index()
        if shapley_rows
        else pd.DataFrame(columns=features)
    )
    interaction_long = pd.DataFrame(long_rows)
    return CapacityStabilityResult(
        shapley=shapley_panel,
        interactions=matrices,
        interaction_long=interaction_long,
        failures=pd.DataFrame(failures),
    )
