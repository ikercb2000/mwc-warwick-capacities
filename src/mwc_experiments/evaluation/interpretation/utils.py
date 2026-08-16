"""Interpretation domain."""

from __future__ import annotations
from collections.abc import Mapping
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from capacities_ml_fin.base.interpretation import (
    pairwise_interaction_matrix,
    shapley_indices,
)
from mwc_experiments.modeling.inspection import unwrap_fitted_estimator

def capacity_summary(
    model,
    feature_names: list[str] | tuple[str, ...],
) -> tuple[pd.Series, pd.DataFrame]:
    """Return named Shapley values and pairwise interactions for a capacity model."""
    estimator = unwrap_fitted_estimator(model)
    if not hasattr(estimator, "capacity_"):
        raise AttributeError("The fitted estimator exposes no capacity.")
    capacity = estimator.capacity_
    raw_shapley = shapley_indices(capacity)
    values = np.asarray(list(raw_shapley.values()), dtype=float)
    if values.size != len(feature_names):
        raise ValueError("Feature-name count does not match capacity dimension.")
    shapley = pd.Series(values, index=feature_names, name="Shapley importance")
    matrix = pd.DataFrame(
        pairwise_interaction_matrix(capacity),
        index=feature_names,
        columns=feature_names,
    )
    return shapley.sort_values(ascending=False), matrix


def top_interactions(matrix: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the strongest unique pairwise interactions in long-table form."""
    rows = []
    for i, first in enumerate(matrix.index):
        for j, second in enumerate(matrix.columns):
            if j <= i:
                continue
            value = float(matrix.iloc[i, j])
            rows.append(
                {
                    "first": first,
                    "second": second,
                    "interaction": value,
                    "absolute interaction": abs(value),
                    "interpretation": (
                        "complementary"
                        if value > 0
                        else "redundant"
                        if value < 0
                        else "neutral"
                    ),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values("absolute interaction", ascending=False)
        .head(n)
        .drop(columns="absolute interaction")
        .reset_index(drop=True)
    )


def orientation_table(model) -> pd.DataFrame:
    """Return the fitted feature-orientation table from a model pipeline."""
    if not isinstance(model, Pipeline):
        raise TypeError("Expected a fitted sklearn Pipeline.")
    preprocessor = model.named_steps.get("preprocessor")
    if not isinstance(preprocessor, Pipeline):
        raise AttributeError("The fitted model has no preprocessing pipeline.")
    orient = preprocessor.named_steps.get("orient")
    if orient is None or not hasattr(orient, "orientation_table"):
        raise AttributeError("The fitted model has no orientation transformer.")
    return orient.orientation_table()


def orientation_tables(
    models: Mapping[object, object],
    *,
    key_names: tuple[str, ...],
) -> pd.DataFrame:
    """Combine final orientation diagnostics for every oriented fitted model."""
    tables: list[pd.DataFrame] = []
    for raw_key, model in models.items():
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        if len(key) != len(key_names):
            raise ValueError("Model key does not match orientation table key names.")
        try:
            table = orientation_table(model).rename_axis("feature").reset_index()
        except AttributeError:
            continue
        for position, (name, value) in enumerate(zip(key_names, key)):
            table.insert(position, name, value)
        tables.append(table)
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True).set_index(
        [*key_names, "feature"]
    )
