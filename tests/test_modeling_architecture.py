from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mwc_experiments.modeling.registries import (
    classification_candidates,
    regression_candidates,
)
from mwc_experiments.modeling.selection import (
    select_classification_model,
    select_regression_model,
)
from mwc_experiments.modeling.types import Candidate
from mwc_experiments.workflows.common import select_model_family_by_validation


def test_registries_use_native_pipelines_and_shared_tree_preprocessing() -> None:
    """Ensure candidates use sklearn containers and trees share standard scaling."""
    regressors = regression_candidates(3, include_mlp=False)
    classifiers = classification_candidates(3, include_mlp=False)

    for candidate in (*regressors.values(), *classifiers.values()):
        assert isinstance(candidate.estimator, Pipeline)

    tree_preprocessor = regressors["Random forest"].estimator.named_steps[
        "preprocessor"
    ]
    assert isinstance(tree_preprocessor.named_steps["scale"], StandardScaler)

    choquet = regressors["Choquet 2-additive"].estimator.named_steps["regressor"]
    assert isinstance(choquet, TransformedTargetRegressor)


def test_one_additive_choquet_matches_monotone_linear_on_simplex_solution() -> None:
    """Verify the expected equality when positive linear slopes sum to one."""
    corners = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    X = pd.DataFrame(
        np.tile(corners, (4, 1)),
        columns=["x1", "x2", "x3"],
    )
    y = pd.Series(0.2 + X.to_numpy() @ np.array([0.55, 0.0, 0.45]))
    candidates = regression_candidates(
        3,
        include_mlp=False,
        include_dummy=False,
        include_regularized_choquet=False,
    )
    monotone = candidates["Monotone linear"].estimator.fit(X, y)
    choquet = candidates["Choquet 1-additive"].estimator.fit(X, y)

    np.testing.assert_allclose(
        choquet.predict(X),
        monotone.predict(X),
        atol=1e-5,
        rtol=1e-5,
    )


def test_choquet_family_selection_uses_validation_score() -> None:
    """Select capacity order and regularization without looking at test RMSE."""
    metrics = pd.DataFrame(
        {
            "validation RMSE": [0.30, 0.20, 0.25, 0.10],
            "RMSE": [0.05, 0.40, 0.10, 0.50],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("A", "Choquet 1-additive"),
                ("A", "Choquet 2-additive"),
                ("B", "Choquet 1-additive"),
                ("B", "Choquet 2-additive L1"),
            ],
            names=["asset", "model"],
        ),
    )

    selected = select_model_family_by_validation(
        metrics,
        family_prefix="Choquet",
        score_column="validation RMSE",
    )

    assert selected.loc["A", "model"] == "Choquet 2-additive"
    assert selected.loc["B", "model"] == "Choquet 2-additive L1"


def test_task_specific_selectors_share_pipeline_compatible_selection() -> None:
    """Exercise both public selectors with sklearn-native candidate pipelines."""
    index = pd.date_range("2020-01-01", periods=24)
    X = pd.DataFrame(
        {"x1": np.linspace(-1.0, 1.0, len(index)), "x2": np.arange(len(index))},
        index=index,
    )

    regression = regression_candidates(
        2,
        include_mlp=False,
        include_dummy=False,
        include_regularized_choquet=False,
    )["Ridge"]
    regression = Candidate(regression.estimator, {"regressor__alpha": [0.1, 1.0]})
    selected_regression = select_regression_model(
        "Ridge",
        regression,
        X.iloc[:16],
        pd.Series(2.0 * X["x1"], index=index).iloc[:16],
        X.iloc[16:],
        pd.Series(2.0 * X["x1"], index=index).iloc[16:],
    )
    assert selected_regression.validation_predictions.shape == (8,)

    classification = classification_candidates(2, include_mlp=False)[
        "Penalized logistic"
    ]
    classification = Candidate(
        classification.estimator,
        {"classifier__C": [0.1, 1.0]},
    )
    target = pd.Series(np.arange(len(index)) % 2, index=index)
    selected_classification = select_classification_model(
        "Penalized logistic",
        classification,
        X.iloc[:16],
        target.iloc[:16],
        X.iloc[16:],
        target.iloc[16:],
    )
    assert selected_classification.validation_predictions.shape == (8,)
