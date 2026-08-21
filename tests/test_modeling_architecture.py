from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from capacities_ml_fin.ml.models import (
    ChoquetClassifier,
    FuzzyChoquetNeuralClassifier,
    FuzzyChoquetNeuralRegressor,
    ScaledChoquetRegressor,
)
from capacities_ml_fin.ml.preprocessing import CapacityNormalizer

from mwc_experiments.evaluation.interpretation import orientation_tables
from mwc_experiments.modeling.inspection import fitted_q
from mwc_experiments.modeling.registries import (
    classification_candidates,
    regression_candidates,
)
from mwc_experiments.modeling.selection import (
    refit_selected,
    select_classification_model,
    select_regression_model,
)
from mwc_experiments.modeling.types import Candidate
from mwc_experiments.modeling.types import CorrelationOrientationTransformer
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
    assert "orient" not in tree_preprocessor.named_steps

    oriented_preprocessor = regressors["OLS oriented"].estimator.named_steps[
        "preprocessor"
    ]
    assert isinstance(
        oriented_preprocessor.named_steps["orient"],
        CorrelationOrientationTransformer,
    )

    capacity_preprocessor = regressors["Choquet 1-additive"].estimator.named_steps[
        "preprocessor"
    ]
    assert isinstance(
        capacity_preprocessor.named_steps["orient"],
        CorrelationOrientationTransformer,
    )
    assert (
        capacity_preprocessor.named_steps[
            "orient"
        ].minimum_absolute_correlation
        == 0.2
    )
    assert capacity_preprocessor.named_steps["orient"].stability_subperiods == 3

    logistic_preprocessor = classifiers["Logistic"].estimator.named_steps[
        "preprocessor"
    ]
    assert "orient" not in logistic_preprocessor.named_steps

    oriented_logistic_preprocessor = classifiers[
        "Logistic oriented"
    ].estimator.named_steps["preprocessor"]
    assert isinstance(
        oriented_logistic_preprocessor.named_steps["orient"],
        CorrelationOrientationTransformer,
    )

    choquet = regressors["Choquet 2-additive"].estimator.named_steps["regressor"]
    assert isinstance(choquet, TransformedTargetRegressor)
    scaled_choquet = regressors[
        "Choquet 1-additive scaled-q"
    ].estimator.named_steps["regressor"]
    assert isinstance(scaled_choquet, ScaledChoquetRegressor)
    for model in (
        "Choquet 2-additive scaled-q",
        "Choquet 2-additive scaled-q L1",
    ):
        assert isinstance(
            regressors[model].estimator.named_steps["regressor"],
            ScaledChoquetRegressor,
        )

    fuzzy_regressor = regression_candidates(3)[
        "Fuzzy Choquet neural network"
    ].estimator
    fuzzy_classifier = classification_candidates(3)[
        "Fuzzy Choquet neural network"
    ].estimator
    mlp_regressor = regression_candidates(3)["MLP"].estimator
    mlp_classifier = classification_candidates(3)["MLP"].estimator
    linear_classifier = classification_candidates(3)[
        "Choquet linear classifier"
    ].estimator
    assert isinstance(
        mlp_regressor.named_steps["regressor"],
        TransformedTargetRegressor,
    )
    assert isinstance(
        mlp_regressor.named_steps["regressor"].regressor,
        MLPRegressor,
    )
    assert (
        mlp_classifier.named_steps["classifier"].class_weight == "balanced"
    )
    assert isinstance(
        linear_classifier.named_steps["classifier"],
        ChoquetClassifier,
    )
    assert isinstance(
        linear_classifier.named_steps["preprocessor"].named_steps["scale"],
        CapacityNormalizer,
    )
    assert linear_classifier.named_steps["classifier"].learn_feature_scales
    assert linear_classifier.named_steps["classifier"].solver_options == {
        "seed": 42
    }
    assert isinstance(
        mlp_regressor.named_steps["preprocessor"].named_steps["scale"],
        StandardScaler,
    )
    assert isinstance(
        fuzzy_regressor.named_steps["preprocessor"].named_steps["scale"],
        CapacityNormalizer,
    )
    assert fuzzy_regressor.named_steps["regressor"].mlp_solver == "adam"
    assert fuzzy_classifier.named_steps["classifier"].mlp_solver == "adam"
    assert isinstance(
        fuzzy_regressor.named_steps["regressor"],
        FuzzyChoquetNeuralRegressor,
    )
    assert isinstance(
        fuzzy_classifier.named_steps["classifier"],
        FuzzyChoquetNeuralClassifier,
    )


def test_ols_orientation_ablation_preserves_predictions() -> None:
    """Show that sign orientation does not change unrestricted OLS predictions."""
    rng = np.random.default_rng(12)
    X = pd.DataFrame(
        rng.normal(size=(80, 3)),
        columns=["positive", "negative", "noise"],
    )
    y = pd.Series(
        1.5 * X["positive"] - 2.0 * X["negative"] + 0.1 * X["noise"]
    )
    candidates = regression_candidates(
        3,
        include_mlp=False,
        include_dummy=False,
        include_regularized_choquet=False,
    )
    plain = candidates["OLS"].estimator.fit(X, y)
    oriented = candidates["OLS oriented"].estimator.fit(X, y)

    np.testing.assert_allclose(
        plain.predict(X),
        oriented.predict(X),
        atol=1e-10,
        rtol=1e-10,
    )


def test_logistic_orientation_ablation_preserves_probabilities() -> None:
    """Show that sign orientation does not change unrestricted logistic fits."""
    rng = np.random.default_rng(21)
    X = pd.DataFrame(
        rng.normal(size=(160, 3)),
        columns=["x1", "x2", "x3"],
    )
    y = pd.Series(
        (
            1.2 * X["x1"]
            - 0.9 * X["x2"]
            + 0.2 * rng.normal(size=len(X))
            > 0
        ).astype(int)
    )
    candidates = classification_candidates(3, include_mlp=False)
    plain = candidates["Logistic"].estimator.fit(X, y)
    oriented = candidates["Logistic oriented"].estimator.fit(X, y)

    np.testing.assert_allclose(
        plain.predict_proba(X),
        oriented.predict_proba(X),
        atol=1e-10,
        rtol=1e-10,
    )


def test_classifier_registry_exposes_balanced_and_unweighted_variants() -> None:
    balanced = classification_candidates(
        3,
        include_mlp=True,
        class_weight="balanced",
    )
    unweighted = classification_candidates(
        3,
        include_mlp=True,
        class_weight=None,
    )
    parameter_by_model = {
        "Logistic": "classifier__class_weight",
        "Logistic oriented": "classifier__class_weight",
        "Penalized logistic": "classifier__class_weight",
        "Explicit interactions": "classifier__logistic__class_weight",
        "RBF SVM": "classifier__class_weight",
        "Random forest": "classifier__class_weight",
        "Gradient boosting": "classifier__class_weight",
        "MLP": "classifier__class_weight",
        "Choquistic 1-additive": "classifier__class_weight",
        "Choquistic 2-additive": "classifier__class_weight",
        "Fuzzy Choquet neural network": "classifier__class_weight",
    }

    for model, parameter in parameter_by_model.items():
        assert balanced[model].estimator.get_params()[parameter] == "balanced"
        assert unweighted[model].estimator.get_params()[parameter] is None


def test_orientation_threshold_and_chronological_sign_stability() -> None:
    """Reject weak and unstable directions while retaining their diagnostics."""
    target_values = np.tile(np.arange(10, dtype=float), 3)
    weak_values = np.tile([1.0, -1.0] * 5, 3)
    unstable_values = np.concatenate(
        [
            -np.arange(10, dtype=float),
            -np.arange(10, dtype=float),
            0.2 * np.arange(10, dtype=float),
        ]
    )
    X = pd.DataFrame(
        {
            "stable negative": -target_values,
            "unstable negative": unstable_values,
            "weak": weak_values,
        }
    )
    transformer = CorrelationOrientationTransformer(
        minimum_absolute_correlation=0.2,
        stability_subperiods=3,
        require_sign_stability=True,
    ).fit(X, pd.Series(target_values))
    table = transformer.orientation_table()

    assert table.loc["stable negative", "orientation_sign"] == -1.0
    assert bool(table.loc["stable negative", "sign_stable"])
    assert abs(table.loc["unstable negative", "training_correlation"]) >= 0.2
    assert not bool(table.loc["unstable negative", "sign_stable"])
    assert table.loc["unstable negative", "orientation_sign"] == 1.0
    assert not bool(table.loc["weak", "meets_correlation_threshold"])
    assert table.loc["weak", "orientation_sign"] == 1.0
    assert "training_subperiod_3_correlation" in table.columns


def test_orientation_tables_skip_direct_prediction_aggregators() -> None:
    """Aggregators have no raw-feature orientation and must not break reporting."""
    X = pd.DataFrame(
        {"feature": np.linspace(-1.0, 1.0, 30)},
        index=pd.date_range("2020-01-01", periods=30),
    )
    y = pd.Series(-X["feature"], index=X.index)
    oriented = regression_candidates(
        1,
        include_mlp=False,
        include_regularized_choquet=False,
    )[
        "OLS oriented"
    ].estimator.fit(X, y)

    table = orientation_tables(
        {"OLS oriented": oriented, "Choquet model aggregator": object()},
        key_names=("model",),
    )

    assert set(table.index.get_level_values("model")) == {"OLS oriented"}


def test_final_refit_freezes_training_only_orientation() -> None:
    """Ensure validation can select a model but cannot change feature direction."""
    train_index = pd.date_range("2017-01-01", periods=30)
    validation_index = pd.date_range("2018-01-01", periods=30)
    X_train = pd.DataFrame(
        {"feature": np.linspace(-1.0, 1.0, len(train_index))},
        index=train_index,
    )
    y_train = pd.Series(-X_train["feature"], index=train_index)
    X_validation = pd.DataFrame(
        {"feature": np.linspace(-10.0, 10.0, len(validation_index))},
        index=validation_index,
    )
    y_validation = pd.Series(X_validation["feature"], index=validation_index)
    candidate = regression_candidates(
        1,
        include_mlp=False,
        include_dummy=False,
        include_regularized_choquet=False,
    )["OLS oriented"]
    selected = select_regression_model(
        "OLS oriented",
        candidate,
        X_train,
        y_train,
        X_validation,
        y_validation,
    )
    training_orient = selected.estimator.named_steps["preprocessor"].named_steps[
        "orient"
    ]
    fitted, _ = refit_selected(
        selected,
        X_train,
        y_train,
        X_validation,
        y_validation,
    )
    final_orient = fitted.named_steps["preprocessor"].named_steps["orient"]

    np.testing.assert_allclose(
        final_orient.correlations_,
        training_orient.correlations_,
    )
    np.testing.assert_array_equal(final_orient.signs_, training_orient.signs_)
    assert final_orient.training_observations_ == len(X_train)
    assert final_orient.orientation_source_.startswith("training only")


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


def test_scaled_q_choquet_recovers_positive_linear_response_scale() -> None:
    """A scaled 1-additive capacity is a positive linear model with free scale."""
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
    weights = np.array([0.55, 0.15, 0.30])
    y = pd.Series(0.2 + 2.5 * (X.to_numpy() @ weights))
    fitted = regression_candidates(
        3,
        include_mlp=False,
        include_dummy=False,
        include_regularized_choquet=False,
    )["Choquet 1-additive scaled-q"].estimator.fit(X, y)

    np.testing.assert_allclose(fitted.predict(X), y, atol=1e-5, rtol=1e-5)
    assert fitted_q(fitted) == pytest.approx(2.5, abs=1e-4)


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
