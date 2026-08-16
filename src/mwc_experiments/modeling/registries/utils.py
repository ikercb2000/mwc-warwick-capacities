"""Registries domain."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Literal
from sklearn.base import BaseEstimator, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.svm import SVC
from capacities_ml_fin.ml.models import ChoquetRegressor, ChoquisticRegression
from capacities_ml_fin.ml.optimization import KAdditivity, L1Penalty
from mwc_experiments.modeling.preprocessing import (
    make_capacity_preprocessor,
    make_oriented_standard_preprocessor,
    make_standard_preprocessor,
)
from mwc_experiments.modeling.types import Candidate

def _model_pipeline(
    preprocessor: Pipeline,
    estimator: BaseEstimator,
    *,
    step_name: Literal["regressor", "classifier"],
) -> Pipeline:
    """Combine a cloned feature preprocessor and estimator in one sklearn pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (step_name, estimator),
        ]
    )


def _target_scaled_regressor(regressor: BaseEstimator) -> TransformedTargetRegressor:
    """Wrap a regressor with reversible unit-interval target scaling."""
    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=MinMaxScaler(feature_range=(0.0, 1.0), clip=False),
    )


def _interaction_penalty(n_features: int, weight: float) -> L1Penalty:
    """Penalize only pairwise Möbius terms of a 2-additive capacity."""
    compilation = KAdditivity(order=2).compile(n_features)
    selected = [
        position
        for position, mask in enumerate(compilation.bundle.parameter_masks)
        if int(mask).bit_count() == 2
    ]
    return L1Penalty(weight=weight, selection=selected)


def apply_parameter_grid_overrides(
    candidates: dict[str, Candidate],
    overrides: Mapping[str, Mapping[str, list[Any]]] | None,
    *,
    n_features: int,
) -> dict[str, Candidate]:
    """Replace candidate grids with values declared by an experiment TOML."""
    if not overrides:
        return candidates
    result = dict(candidates)
    unknown_models = sorted(set(overrides).difference(result))
    if unknown_models:
        raise KeyError(
            "Configuration contains unknown models: " + ", ".join(unknown_models)
        )
    for model_name, raw_grid in overrides.items():
        candidate = result[model_name]
        valid_parameters = candidate.estimator.get_params(deep=True)
        grid: dict[str, list[Any]] = {}
        for parameter, raw_values in raw_grid.items():
            if parameter not in valid_parameters:
                raise KeyError(
                    f"Unknown parameter {parameter!r} for model {model_name!r}."
                )
            values: list[Any] = list(raw_values)
            if parameter.endswith("__penalty"):
                values = [
                    _interaction_penalty(n_features, float(weight))
                    for weight in values
                ]
            elif parameter.endswith("__hidden_layer_sizes"):
                values = [
                    tuple(value) if isinstance(value, list) else value
                    for value in values
                ]
            grid[parameter] = values
        result[model_name] = Candidate(candidate.estimator, grid)
    return result


def regression_candidates(
    n_features: int,
    *,
    random_state: int = 42,
    include_mlp: bool = True,
    include_dummy: bool = True,
    include_regularized_choquet: bool = True,
) -> dict[str, Candidate]:
    """Build regression candidates and validation grids as sklearn pipelines."""
    if n_features < 1:
        raise ValueError("n_features must be positive.")
    standard = make_standard_preprocessor()
    oriented_standard = make_oriented_standard_preprocessor()
    capacity = make_capacity_preprocessor()

    def pipeline(regressor: BaseEstimator, *, capacity_input: bool = False) -> Pipeline:
        """Build one regression pipeline with the appropriate feature scaling."""
        preprocessor = capacity if capacity_input else standard
        return _model_pipeline(preprocessor, regressor, step_name="regressor")

    candidates: dict[str, Candidate] = {}
    if include_dummy:
        candidates["Historical mean"] = Candidate(
            pipeline(DummyRegressor(strategy="mean")),
            {},
        )
    candidates.update(
        {
            "OLS": Candidate(pipeline(LinearRegression()), {}),
            "OLS oriented": Candidate(
                _model_pipeline(
                    oriented_standard,
                    LinearRegression(),
                    step_name="regressor",
                ),
                {},
            ),
            "Monotone linear": Candidate(
                pipeline(
                    _target_scaled_regressor(LinearRegression(positive=True)),
                    capacity_input=True,
                ),
                {},
            ),
            "Ridge": Candidate(
                pipeline(Ridge()),
                {"regressor__alpha": [0.01, 0.1, 1.0, 10.0]},
            ),
            "Lasso": Candidate(
                pipeline(Lasso(max_iter=20_000)),
                {"regressor__alpha": [1e-5, 1e-4, 1e-3, 1e-2]},
            ),
            "Elastic net": Candidate(
                pipeline(ElasticNet(max_iter=20_000)),
                {
                    "regressor__alpha": [1e-5, 1e-4, 1e-3],
                    "regressor__l1_ratio": [0.25, 0.5, 0.75],
                },
            ),
            "Explicit interactions": Candidate(
                pipeline(
                    Pipeline(
                        [
                            (
                                "interactions",
                                PolynomialFeatures(
                                    degree=2,
                                    interaction_only=True,
                                    include_bias=False,
                                ),
                            ),
                            ("ridge", Ridge()),
                        ]
                    )
                ),
                {"regressor__ridge__alpha": [0.01, 0.1, 1.0, 10.0]},
            ),
            "Random forest": Candidate(
                pipeline(
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=random_state,
                        n_jobs=-1,
                    )
                ),
                {
                    "regressor__max_depth": [3, 6, None],
                    "regressor__min_samples_leaf": [5, 20],
                },
            ),
            "Gradient boosting": Candidate(
                pipeline(HistGradientBoostingRegressor(random_state=random_state)),
                {
                    "regressor__learning_rate": [0.03, 0.1],
                    "regressor__max_leaf_nodes": [7, 15],
                    "regressor__l2_regularization": [0.0, 1.0],
                },
            ),
            "Choquet 1-additive": Candidate(
                pipeline(
                    _target_scaled_regressor(
                        ChoquetRegressor(
                            sparsity=KAdditivity(order=1),
                            solver="scipy",
                            solver_options={"options": {"maxiter": 2_000}},
                        )
                    ),
                    capacity_input=True,
                ),
                {},
            ),
            "Choquet 2-additive": Candidate(
                pipeline(
                    _target_scaled_regressor(
                        ChoquetRegressor(
                            sparsity=KAdditivity(order=2),
                            solver="scipy",
                            solver_options={"options": {"maxiter": 2_000}},
                        )
                    ),
                    capacity_input=True,
                ),
                {},
            ),
        }
    )

    if include_regularized_choquet:
        candidates["Choquet 2-additive L1"] = Candidate(
            pipeline(
                _target_scaled_regressor(
                    ChoquetRegressor(
                        sparsity=KAdditivity(order=2),
                        solver="scipy",
                        solver_options={"options": {"maxiter": 2_000}},
                        penalty=_interaction_penalty(n_features, 1e-3),
                    )
                ),
                capacity_input=True,
            ),
            {
                "regressor__regressor__penalty": [
                    _interaction_penalty(n_features, weight)
                    for weight in (1e-5, 1e-4, 1e-3, 1e-2)
                ]
            },
        )

    if include_mlp:
        candidates["MLP"] = Candidate(
            pipeline(
                MLPRegressor(
                    max_iter=2_000,
                    early_stopping=True,
                    random_state=random_state,
                )
            ),
            {
                "regressor__hidden_layer_sizes": [(32,), (32, 16)],
                "regressor__alpha": [1e-4, 1e-3],
            },
        )
    return candidates


def classification_candidates(
    n_features: int,
    *,
    random_state: int = 42,
    include_mlp: bool = True,
    class_weight: str | None = "balanced",
) -> dict[str, Candidate]:
    """Build classification candidates and validation grids as sklearn pipelines."""
    if n_features < 1:
        raise ValueError("n_features must be positive.")
    if class_weight not in {"balanced", None}:
        raise ValueError("class_weight must be 'balanced' or None.")
    standard = make_standard_preprocessor()
    oriented_standard = make_oriented_standard_preprocessor()
    capacity = make_capacity_preprocessor()

    def pipeline(classifier: BaseEstimator, *, capacity_input: bool = False) -> Pipeline:
        """Build one classifier pipeline with the appropriate feature scaling."""
        preprocessor = capacity if capacity_input else standard
        return _model_pipeline(preprocessor, classifier, step_name="classifier")

    candidates: dict[str, Candidate] = {
        "Rolling prior probability": Candidate(
            pipeline(DummyClassifier(strategy="prior")),
            {},
        ),
        "Logistic": Candidate(
            pipeline(
                LogisticRegression(
                    C=1e6,
                    max_iter=5_000,
                    class_weight=class_weight,
                )
            ),
            {},
        ),
        "Logistic oriented": Candidate(
            _model_pipeline(
                oriented_standard,
                LogisticRegression(
                    C=1e6,
                    max_iter=5_000,
                    class_weight=class_weight,
                ),
                step_name="classifier",
            ),
            {},
        ),
        "Penalized logistic": Candidate(
            pipeline(
                LogisticRegression(
                    max_iter=5_000,
                    class_weight=class_weight,
                )
            ),
            {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Explicit interactions": Candidate(
            pipeline(
                Pipeline(
                    [
                        (
                            "interactions",
                            PolynomialFeatures(
                                degree=2,
                                interaction_only=True,
                                include_bias=False,
                            ),
                        ),
                        (
                            "logistic",
                            LogisticRegression(
                                max_iter=5_000,
                                class_weight=class_weight,
                            ),
                        ),
                    ]
                )
            ),
            {"classifier__logistic__C": [0.01, 0.1, 1.0]},
        ),
        "RBF SVM": Candidate(
            pipeline(
                SVC(
                    probability=True,
                    class_weight=class_weight,
                    random_state=random_state,
                )
            ),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__gamma": ["scale", 0.1],
            },
        ),
        "Random forest": Candidate(
            pipeline(
                RandomForestClassifier(
                    n_estimators=400,
                    class_weight=class_weight,
                    random_state=random_state,
                    n_jobs=-1,
                )
            ),
            {
                "classifier__max_depth": [3, 6, None],
                "classifier__min_samples_leaf": [5, 20],
            },
        ),
        "Gradient boosting": Candidate(
            pipeline(
                HistGradientBoostingClassifier(
                    random_state=random_state,
                    class_weight=class_weight,
                )
            ),
            {
                "classifier__learning_rate": [0.03, 0.1],
                "classifier__max_leaf_nodes": [7, 15],
                "classifier__l2_regularization": [0.0, 1.0],
            },
        ),
        "Choquistic 1-additive": Candidate(
            pipeline(
                ChoquisticRegression(
                    sparsity=KAdditivity(order=1),
                    solver="scipy",
                    solver_options={"options": {"maxiter": 2_000}},
                    class_weight=class_weight,
                ),
                capacity_input=True,
            ),
            {},
        ),
        "Choquistic 2-additive": Candidate(
            pipeline(
                ChoquisticRegression(
                    sparsity=KAdditivity(order=2),
                    solver="scipy",
                    solver_options={"options": {"maxiter": 2_000}},
                    class_weight=class_weight,
                ),
                capacity_input=True,
            ),
            {},
        ),
    }
    if include_mlp:
        candidates["MLP"] = Candidate(
            pipeline(
                MLPClassifier(
                    max_iter=2_000,
                    early_stopping=True,
                    random_state=random_state,
                )
            ),
            {
                "classifier__hidden_layer_sizes": [(32,), (32, 16)],
                "classifier__alpha": [1e-4, 1e-3],
            },
        )
    return candidates
