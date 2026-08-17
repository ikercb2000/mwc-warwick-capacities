# MSc Warwick Capacities

Reproducible empirical code for the MSc Mathematical Finance dissertation at
the University of Warwick. The repository studies Choquet-capacity methods and
standard machine-learning benchmarks in financial factor modelling, future-loss
forecasting, tail-event classification, distortion risk and autoregression.

## How the repository works

The project separates reusable modelling code, executable experiments and
presentation:

```text
Raw Bloomberg/FRED files
          |
          v
scripts/build_experiment_data.py
          |
          v
Model-ready Parquet datasets
          |
          v
scripts/experiment_*.py
          |
          v
CSV/Parquet tables + PNG figures
          |
          v
Read-only reporting notebooks
```

The scripts contain the complete experiment pipelines and are intended to be
executed with `poetry run python`. They are not importable modules: importing a
file from `scripts/` also executes its pipeline.

Reusable implementation is organised as follows:

| Path | Responsibility |
| --- | --- |
| `src/mwc_experiments/data/` | Load Bloomberg/FRED inputs, construct returns, portfolios, predictors and targets, and persist processed datasets. |
| `src/mwc_experiments/modeling/` | Preprocessing, chronological splits, candidate registries, hyperparameter selection and estimator inspection. |
| `src/mwc_experiments/workflows/` | Reusable factor, forecasting, classification, risk and autoregression procedures used by the scripts. |
| `src/mwc_experiments/evaluation/` | Metrics, inference, capacity interpretation and plotting utilities. |
| `src/mwc_experiments/reporting/` | Load saved tables and figures without recomputing experiments. |
| `configs/experiment_settings.toml` | Shared market-data universe, source mappings and reusable library defaults. |
| `configs/experiments.toml` | Small index mapping experiment identifiers to their own TOML files. |
| `configs/experiments/*.toml` | Models, grids, horizons, samples and analysis parameters owned by each executable experiment. |
| `scripts/` | Directly executable data and experiment pipelines. |
| `notebooks/` | Lightweight reports over persisted artifacts. |
| `tests/` | Data, split, model-selection and architectural checks. |

Every reusable domain follows the same internal convention:

```text
domain/
|-- __init__.py   # stable public API and re-exports
|-- types.py      # classes, dataclasses and exceptions, when present
|-- utils.py      # functions and runtime state
|-- mappings.py   # dictionary mappings, when present
`-- constants.py  # non-mapping settings constants, when needed
```

For example, run management is implemented in `mwc_experiments/runs/`, with
`ExperimentRunPaths` in `types.py`, run operations in `utils.py`, and artifact
and script mappings in `mappings.py`. Callers import from the domain package
(`from mwc_experiments.runs import ...`) rather than from its internal modules.
An architectural test enforces this separation throughout `src/mwc_experiments`.

## Installation

The project requires Python 3.12–3.14 and Poetry.

```powershell
poetry install
poetry run pytest -q
```

Run commands from the repository root. Repository paths are discovered from the
nearest directory containing both `pyproject.toml` and `data/`.

## Input data

Raw inputs are expected under `data/raw/`:

```text
data/raw/
|-- bloomberg/
|   `-- market_data/
|       |-- equities/daily/
|       `-- etfs/daily/
`-- fred/
```

Bloomberg equity files use names such as
`AAPL_daily_market_data.xlsx`. ETF files use names such as
`SPY_daily_etf_data.xlsx`. FRED files must contain `date` and `value`
columns and follow the pattern `{SERIES_ID}_*.csv`.

Bloomberg files are licensed inputs and are not stored in Git. The complete
acquisition contract—including securities, Bloomberg mnemonics, required
fields, workbook headers, date parsing, units and validation commands—is in
[`data/raw/README.md`](data/raw/README.md). Quarterly fundamental exports are
not required by the current experiments.

The raw-data universe and FRED mappings are defined in
`configs/experiment_settings.toml`. Target horizons and point-in-time tail
labels are controlled by `configs/experiments/data_preparation.toml`; changing
that file invalidates the processed-data fingerprint and triggers a rebuild.

## Execution order

Build the common data layer first:

```powershell
poetry run python scripts/build_experiment_data.py
```

This command:

1. loads and aligns Bloomberg and FRED observations;
2. constructs equity and ETF returns;
3. builds equal-weight and lagged market-cap-weight portfolios;
4. creates factor frames, risk predictors and forward-loss targets;
5. computes point-in-time tail thresholds and event labels;
6. saves model-ready datasets, audits, summary tables and EDA figures.

Processed datasets are written to `data/experiments/`. Experiment scripts also
call `load_or_build_processed_data()`. The data manifest hashes the raw inputs,
configuration and data-building source code. Missing, modified or stale Parquet
files are rebuilt automatically. Running the build script explicitly is still
recommended because it produces the complete data audit and EDA output.

Run experiments independently after preparing the data:

```powershell
poetry run python scripts/experiment_factors.py --full
poetry run python scripts/experiment_predict_loss.py --full
poetry run python scripts/experiment_tail_risk.py --full
poetry run python scripts/experiment_distortion_risk.py
poetry run python scripts/experiment_autoregression.py
```

Alternatively, run the complete sequence, final artifact audit and HTML report
rendering with the platform-specific wrapper.

Windows PowerShell:

```powershell
.\scripts\run_all_experiments.ps1
```

macOS/Linux:

```bash
bash scripts/run_all_experiments.sh
```

Pass `--quick` for a smoke-sized run:

```powershell
.\scripts\run_all_experiments.ps1 --quick
```

```bash
bash scripts/run_all_experiments.sh --quick
```

Pass `--no-render` if the experiments should finish without executing the
reporting notebooks:

```powershell
.\scripts\run_all_experiments.ps1 --full --no-render
```

```bash
bash scripts/run_all_experiments.sh --full --no-render
```

Inspect existing runs and their status with:

```powershell
poetry run python scripts/list_runs.py
```

The three predictive scripts accept:

- `--quick`: use one representative configuration per model family;
- `--full`: evaluate the complete validation grids;
- `--quiet`: suppress model-by-model progress messages.

If no mode is given, `quick_mode_default` from the TOML configuration is used.
The smoke script provides a small end-to-end check:

```powershell
poetry run python scripts/run_smoke_experiments.py
```

## Experiments

| Script | Purpose |
| --- | --- |
| `build_experiment_data.py` | Construct processed datasets and generate data audits and EDA. |
| `experiment_factors.py` | Compare classical linear and 1-additive Choquet factor models for each equity. |
| `experiment_predict_loss.py` | Forecast 1-, 5- and 10-day portfolio losses and test portfolio-weight robustness. |
| `experiment_tail_risk.py` | Classify 95% tail events and run a 97.5% robustness specification. |
| `experiment_distortion_risk.py` | Study distortion capital, backtests and diversification using observed Bloomberg equity losses. |
| `experiment_autoregression.py` | Compare linear and Choquet autoregressions and evaluate high-volatility performance. |
| `run_smoke_experiments.py` | Run representative model-family checks without producing the complete report. |

## Model selection and evaluation

Predictive experiments 2a and 2b use rolling walk-forward evaluation. The first
OOS block starts in 2020 and each block covers one year (the final block may be
partial). Before predicting a block, the complete selection and fitting process
uses only the configurable five-year lookback immediately preceding it. After
the block finishes, both the lookback and OOS boundaries advance by one year.
All OOS block predictions are concatenated before calculating final metrics.

For 2a, the last 12 months of each lookback are internal validation and the
earlier four years are inner training. After hyperparameter selection, the model
is refitted using both past partitions. For 2b, each five-year lookback is split
into 18 months of inner training, 18 months of selection validation and 24
months of probability calibration. The base classifier is refitted on the first
three years, then its complete fitted pipeline is frozen and calibrated on the
final two years.

Forecast-horizon observations are purged before every inner-validation,
calibration and OOS boundary, preventing 5-day and 10-day targets from crossing
into the next partition. Orientation, clipping, scaling, hyperparameters and
Choquet capacities are newly estimated in every fold. No preprocessing or
selection state is carried forward, although outcomes from completed OOS blocks
may legitimately enter later rolling windows as past observations.
The Amihud liquidity feature is normalised point-in-time: each date uses the
expanding median of positive portfolio illiquidity observed strictly before that
date, rather than a full-sample median.

Regression candidates are selected using each fold's validation performance,
refitted on that fold's past training and validation samples, and evaluated on
the immediately following OOS block.
Regression grids minimise validation RMSE. Classification candidates maximise
PR AUC on the chronological selection-validation block and are then refitted on
training plus that block. The later calibration block is not included in model
selection or base-model fitting.

Experiment 2b fits every classifier that supports class weighting in two
separate forms: `class_weight="balanced"` and unweighted (`class_weight=None`).
The rolling-prior benchmark, conventional MLP and package
`Choquet linear classifier` are fitted once because they do not expose
classifier weights. The latter is the package's deterministic
`ChoquetClassifier`: it learns a 2-additive capacity, non-negative feature
scales and a decision threshold. Its direct `[0, 1]` Choquet score is used for
uncalibrated discrimination, while its `[sigmoid]` variant is the proper
probability estimate calibrated on the chronological calibration block.
Each fitted variant is evaluated uncalibrated and
with sigmoid calibration. `CalibratedClassifierCV` is applied to a
`FrozenEstimator` containing the complete fitted pipeline, so calibration
cannot refit preprocessing or the classifier. Its
decision threshold is estimated on the calibration block; the uncalibrated
threshold remains selection-validation based. The current or future OOS block
is never used for selection, fitting, threshold choice or calibration.

The two 2b aggregators also remain weight-specific. The balanced aggregator
receives only balanced base classifiers plus weight-independent controls; the
unweighted aggregator receives only unweighted base classifiers plus those
same controls. They do not combine both versions of every classifier in one
capacity. Both aggregators use a 2-additive capacity
(`KAdditivity(order=2)`), as does the single regression aggregator in 2a;
neither currently adds an L1 penalty.

The no-feature benchmark is named `Rolling prior probability`: it predicts a
constant prevalence within each OOS block, but re-estimates that prevalence
from the past window before every new block. It is therefore constant within a
fold, not across the complete concatenated OOS sample.

Discrimination is reported separately with ROC-AUC and PR-AUC. Probability
calibration is reported with Brier score, log loss, mean predicted probability,
observed event prevalence and their calibration gap. Window lengths and block
size is configurable through `[walk_forward]`; calibration and weighting are
configured under `[calibration]` and `[class_weight]` in
`configs/experiments/tail_risk.toml`. Separate
`experiment_2b_discrimination_*`, `experiment_2b_calibration_*` and
`experiment_2b_calibration_sample_*` tables make the comparison auditable.
Both forecasting experiments additionally persist `*_walk_forward_folds_*`,
`*_walk_forward_metrics_*`, `*_orientation_history_*` and
`*_shapley_history_*` tables.

Experiments 2a and 2b compare a conventional `MLP` with
`Fuzzy Choquet neural network`. The latter first learns a supervised
2-additive Choquet integral of the input variables and then passes the resulting
aggregate through a tanh neural network. Both its capacity layer and neural
parameters are re-estimated inside every rolling window. The classifier is
evaluated with both balanced and unweighted fitting, like the other classifiers
that expose `class_weight`; scikit-learn's conventional MLP has no
`class_weight` parameter and is therefore fitted once per fold.
The conventional network receives clipped, standardised predictors. The fuzzy
network receives clipped, oriented predictors normalised to `[0, 1]`, as
required by the Choquet capacity, and its Choquet output is also bounded before
entering the neural part. Both neural benchmarks use the `adam` optimiser; this
avoids the repeated `lbfgs` iteration-limit failures produced by the package's
default solver during walk-forward fitting.

Experiments 2a and 2b also include a 2-additive Choquet model aggregator. It is
deliberately fitted only from the outputs of the configured classical models;
the Choquet, Choquistic and fuzzy-Choquet candidates are excluded from its
inputs. The conventional MLP is eligible because it is a classical non-Choquet
model. In 2a the aggregation capacity is fitted on the inner-validation
predictions and then applied to the next OOS block. In 2b it is fitted on the
selection-validation probabilities; its optional sigmoid calibration is fitted
on the later calibration block before evaluating the next OOS block. Therefore,
the aggregation capacity, calibration and final comparison never use the test
block. The aggregator's fit-block score is descriptive and in-sample; model
comparisons should use the concatenated OOS metrics.

Experiment 1 compares OLS, oriented OLS, monotone linear regression, Ridge,
Lasso and Elastic Net with six Choquet specifications: 1- and 2-additive,
their scaled-q versions, and L1-regularised versions of both 2-additive
models. The 1-additive models remain positively monotone linear
specifications; the 2-additive models add pairwise interactions. A scaled-q
model estimates `intercept + q * C_mu(X)` with `q >= 0`, removing the fixed
overall scale without removing monotonicity. L1 penalises only pairwise
Möbius terms. Fitted q values, Shapley indices and pairwise interactions are
persisted in the experiment tables.

Experiment 2a retains the same complete set of six Choquet specifications,
alongside its broader nonlinear benchmarks. Thus scaled-q models supplement
rather than replace the ordinary 1-additive, 2-additive and 2-additive-L1
versions. Hyperparameters are selected using validation performance, never
test performance.

Classical models use clipping and their usual scaling without target-driven
feature orientation. Capacity models are additionally oriented using
training-sample correlations and scaled to a common interval because their
monotonicity constraints require a common increasing direction. The monotone
linear benchmark uses the same capacity preprocessing.

Orientation is governed by `[orientation]` in
`configs/experiment_settings.toml`. By default, a feature is eligible for
orientation only when its absolute training correlation is at least `0.2`.
The transformer also reports sign agreement across three contiguous
chronological training subperiods; set `require_sign_stability = true` to leave
features with an unstable subperiod sign unchanged. Validation may still select
hyperparameters, but every fold's train-validation refit freezes the
correlations and signs learned strictly on that fold's inner training sample.
Its OOS observations are never involved.
The experiment outputs include complete `*_orientations*.csv` tables containing
the training correlations, subperiod correlations, stability decision and final
orientation of every oriented fitted model.

The experiment registries also include `OLS oriented` and
`Logistic oriented` controls. Dedicated orientation-ablation tables compare
the ordinary classical model, its oriented counterpart, the monotone linear
benchmark where applicable, and the Choquet/Choquistic models. This makes it
possible to distinguish effects caused by orientation from those caused by
monotonicity, capacity normalisation and interactions.

### Extremes, clipping and factor-risk diagnostics

Clipping is part of each fitted preprocessing pipeline. Its bounds are estimated
without the test sample and it caps predictor values passed to the estimator; it
does not discard observations, alter the target or overwrite the raw datasets.
The experiment scripts save clipping audits with the raw minima and maxima,
fitted bounds, and the number of observations affected in estimation and test
samples.

Regression robustness uses one empirical stress definition shared by all
models in the final rolling block. Thresholds are estimated on that block's
past training and validation sample: an observation is extreme when its raw loss is at or above the 99th
percentile, or when any raw predictor lies outside its 0.5th--99.5th percentile
interval. Those fixed thresholds are then applied to the final OOS block.
The saved stress tables therefore compare classical and Choquet models on the
same dates, before model-specific preprocessing.

A separate estimation-stability check clones each already selected model,
removes the extreme observations from its estimation sample, and refits it with
the same selected hyperparameters. The resulting tables report changes in test
and stress-period RMSE and in predictions. This isolates sensitivity of the
estimated model from performance conditional on an extreme event.

For the factor experiment, in-sample fitted values and residual covariance are
saved as descriptive risk diagnostics. They answer how the factor
representation decomposes the observed sample, but they are not used to rank
forecasting models. Model comparison and selection remain based on validation
and held-out out-of-sample results.

### Validation-regime robustness

In the primary rolling design, 2020--2021 is genuinely OOS when first predicted
and can only enter later windows after those outcomes have occurred. A separate
historical robustness design remains available to isolate how selection changes
when the stress episode is included or excluded from a common validation score.

This comparison generates annual walk-forward predictions from 2018 through
2021: every validation year is predicted by a model fitted on all strictly
earlier observations. One regime scores candidates on all those pseudo-OOS
predictions; the other excludes 2020--2021 from scoring. After selection, both
regimes refit on the identical complete sample through 2021 and use the common
post-2021 test. Consequently, differences in selected parameters, validation
ranks and test RMSE isolate the selection effect of the stressed period rather
than a difference in final training data. The fitted test models use the full
history through 2021, rather than only the short initial training window.
The relevant dates are configurable under `[validation_stress]` in
`factor_models.toml` and `future_loss.toml`. Use `--full` when analysing this
comparison because `--quick` retains only one hyperparameter combination per
model family.

Each experiment owns one self-contained configuration file:

```text
configs/experiments/
|-- data_preparation.toml
|-- factor_models.toml
|-- future_loss.toml
|-- tail_risk.toml
|-- distortion_risk.toml
`-- autoregression.toml
```

The predictive TOMLs contain explicit `[models]` lists and
`[parameter_grids.<model>]` tables. These grids replace the defaults in the
sklearn candidate registry and are validated against the actual pipeline
parameter names before fitting. The scripts load the values near the beginning
of the file, while the complete pipeline remains written directly in the
script.

## Outputs and notebooks

Every script execution creates a unique immutable run:

```text
data/
|-- experiments/
|   |-- manifest.json
|   `-- model-ready Parquet datasets
`-- results/
    |-- runs/{experiment}/{run_id}/
    |   |-- manifest.json
    |   |-- shared_config.toml
    |   |-- experiment_config.toml
    |   |-- SUCCESS
    |   |-- logs/run.log
    |   |-- tables/
    |   |-- figures/
    |   `-- models/
    |-- latest/{experiment}.json
    |-- tables/                 legacy/published snapshot
    `-- figures/                legacy/published snapshot
```

Run identifiers contain the UTC timestamp, Git commit, execution mode and
process identifier. A manifest records the command, Git state, Python and
package versions, configuration hash, data fingerprint, duration, artifact
sizes and SHA-256 checksums. Files are written atomically. `latest` is updated
only after every artifact required by the corresponding notebook exists and a
`SUCCESS` marker has been written. Failed executions therefore cannot replace a
valid report or mix with artifacts from another run. Local `runs/` and `latest/`
are ignored by Git; selected flat results can remain as a deliberate dissertation
snapshot.

Promote a reviewed latest run into that Git-trackable snapshot explicitly:

```powershell
poetry run python scripts/publish_results.py factor_models
```

Use `--run-id` to publish a historical run instead. Publication verifies every
checksum and records its origin under `data/results/published/`; ordinary
experiment execution never alters the flat dissertation snapshot.

Notebooks do not build data or fit models. They load the saved artifacts through
`mwc_experiments.reporting`, so they are inexpensive to open and re-execute. By
default, reporting uses the latest successful run and falls back to the legacy
flat snapshot only when no run has yet been published. To reproduce a specific
factor run in PowerShell, set its identifier before starting Jupyter:

```powershell
$env:MWC_RUN_FACTOR_MODELS="20260815T120000Z_abcdef0_full_1234"
poetry run jupyter lab
```

Equivalent variables use the catalogue identifiers `DATA_PREPARATION`,
`FUTURE_LOSS`, `TAIL_RISK`, `DISTORTION_RISK` and `AUTOREGRESSION`.
Checksums are verified when an artifact is loaded.

Start Jupyter with:

```powershell
poetry run jupyter lab
```

Alternatively, execute all reporting notebooks and render standalone HTML
reports without modifying the source notebooks:

```powershell
.\scripts\render_notebooks.ps1
```

On macOS/Linux:

```bash
bash scripts/render_notebooks.sh
```

The generated files are written to `data/results/reports/`. They are local,
regenerable presentation artifacts and are ignored by Git. Rendering stops on
the first notebook error, so a completed command also verifies that every
notebook can load and display its selected run artifacts.

To open every generated report at once, use the command for your platform. The
browser will normally open one tab per report.

Windows PowerShell:

```powershell
Get-ChildItem .\data\results\reports\*.html | ForEach-Object {
    Start-Process $_.FullName
}
```

macOS:

```bash
open data/results/reports/*.html
```

Linux:

```bash
for report in data/results/reports/*.html; do
    xdg-open "$report"
done
```

The notebooks are ordered from data preparation through the five empirical
experiments:

1. `00_data_preparation_and_eda.ipynb`
2. `01_choquet_factor_models.ipynb`
3. `02_future_loss_regression.ipynb`
4. `03_tail_risk_classification.ipynb`
5. `04_distortion_risk.ipynb` — distortion-risk analysis with observed market losses
6. `05_choquet_autoregression_robustness.ipynb`

## Reproducibility checks

Run the complete test suite after changing data construction, preprocessing,
model registries or experiment architecture:

```powershell
poetry run pytest -q
```

Audit every notebook against the latest successful runs without opening Jupyter:

```powershell
poetry run python scripts/audit_results.py
```

The tests cover forward targets, lagged weights, chronological purging,
configuration loading, sklearn-compatible model selection, the relationship
between monotone linear and 1-additive Choquet regression, and the separation
between executable scripts and reporting notebooks. They also cover run
publication contracts, checksums, latest-run resolution and stale-data
invalidation.
