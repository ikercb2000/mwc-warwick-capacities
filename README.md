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

Alternatively, run the complete sequence and final artifact audit with:

```powershell
.\scripts\run_all_experiments.ps1
```

Pass `--quick` for a smoke-sized run:

```powershell
.\scripts\run_all_experiments.ps1 --quick
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
| `experiment_factors.py` | Compare linear, nonlinear and Choquet factor models for each equity. |
| `experiment_predict_loss.py` | Forecast 1-, 5- and 10-day portfolio losses and test portfolio-weight robustness. |
| `experiment_tail_risk.py` | Classify 95% tail events and run a 97.5% robustness specification. |
| `experiment_distortion_risk.py` | Simulate regime-switching losses and study distortion capital, backtests and diversification. |
| `experiment_autoregression.py` | Compare linear and Choquet autoregressions and evaluate high-volatility performance. |
| `run_smoke_experiments.py` | Run representative model-family checks without producing the complete report. |

## Model selection and evaluation

Predictive experiments use chronological train, validation and test samples.
The boundaries used by each executable are configured in its file under
`configs/experiments/`:

- regression training ends on 2018-12-31;
- classification training ends on 2017-12-31;
- validation ends on 2019-12-31;
- later observations form the held-out test sample.

Forecast-horizon observations are purged from the end of training and
validation partitions to prevent overlapping future targets from crossing split
boundaries. Preprocessing is fitted only on the relevant training sample.
The Amihud liquidity feature is normalised point-in-time: each date uses the
expanding median of positive portfolio illiquidity observed strictly before that
date, rather than a full-sample median.

Each candidate is selected using validation performance, refitted on the
combined training and validation samples, and evaluated on the test sample.
Regression grids minimise validation RMSE; classification grids maximise
validation PR AUC, while probability thresholds are also selected from
validation data.

Regression benchmarks include OLS, monotone linear regression, regularised
linear models, explicit interactions, trees, boosting, neural networks and
Choquet regressions. Choquet specifications include 1-additive, 2-additive and
interaction-regularised 2-additive capacities. Capacity order and interaction
regularisation are selected using validation RMSE, never test performance.

Classical models use clipping and their usual scaling without target-driven
feature orientation. Capacity models are additionally oriented using
training-sample correlations and scaled to a common interval because their
monotonicity constraints require a common increasing direction. The monotone
linear benchmark uses the same capacity preprocessing.

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
models. Thresholds are estimated on the combined training and validation
sample: an observation is extreme when its raw loss is at or above the 99th
percentile, or when any raw predictor lies outside its 0.5th--99.5th percentile
interval. Those fixed thresholds are then applied to the held-out test sample.
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

The primary split is unchanged: its validation sample ends in 2019, so the
2020--2021 financial-stress period belongs to the primary test and cannot affect
the main hyperparameter selection. A separate robustness design tests whether
results would change if that episode were available during validation.

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

The notebooks are ordered from data preparation through the five empirical
experiments:

1. `00_data_preparation_and_eda.ipynb`
2. `01_choquet_factor_models.ipynb`
3. `02_future_loss_regression.ipynb`
4. `03_tail_risk_classification.ipynb`
5. `04_distortion_risk_simulation.ipynb`
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
