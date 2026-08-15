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
| `src/mwc_experiments/reporting.py` | Load saved tables and figures without recomputing experiments. |
| `configs/experiment_settings.toml` | Dates, assets, horizons, tail levels, random seed and feature lists. |
| `scripts/` | Directly executable data and experiment pipelines. |
| `notebooks/` | Lightweight reports over persisted artifacts. |
| `tests/` | Data, split, model-selection and architectural checks. |

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

The assets, FRED series and sample dates actually loaded are defined in
`configs/experiment_settings.toml`.

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
call `load_or_build_processed_data()`, so missing Parquet files are rebuilt
automatically. Running the build script explicitly is still recommended because
it produces the complete data audit and EDA output.

Run experiments independently after preparing the data:

```powershell
poetry run python scripts/experiment_factors.py --full
poetry run python scripts/experiment_predict_loss.py --full
poetry run python scripts/experiment_tail_risk.py --full
poetry run python scripts/experiment_distortion_risk.py
poetry run python scripts/experiment_autoregression.py
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
The default boundaries are configured in
`configs/experiment_settings.toml`:

- regression training ends on 2018-12-31;
- classification training ends on 2017-12-31;
- validation ends on 2019-12-31;
- later observations form the held-out test sample.

Forecast-horizon observations are purged from the end of training and
validation partitions to prevent overlapping future targets from crossing split
boundaries. Preprocessing is fitted only on the relevant training sample.

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

Capacity inputs are clipped, oriented using training-sample correlations and
scaled to a common interval. The monotone linear benchmark uses the same
capacity preprocessing, which separates the effect of monotonicity from the
additional normalisation and interaction structure imposed by Choquet models.

## Outputs and notebooks

Experiment artifacts are written to:

```text
data/experiments/       processed model-ready Parquet datasets
data/results/tables/    metrics, predictions, diagnostics and interpretation
data/results/figures/   generated PNG figures
data/results/models/    reserved model-output directory
```

Notebooks do not build data or fit models. They load the saved artifacts through
`mwc_experiments.reporting`, so they are inexpensive to open and re-execute.
Run the corresponding experiment first; missing artifacts raise an error naming
what must be generated.

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

The tests cover forward targets, lagged weights, chronological purging,
configuration loading, sklearn-compatible model selection, the relationship
between monotone linear and 1-additive Choquet regression, and the separation
between executable scripts and reporting notebooks.
