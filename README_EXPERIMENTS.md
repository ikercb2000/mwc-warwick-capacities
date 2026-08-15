# Empirical experiment execution

The repository separates numerical execution from presentation:

```text
src/mwc_experiments/
|-- data/                 reusable data construction and processed-data loading
|-- workflows/            reusable model-fitting, evaluation and script helpers
`-- reporting.py          artifact loaders used by notebooks
scripts/                  complete, directly executable experiment pipelines
notebooks/                read-only reports over saved artifacts
data/experiments/         model-ready Parquet datasets
data/results/tables/      CSV and Parquet results
data/results/figures/     generated PNG figures
```

Reusable functions, classes, candidate registries and plotting primitives are
defined under `src/`. Each experiment pipeline itself is written directly at
the top level of its script. The scripts are executables rather than importable
modules, so importing one also executes its pipeline. Notebooks do not construct
data, fit models, calculate metrics or define helpers; they load the saved
artifacts and present them as DataFrames and images.

## Execution order

Prepare the data layer once:

```powershell
poetry run python scripts/build_experiment_data.py
```

Then run any experiment independently:

```powershell
poetry run python scripts/experiment_factors.py --quick
poetry run python scripts/experiment_predict_loss.py --quick
poetry run python scripts/experiment_tail_risk.py --quick
poetry run python scripts/experiment_distortion_risk.py
poetry run python scripts/experiment_autoregression.py
```

The three predictive scripts support:

- `--quick`: one representative hyperparameter combination per model family;
- `--full`: complete chronological validation grids;
- `--quiet`: suppress model-by-model progress messages.

Regression experiments include an unrestricted OLS benchmark and a monotone
linear benchmark fitted with the same orientation and scaling as the capacity
models. Choquet order and interaction regularization are selected using
validation RMSE only; the selected specification is then reported with its
held-out test metrics. For a normalized 1-additive capacity, singleton weights
may be zero but must be non-negative and sum to one, so an L1 penalty on those
weights would be constant rather than sparsity-inducing.

If neither `--quick` nor `--full` is supplied, `quick_mode_default` from
`configs/experiment_settings.toml` is used. Processed data are read directly
from `data/experiments/`; if required Parquet files are missing, an experiment
rebuilds and persists them from the raw data.

## Experiment scripts

- `scripts/build_experiment_data.py` contains the complete data-preparation
  pipeline: it builds the structured data layer, audits coverage and
  missingness, and saves the EDA figures.
- `scripts/experiment_factors.py` fits the equity factor models and saves
  metrics, predictions, residuals, covariance summaries, HAC comparisons,
  capacities and stability outputs.
- `scripts/experiment_predict_loss.py` runs the 1-, 5- and 10-day loss
  forecasts, high-loss metrics, HAC comparisons, capacity summaries and
  portfolio-weight robustness.
- `scripts/experiment_tail_risk.py` runs the 95% tail classifiers, the 97.5%
  robustness specification, probability diagnostics, thresholds and capacity
  stability.
- `scripts/experiment_distortion_risk.py` runs the regime-switching simulation,
  distortion measures, rolling capital, coverage tests, contributions,
  diversification and axiom checks.
- `scripts/experiment_autoregression.py` runs the linear/Choquet AR comparison,
  stationarity diagnostics, fixed-parameter forecasts and high-VIX evaluation.

## Notebook reports

The notebooks retain the original numerical order and subject matter:

1. `00_data_preparation_and_eda.ipynb`
2. `01_choquet_factor_models.ipynb`
3. `02_future_loss_regression.ipynb`
4. `03_tail_risk_classification.ipynb`
5. `04_distortion_risk_simulation.ipynb`
6. `05_choquet_autoregression_robustness.ipynb`

They are intentionally inexpensive to open or re-execute. A missing artifact
raises a direct error naming the script output that must be generated first.

## Outputs and reproducibility

Existing table filenames are preserved. Additional numerical summaries,
predictions, failures and diagnostics are stored alongside them. Figures are
saved as PNG under `data/results/figures/` before notebooks display them.
Fitted solver objects are not pickled; portable numerical results and model
interpretation tables are persisted instead.

Chronological train/validation/test splits, horizon purging, lagged portfolio
weights, point-in-time tail thresholds and train-only preprocessing remain in
the reusable data, modeling and workflow layers.
