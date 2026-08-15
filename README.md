# MSc Warwick Capacities

Reproducible experiments for the MSc Mathematical Finance dissertation at the
University of Warwick. The project compares additive, nonlinear and
Choquet-capacity methods for factor modelling, future-loss forecasting, tail
classification, distortion risk and autoregression.

## Setup

```powershell
poetry install
poetry run pytest -q
```

## Data pipeline

Raw inputs live under `data/raw/`. Build the structured Parquet layer and its
EDA artifacts with:

```powershell
poetry run python scripts/build_experiment_data.py
```

The model-ready datasets are written to `data/experiments/`; tables and figures
are written to `data/results/tables/` and `data/results/figures/`.

## Run experiments

Each experiment is now a standalone script. Predictive scripts use the default
mode from `configs/experiment_settings.toml`; pass `--quick` explicitly while
developing or `--full` for the complete validation grids.

```powershell
poetry run python scripts/experiment_factors.py --quick
poetry run python scripts/experiment_predict_loss.py --quick
poetry run python scripts/experiment_tail_risk.py --quick
poetry run python scripts/experiment_distortion_risk.py
poetry run python scripts/experiment_autoregression.py
```

Each file under `scripts/` shows the complete pipeline that it executes, so it
can be read and run directly without jumping through a pipeline-sized
`main()`. Reusable model-fitting, evaluation and artifact helpers live under
`src/mwc_experiments/workflows/`, while reusable data-construction primitives
live under `src/mwc_experiments/data/`. Files under `scripts/` are executables,
not importable modules: importing one executes its pipeline.

## View reports

The notebooks no longer fit models or calculate results. Open them after the
corresponding script has run; they only load persisted CSV/Parquet tables and
PNG figures into a user-friendly report.

```powershell
poetry run jupyter lab
```

See [README_EXPERIMENTS.md](README_EXPERIMENTS.md) for the experiment catalogue,
execution modes and output layout.
