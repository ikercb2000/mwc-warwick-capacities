# Temporary FRED data downloader

This folder lives under `data/fred_data_download/` and is intentionally **outside the Python package**. It is a temporary utility for downloading and preparing the FRED/ALFRED variables needed for the MSc dissertation experiments.

Expected repository layout:

```text
capacities-ml-fin/
├── src/
│   └── capacities_ml_fin/
├── tests/
├── data/
│   └── fred_data_download/  <- this folder
├── pyproject.toml
└── ...
```

After the data have been downloaded and processed, this folder can be deleted if desired.

## Data downloaded

Daily / market variables:

- `VIXCLS`: VIX, forward-looking market volatility proxy.
- `DGS3MO`: 3-month Treasury constant-maturity yield, used as risk-free proxy.
- `DGS2`: 2-year Treasury yield.
- `DGS10`: 10-year Treasury yield.
- `BAA10Y`: broad Baa corporate credit spread.
- `DFF`: effective federal funds rate.

Monthly macro variables, downloaded as **initial releases**:

- `CPIAUCSL`: CPI.
- `UNRATE`: unemployment rate.
- `INDPRO`: industrial production.

The empirical sample is:

```text
2014-01-01 to 2026-07-30
```

The downloader starts in 2012 to provide enough pre-sample history for lagged and year-on-year macro transformations.

## 1. Obtain a FRED API key

Create/sign into a FRED account and create an API key from the API Keys section of your account.

Do not commit the key to Git.

Create `.env` in the repository root, next to `pyproject.toml`, and add:

```text
FRED_API_KEY=your_real_key
```

## 2. Install the temporary dependencies

From inside this folder:

```powershell
python -m pip install -r requirements.txt
```

If your repository environment already contains these packages, no separate installation is needed.

## 3. Download the raw data

From inside `data/fred_data_download/`:

```powershell
python download_fred.py
```

The scripts write outside this utility folder, into the repository-level data layer:

```text
data/
├── raw/
│   ├── fred/
│   └── alfred/
│       └── initial_releases/
└── experiments/
```

## 4. Build the daily feature panel

For a first test:

```powershell
python build_daily_features.py
```

This uses a Monday-Friday calendar.

For the final dissertation experiments, use the exact equity trading calendar from an experiment dataset, for example:

```powershell
python build_daily_features.py `
    --calendar ../experiments/portfolio_equal_weight.parquet `
    --calendar-date-column date
```

This creates:

```text
data/experiments/fred_daily_features.parquet
```

## Important

The macro variables are not assigned to the month to which they refer. They are aligned according to their initial publication availability date, so the forecasting dataset does not use a macro observation before it was publicly available.

Do not normalise the variables globally here. Scaling to `[0,1]`, standardisation, winsorisation, feature selection, etc. should be fitted only on each training sample/window during the experiments.

## Suggested `.gitignore`

At minimum:

```gitignore
/.env
data/raw/
```
