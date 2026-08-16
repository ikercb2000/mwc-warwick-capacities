# Raw-data contract and provenance

The raw files are immutable experiment inputs. Bloomberg exports are licensed
and intentionally excluded from Git. A new checkout must acquire them using the
contract below before running `scripts/build_experiment_data.py`.

The authoritative asset universe, sector mapping, sample dates and FRED series
are in `configs/experiment_settings.toml`. If that configuration changes, the
required files change with it.

## Required directory layout

```text
data/raw/
|-- bloomberg/
|   `-- market_data/
|       |-- equities/daily/{TICKER}_daily_market_data.xlsx
|       `-- etfs/daily/<any-subdirectory>/{TICKER}_daily_etf_data.xlsx
`-- fred/{SERIES_ID}_<descriptive-name>.csv
```

The current equity universe is `AAPL`, `AMZN`, `JPM`, `KO`, `MSFT`, `NVDA`,
`UNH`, `WMT` and `XOM`. The current ETF universe is:

- market: `SPY`;
- style: `IWM`, `IVE`, `IVW`, `MTUM`, `QUAL`, `USMV`;
- sector: `XLE`, `XLF`, `XLK`, `XLP`, `XLV`, `XLY`.

The same request is available in machine-readable form in
`data/raw/bloomberg_requirements.csv`. A test keeps its ticker universe, end
date, frequency, currency, fields and filename patterns synchronized with the
repository configuration and loader contract.

Equity filenames and directories are exact. ETF files may be placed in nested
market/style/sector directories because the loader searches recursively, but
their filenames are also exact. There must be exactly one matching file per
ticker to avoid an ambiguous data snapshot.

## Bloomberg request

Request daily historical data (`Period = D`) for securities expressed as
`{TICKER} US Equity`. The export must begin no later than `2013-12-31`: the
configured model sample begins on `2014-01-02`, but one earlier observation is
needed to calculate the first return. It must extend through the configured
`sample_end` date. The current snapshot ends on `2026-07-30`.

### Equity fields

| Bloomberg field | Internal name | Required | Use |
| --- | --- | --- | --- |
| `TOT_RETURN_INDEX_GROSS_DVDS` | `total_return_index` | yes | Log total returns and all equity targets. |
| `PX_LAST` | `px_last` | yes | Dollar-volume input for Amihud illiquidity. |
| `VOLUME` | `volume` | yes | Dollar-volume input for Amihud illiquidity. |
| `CUR_MKT_CAP` | `market_cap` | yes | One-day-lagged market-cap portfolio weights. |
| `PX_OPEN` | `px_open` | no | Accepted and preserved, but not used currently. |
| `PX_HIGH` | `px_high` | no | Accepted and preserved, but not used currently. |
| `PX_LOW` | `px_low` | no | Accepted and preserved, but not used currently. |

### ETF fields

| Bloomberg field | Internal name | Required | Use |
| --- | --- | --- | --- |
| `TOT_RETURN_INDEX_GROSS_DVDS` | `total_return_index` | yes | Market, style and sector factor returns. |
| `PX_LAST` | `px_last` | no | Accepted, but not used by the current experiments. |

Use the gross-dividend total-return index rather than `PX_LAST` to reproduce the
return series. Prices and market capitalisations must use a common currency
(the reference exports use USD). `CUR_MKT_CAP` must use the same scale for every
equity; Bloomberg commonly exports it in millions of the selected currency.
`VOLUME` must use a consistent unit across equities. The loader does not infer
or convert currencies or units.

Quarterly fundamental workbooks are not inputs to the current experiments and
do not need to be acquired. The loader deliberately ignores
`data/raw/bloomberg/fundamentals/`; `fundamentals_included = false` is recorded
in the shared configuration.

## Workbook format read by the code

The parser uses the first worksheet; the worksheet name is irrelevant. It reads
the first 20 rows without a header and selects the first row whose first cell,
after trimming and lower-casing, is exactly `date`. This allows Bloomberg
metadata rows before the observations. A recommended workbook looks like:

```text
Security     AAPL US Equity
Start Date   2013-12-31
End Date     2026-07-30
Period       D
Currency     USD

Date         TOT_RETURN_INDEX_GROSS_DVDS  PX_LAST  VOLUME  CUR_MKT_CAP
2013-12-31   20.0364                       20.0364  223277488  504770.9993
...
```

Parsing and cleaning rules are part of the reproducibility contract:

1. `Date` is case-sensitive after the header row has been selected.
2. Dates may be Excel datetimes, parseable date strings, or Excel serial day
   numbers using origin `1899-12-30`.
3. Rows without a valid date and completely empty rows are removed.
4. Dates are sorted; if a date is duplicated, the last row is retained.
5. Bloomberg field names are renamed using
   `mwc_experiments.data.loaders.mappings.COLUMN_RENAMES`.
6. Every non-date column is converted with `pandas.to_numeric(errors="coerce")`;
   text such as `#N/A N/A` therefore becomes missing data.
7. The common market calendar contains dates for which all nine equity total-
   return indices are present, restricted to `[sample_start, sample_end]`.

Additional numeric Bloomberg columns are accepted but do not influence the
current feature pipeline. Missing required field names cause the build to fail;
missing observations may propagate as `NaN` and reduce the usable model sample.

## FRED format

The required identifiers are `VIXCLS`, `DGS3MO`, `DGS2`, `DGS10`, `BAA10Y` and
`DFF`. Each CSV filename must start with `{SERIES_ID}_` and must contain the
columns `date` and `value` (additional columns are harmless).
Dates are parsed by pandas, values are coerced to numeric, duplicate dates keep
the last observation, and values are forward-filled onto the equity calendar.
Do not pre-convert percentage series to decimals: yields and spreads are read in
percentage points and transformed by the feature code where required.

## Validation and build

From the repository root, validate the complete raw snapshot with:

```powershell
poetry run python -c "from mwc_experiments.data import load_raw_market_data; print(load_raw_market_data().audit().to_string(index=False))"
poetry run python scripts/build_experiment_data.py
```

The build writes the regenerated cache under `data/experiments/`, which is also
excluded from Git. It hashes every raw input, the data-building code and the
data-preparation configuration in `data/experiments/manifest.json`. Every
experiment run records that input fingerprint in its own manifest.

Do not edit vendor files in place. Replace the complete snapshot, rebuild the
processed data and preserve the resulting manifests and run metadata. Confirm
Bloomberg redistribution rights independently before sharing any raw or derived
data outside the licensed environment.
