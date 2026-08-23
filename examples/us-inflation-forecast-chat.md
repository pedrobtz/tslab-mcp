# U.S. inflation: forecasting with macroeconomic regressors

This example forecasts monthly U.S. CPI inflation for the next 12 months. It
compares a dynamic-regression `AutoARIMA` model with univariate benchmarks, so
the result answers a useful empirical question: do the chosen macroeconomic
inputs improve forecasts over models that use inflation history alone?

The target is the 12-month log change in seasonally adjusted CPI. The four
inputs are unemployment, the federal funds rate, 12-month industrial-production
growth, and 12-month producer-price inflation. Every input is lagged 12 months.
That lag is deliberate: at any forecast origin, all inputs needed for the next
12 forecast months are already observed. The example therefore needs no
invented future policy-rate or unemployment scenario and leaks no future input
values into rolling cross-validation.

The worked dataset ends in December 2024 and forecasts January through December
2025. This fixed historical origin keeps the walkthrough stable and makes it
possible to compare the forecast later with subsequently observed inflation.

`AutoARIMA` consumes the extra columns as regressors. `SeasonalNaive` and
`AutoETS` ignore them and provide honest univariate comparisons.

## Prepare the two data files

From the repository root, run:

```bash
uv run python examples/download_us_inflation_data.py
```

The downloader uses FRED's public CSV endpoint and requires no API key. By
default it retains January 1960 through December 2024 and writes:

- `examples/us_inflation.csv`: `unique_id, ds, y` plus four historical
  regressor columns;
- `examples/us_inflation_future_regressors.csv`: the same ID, dates, and four
  regressors for the next 12 months, without a target.

FRED observations can be revised. This is a reproducible forecasting workflow,
but not a vintage-correct historical policy evaluation: the manifest pins the
downloaded file hashes, while the downloader always fetches the latest
available vintage.

Obtain the absolute paths required by the tools:

```bash
uv run python -c "from pathlib import Path; print(Path('examples/us_inflation.csv').resolve()); print(Path('examples/us_inflation_future_regressors.csv').resolve())"
```

Replace `<INFLATION_CSV>` and `<FUTURE_REGRESSORS_CSV>` below with those paths.
Keep all prompts in one chat because the handle lives in the MCP server process.

## 1. Load and validate

```text
Use the tslab MCP tools for every time-series calculation.

Load <INFLATION_CSV> as `us_inflation` with frequency `MS`. Report its date
range, observation count, integrity result, and regressor columns. Stop if the
monthly target is irregular or any target or regressor value is invalid.
```

The load result should identify one target series and these regressors:

```text
unemployment_rate_lag12
federal_funds_rate_lag12
industrial_production_yoy_lag12
producer_price_inflation_yoy_lag12
```

## 2. Describe the inflation target

```text
Describe `us_inflation`. Interpret the target's trend strength, seasonal
strength, coefficient of variation, and differenced lag-1 autocorrelation.
Explain that these features describe CPI inflation itself; they do not measure
the predictive value of the four regressors. Do not forecast yet.
```

## 3. Confirm the models

```text
List only the statistical models. Confirm that SeasonalNaive, AutoETS, and
AutoARIMA are available and that AutoARIMA is reported as regressor-aware. Do
not use foundation models in this example.
```

## 4. Compare rolling 12-month forecasts

```text
Cross-validate SeasonalNaive, AutoETS, and AutoARIMA on `us_inflation` with a
12-month horizon, 5 rolling windows, and MASE, sMAPE, and MAE. Use MASE as the
primary metric. Explain that AutoARIMA uses the four lag-12 regressors while the
other models use only inflation history. Show the ranking and the Parquet path.
Do not select or forecast yet.
```

Five non-overlapping windows evaluate five distinct one-year forecast periods.
Because every regressor is lagged 12 months, each window uses only information
that would have been available at its cutoff.

## 5. Validate the model choice

```text
Propose the model with the lowest finite MASE from the latest comparison and
validate it with `tsf_select_model`, using MASE and no minimum relative
improvement. Quote its score and the SeasonalNaive score.

If the tool requests another comparison, add one justified statistical model,
repeat cross-validation once, and validate again. If no challenger qualifies,
finalize SeasonalNaive explicitly. Do not forecast without a non-null
selected_model.
```

The regressors are useful only if the dynamic-regression model wins out of
sample. A baseline victory is a valid result rather than a reason to override
the evidence.

## 6. Forecast the next 12 months

Replace `<SELECTED_MODEL>` with the validated selection:

```text
Forecast `us_inflation` for 12 months with <SELECTED_MODEL>, with 80% and 95%
prediction intervals, and preview all 12 rows.

If <SELECTED_MODEL> is AutoARIMA, pass
<FUTURE_REGRESSORS_CSV> as future_regressors_path. Otherwise omit that path
because the selected model does not consume regressors.

Interpret y as approximate year-over-year CPI inflation in percent. Confirm
that the forecast dates are January through December 2025. Include the forecast
Parquet path and do not describe prediction intervals as guarantees.
```

## 7. Export the reproducible record

Replace the bracketed values with the actual result:

```text
Export the `us_inflation` manifest with this note:

"Forecasted U.S. year-over-year CPI inflation for 12 months. Compared
SeasonalNaive, AutoETS, and regressor-aware AutoARIMA over five rolling
12-month windows. Selected <SELECTED_MODEL> with MASE <MODEL_MASE> versus
<BASELINE_MASE> for SeasonalNaive. AutoARIMA's inputs were unemployment, the
federal funds rate, industrial-production growth, and producer-price inflation,
all lagged 12 months. Forecast intervals were 80% and 95%."

Then export a self-contained HTML report from the returned manifest path.
Return both paths and list the recorded steps.
```

The forecast run records the future-regressor file's absolute path and SHA-256,
so an AutoARIMA result remains tied to both its target history and future input
values.

## One-prompt version

After downloading the files, the full exercise can be initiated with:

```text
Use the tslab MCP tools to forecast U.S. CPI inflation from <INFLATION_CSV>.
Use handle `us_inflation` and frequency `MS`.

Validate the monthly target and its four regressor columns, then describe the
target. Confirm the statistical models and compare SeasonalNaive, AutoETS, and
AutoARIMA with h=12, 5 rolling windows, and MASE/sMAPE/MAE. AutoARIMA uses the
lag-12 regressors; the other candidates are univariate benchmarks. Propose the
lowest-MASE model and validate it with `tsf_select_model`, following the one
retry or explicit baseline-fallback instruction if necessary. Forecast the
validated model for 12 months with 80% and 95% intervals. If AutoARIMA is
selected, pass <FUTURE_REGRESSORS_CSV> as future_regressors_path; otherwise
omit it. Export a manifest with the numerical selection rationale and then an
HTML report. Return artifact paths instead of dumping large frames.
```
