# AirPassengers: a guided LLM chat example

This example uses the classic AirPassengers dataset: monthly international
airline passenger totals from January 1949 through December 1960. It is a good
first forecasting exercise because one small series contains a clear upward
trend, annual seasonality, and seasonal variation that grows with the level.

The included [`air_passengers.csv`](air_passengers.csv) is already in the
required long format:

```text
unique_id,ds,y
air_passengers,1949-01-01,112
air_passengers,1949-02-01,118
...
```

The goal is to forecast the next 12 months while making the model choice from
cross-validation evidence rather than intuition alone.

## Before opening the chat

Configure `tslab-mcp` in your chat client as described in the project
[`README.md`](../README.md#configure), and confirm that its nine `tsf_*` tools
are enabled.

The load tool needs an absolute path. From the repository root, obtain it with:

```bash
python -c "from pathlib import Path; print(Path('examples/air_passengers.csv').resolve())"
```

In the prompts below, replace `<AIR_PASSENGERS_CSV>` with that output. Keep all
steps in one chat so the in-memory handle and the agent's reasoning remain in
the same session.

## Step 1: load and validate the data

Paste:

```text
Use the tslab MCP tools for every time-series calculation.

Load <AIR_PASSENGERS_CSV> with the handle `air_passengers`. Report the inferred
frequency, date range, observation count, and data-integrity result. Do not
continue if the series contains missing, duplicate, or off-grid timestamps.
```

The response should show one series, 144 observations, monthly-start frequency
`MS`, dates from 1949-01-01 through 1960-12-01, and a regular integrity result.

## Step 2: describe the series

Paste:

```text
Describe the loaded `air_passengers` series. Explain in plain language what its
trend strength, seasonal strength, coefficient of variation, and differenced
lag-1 autocorrelation imply for model selection. Keep the numerical feature
values in the answer and do not forecast yet.
```

The useful question here is not merely whether the series is seasonal. Ask
whether a model should represent both the upward trend and the strong annual
cycle, and keep a simple seasonal baseline in the comparison.

## Step 3: discover available models

Paste:

```text
List the statistical models available in this tslab installation. Do not load
or propose foundation models for this example. Confirm that SeasonalNaive,
AutoETS, AutoARIMA, Theta, and DynamicOptimizedTheta are available.
```

This keeps the example fast and reproducible without model-weight downloads.

## Step 4: compare candidates by rolling cross-validation

Paste:

```text
Cross-validate SeasonalNaive, AutoETS, AutoARIMA, Theta, and
DynamicOptimizedTheta on `air_passengers`.

Use a 12-month horizon, 4 rolling windows, and report MASE, sMAPE, and MAE.
Treat MASE as the primary selection metric. Show the ranking, compare every
candidate with SeasonalNaive, and give the Parquet artifact path. Do not make
the final forecast yet.
```

Why these settings:

- A 12-step horizon matches a full year of monthly planning.
- Four windows test the models at multiple historical origins.
- SeasonalNaive is the minimum credible baseline for this series.
- MASE is scale-free; MAE keeps the error interpretable in thousands of
  passengers; sMAPE provides a percentage-style comparison.

Exact scores can vary when forecasting-library versions change. The exported
manifest records those versions, so the run remains auditable.

## Step 5: make and explain the model decision

Paste:

```text
Using only the cross-validation evidence already returned, propose one model
for the 12-month forecast. Prefer the lowest MASE, use sMAPE as a secondary
check, and prefer the simpler model among exact-score ties.

Call `tsf_select_model` with that proposal, MASE as the primary metric, and no
minimum relative-improvement threshold. State the validation status, quote the
proposal and SeasonalNaive scores, and explain the result. Do not forecast
unless the tool returns a non-null selected_model.

If it requests an additional comparison, add one justified available model,
repeat cross-validation, and validate again. If no justified candidate remains
or none qualifies, finalize SeasonalNaive explicitly through the selection tool.
```

The deterministic checkpoint makes the rationale explicit and prevents the
forecast call from silently diverging from the CV evidence.

## Step 6: forecast the next year

Replace `<SELECTED_MODEL>` with the model chosen in Step 5, then paste:

```text
Forecast `air_passengers` for 12 months with <SELECTED_MODEL>. Request 80% and
95% prediction intervals and preview all 12 forecast rows.

Summarize the expected seasonal pattern and uncertainty without claiming that
the prediction intervals guarantee future outcomes. Include the forecast
Parquet artifact path.
```

The resulting dates should run from 1961-01-01 through 1961-12-01.

## Step 7: check historical anomalies

Use the same model selected for forecasting:

```text
Detect historical anomalies in `air_passengers` with <SELECTED_MODEL>. Use
h=1, 12 rolling windows, and a 99% confidence level. Report the number and dates
of flagged observations, explain that model errors can appear as anomalies, and
include the full-result Parquet path.
```

Using the forecast winner keeps the definition of expected behavior consistent.
Twelve windows bounds runtime while still checking a complete recent year.

## Step 8: export the reproducible record and report

Paste this after replacing the bracketed values with the actual model and
cross-validation scores:

```text
Export the `air_passengers` run manifest with this note:

"Selected <SELECTED_MODEL> for a 12-month forecast after four rolling-origin
windows. Its MASE was <MODEL_MASE> versus <BASELINE_MASE> for SeasonalNaive.
The choice also considered sMAPE and model simplicity. Forecast intervals are
80% and 95%; anomaly detection used 12 windows at 99%."

Then export a self-contained HTML report from the manifest path returned by the
first export. Return both file paths and list the recorded analysis steps.
```

The note is intentionally concrete: it preserves the decision, evidence,
baseline, horizon, and uncertainty settings alongside the data hash and pinned
environment.

## Step 9: ask for the final human-readable summary

Paste:

```text
Give me a concise final summary of this AirPassengers exercise. Include:

1. the data range and integrity result;
2. the main trend and seasonality evidence;
3. the models compared and the winning cross-validation scores;
4. the selected model and why it beat or did not materially beat the baseline;
5. the 1961 forecast's broad seasonal shape;
6. the anomaly result;
7. the forecast, anomaly, manifest, and HTML report paths.

Do not paste the complete Parquet data or invent values that were not returned
by a tool.
```

## One-prompt alternative

Once the staged workflow is familiar, the same exercise can be initiated with
one prompt:

```text
Use the tslab MCP tools to analyze <AIR_PASSENGERS_CSV> and forecast the next 12
months. Use handle `air_passengers`.

First validate the data and stop on any integrity issue. Describe the series,
then confirm which statistical models are installed. Cross-validate
SeasonalNaive, AutoETS, AutoARIMA, Theta, and DynamicOptimizedTheta using h=12,
4 windows, and MASE/sMAPE/MAE. Propose the winner using MASE first and validate
it with `tsf_select_model`; perform its one retry or baseline fallback when
needed. Forecast the validated model with 80% and 95% intervals. Export
Detect anomalies with the selected model using h=1, 12 windows, and a 99% level.
Export a manifest whose note contains the model-selection and anomaly settings,
then export an HTML report from that manifest. Explain each decision and return
artifact paths instead of large data frames.
```

The staged version is better for learning and review because you can inspect
and challenge each decision before the next tool call.
