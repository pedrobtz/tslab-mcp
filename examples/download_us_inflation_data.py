"""Download and prepare the FRED data used by us-inflation-forecast-chat.md.

The script writes one historical modeling file and one future-regressor file.
It needs no FRED API key and uses only tslab-mcp's core Python environment.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
SERIES_IDS = ("CPIAUCSL", "UNRATE", "FEDFUNDS", "INDPRO", "PPIACO")
REGRESSOR_COLUMNS = (
    "unemployment_rate_lag12",
    "federal_funds_rate_lag12",
    "industrial_production_yoy_lag12",
    "producer_price_inflation_yoy_lag12",
)
UNIQUE_ID = "us_cpi_inflation"


def download_fred_series(series_id: str) -> pd.Series:
    """Download one FRED series from its public graph CSV endpoint."""
    request = Request(
        FRED_CSV_URL.format(series_id=series_id),
        headers={"User-Agent": "tslab-mcp inflation example"},
    )
    with urlopen(request, timeout=30) as response:
        payload = response.read()
    frame = pd.read_csv(BytesIO(payload), na_values=".")
    expected = {"observation_date", series_id}
    if not expected.issubset(frame.columns):
        raise RuntimeError(
            f"FRED returned columns {list(frame.columns)} for {series_id}; "
            f"expected {sorted(expected)}."
        )
    dates = pd.to_datetime(frame["observation_date"], errors="raise")
    values = pd.to_numeric(frame[series_id], errors="coerce")
    return pd.Series(values.to_numpy(), index=dates, name=series_id)


def build_frames(
    start: str, end: str, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create leakage-safe history and known future lagged regressors."""
    raw = pd.concat(
        [download_fred_series(series_id) for series_id in SERIES_IDS], axis=1
    ).sort_index()
    required_start = pd.Timestamp(start) - pd.DateOffset(months=24)
    forecast_origin = pd.Timestamp(end)
    monthly_index = pd.date_range(required_start, forecast_origin, freq="MS")
    raw = raw.reindex(monthly_index)
    incomplete = raw[list(SERIES_IDS)].isna().any(axis=1)
    if incomplete.any():
        missing_dates = incomplete[incomplete].index
        examples = [date.date() for date in missing_dates[:5]]
        raise RuntimeError(
            "At least one FRED series is unavailable in the requested modeling "
            f"period; missing dates include {examples}. "
            "Choose different --start/--end dates."
        )

    current_inputs = pd.DataFrame(index=raw.index)
    current_inputs["unemployment_rate_lag12"] = raw["UNRATE"]
    current_inputs["federal_funds_rate_lag12"] = raw["FEDFUNDS"]
    current_inputs["industrial_production_yoy_lag12"] = 100 * np.log(
        raw["INDPRO"]
    ).diff(12)
    current_inputs["producer_price_inflation_yoy_lag12"] = 100 * np.log(
        raw["PPIACO"]
    ).diff(12)

    modeling = pd.DataFrame(index=raw.index)
    modeling["y"] = 100 * np.log(raw["CPIAUCSL"]).diff(12)
    modeling[list(REGRESSOR_COLUMNS)] = current_inputs.shift(12)
    modeling = modeling.loc[pd.Timestamp(start) : forecast_origin].dropna()
    if modeling.empty:
        raise RuntimeError(
            f"No complete modeling rows remain at or after {start}. Choose an "
            "earlier --start date."
        )
    expected_history = pd.date_range(
        modeling.index.min(), modeling.index.max(), freq="MS"
    )
    if not modeling.index.equals(expected_history):
        missing = expected_history.difference(modeling.index)
        raise RuntimeError(
            "The transformed history is not regular monthly data; missing "
            f"dates include {[date.date() for date in missing[:5]]}."
        )

    future_dates = pd.date_range(
        forecast_origin + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
    )
    source_dates = future_dates - pd.DateOffset(months=12)
    future_values = current_inputs.reindex(source_dates)
    if future_values.isna().any().any():
        missing_columns = future_values.columns[future_values.isna().any()].tolist()
        raise RuntimeError(
            "Known lag-12 future inputs are incomplete for columns "
            f"{missing_columns}. Reduce --horizon or retry after FRED updates."
        )

    history = modeling.reset_index(names="ds")
    history.insert(0, "unique_id", UNIQUE_ID)
    future = future_values.reset_index(drop=True)
    future.insert(0, "ds", future_dates)
    future.insert(0, "unique_id", UNIQUE_ID)
    return history.round(6), future.round(6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for the two generated CSV files (default: examples/).",
    )
    parser.add_argument(
        "--start",
        default="1960-01-01",
        help=(
            "First target month to retain after transformations "
            "(default: 1960-01-01)."
        ),
    )
    parser.add_argument(
        "--end",
        default="2024-12-01",
        help=(
            "Last target month and forecast origin "
            "(default: 2024-12-01 for a stable worked example)."
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        choices=range(1, 13),
        metavar="1..12",
        help="Future months to prepare; lag-12 inputs support at most 12.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history, future = build_frames(args.start, args.end, args.horizon)
    history_path = (args.output_dir / "us_inflation.csv").resolve()
    future_path = (args.output_dir / "us_inflation_future_regressors.csv").resolve()
    history.to_csv(history_path, index=False, date_format="%Y-%m-%d")
    future.to_csv(future_path, index=False, date_format="%Y-%m-%d")

    print(f"Historical data: {history_path}")
    print(f"Future regressors: {future_path}")
    print(
        f"Target coverage: {history['ds'].min().date()} through "
        f"{history['ds'].max().date()} ({len(history)} monthly rows)"
    )
    print(
        f"Forecast inputs: {future['ds'].min().date()} through "
        f"{future['ds'].max().date()} ({len(future)} monthly rows)"
    )


if __name__ == "__main__":
    main()
