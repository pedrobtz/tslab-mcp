"""Process-local store of loaded panels, keyed by short opaque handles.

Design invariant I1: raw frames never cross the tool boundary. A panel is read
once, validated, and parked here; every other tool takes the handle. The store
is the only mutable global in the package, kept in one place so moving it into
a FastMCP ``lifespan`` context later is a mechanical change.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import sha256_file

#: The Nixtla long-format contract every tool assumes.
REQUIRED_COLUMNS = ("unique_id", "ds", "y")
FUTURE_REQUIRED_COLUMNS = ("unique_id", "ds")

_PARQUET_SUFFIXES = frozenset({".parquet", ".pq"})
_CSV_SUFFIXES = frozenset({".csv", ".txt", ".tsv"})

_EXAMPLE_ALIASES = (
    "'D' (daily), 'B' (business day), 'W-MON', 'MS' (month start), "
    "'QS', 'YS', 'h'"
)


@dataclass
class SeriesHandle:
    """A loaded panel plus everything needed to reproduce the load."""

    handle: str
    source: str
    df: pd.DataFrame
    freq: str
    sha256: str
    loaded_at: str
    integrity: dict[str, Any] = field(default_factory=dict)
    runs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_series(self) -> int:
        return int(self.df["unique_id"].nunique())

    @property
    def n_obs(self) -> int:
        return len(self.df)


_STORE: dict[str, SeriesHandle] = {}


def get(handle: str) -> SeriesHandle:
    """Look up a handle, or raise an error naming what *is* loaded."""
    try:
        return _STORE[handle]
    except KeyError:
        known = ", ".join(sorted(_STORE)) or "none"
        raise ValueError(
            f"Unknown handle '{handle}'. Loaded handles: {known}. "
            "Call tsf_load_series first."
        ) from None


def put(series: SeriesHandle) -> None:
    """Register (or replace) a handle."""
    _STORE[series.handle] = series


def handles() -> list[str]:
    """Every currently registered handle, sorted."""
    return sorted(_STORE)


def clear() -> None:
    """Drop every handle. Used by tests; there is no tool that calls this."""
    _STORE.clear()


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in _PARQUET_SUFFIXES:
        return pd.read_parquet(path)
    if suffix in _CSV_SUFFIXES:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    raise ValueError(
        f"Unsupported file extension '{path.suffix}' for {path}. "
        "Use .csv, .tsv or .parquet."
    )


def _validate_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing}. Found: {list(df.columns)}. "
            "Rename to the Nixtla convention (unique_id, ds, y) before loading "
            f"{path.name}."
        )


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing_ids = df["unique_id"].isna()
    if missing_ids.any():
        raise ValueError(
            f"Column 'unique_id' contains {int(missing_ids.sum())} missing "
            "value(s). Every row must belong to a named series."
        )
    df["unique_id"] = df["unique_id"].astype(str)
    blank_ids = df["unique_id"].str.strip().eq("")
    if blank_ids.any():
        raise ValueError(
            f"Column 'unique_id' contains {int(blank_ids.sum())} blank value(s). "
            "Every row must belong to a named series."
        )
    try:
        df["ds"] = pd.to_datetime(df["ds"])
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Column 'ds' could not be parsed as timestamps ({exc}). "
            "Use an ISO-8601 date column, e.g. '2024-01-31'."
        ) from None
    missing_ds = df["ds"].isna()
    if missing_ds.any():
        raise ValueError(
            f"Column 'ds' contains {int(missing_ds.sum())} missing or invalid "
            "timestamp(s). Use an ISO-8601 date for every row."
        )

    numeric_y = pd.to_numeric(df["y"], errors="coerce")
    invalid_y = numeric_y.isna() | numeric_y.isin([float("inf"), float("-inf")])
    if invalid_y.any():
        examples = df.loc[invalid_y, "y"].head(5).tolist()
        raise ValueError(
            f"Column 'y' contains {int(invalid_y.sum())} missing, non-numeric, "
            f"or non-finite value(s), for example {examples}. Clean or impute "
            "the target explicitly before loading; this server never invents "
            "target values."
        )
    df["y"] = numeric_y
    _coerce_numeric_columns(
        df,
        [column for column in df.columns if column not in REQUIRED_COLUMNS],
        label="Regressor",
    )
    ordered: pd.DataFrame = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return ordered


def _coerce_numeric_columns(
    df: pd.DataFrame, columns: list[str], *, label: str
) -> None:
    """Coerce model inputs in place and reject missing/non-finite values."""
    for column in columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid = numeric.isna() | numeric.isin([float("inf"), float("-inf")])
        if invalid.any():
            examples = df.loc[invalid, column].head(5).tolist()
            raise ValueError(
                f"{label} column '{column}' contains {int(invalid.sum())} "
                "missing, non-numeric, or non-finite value(s), for example "
                f"{examples}. Clean or impute regressors explicitly before "
                "loading; this server never invents model inputs."
            )
        df[column] = numeric


def _check_duplicates(df: pd.DataFrame) -> None:
    duplicated = df.duplicated(subset=["unique_id", "ds"])
    if not duplicated.any():
        return
    offenders = df.loc[duplicated, "unique_id"].unique()[:5]
    raise ValueError(
        f"Found {int(duplicated.sum())} duplicate (unique_id, ds) rows, "
        f"affecting series {list(offenders)}. Aggregate or de-duplicate the "
        "source before loading -- forecasting a panel with repeated timestamps "
        "silently produces wrong numbers."
    )


def infer_frequency(df: pd.DataFrame) -> str:
    """Infer a pandas offset alias from the longest series in the panel.

    Raises with the offending series named when nothing can be inferred, since
    the caller's next action is to pass ``freq`` explicitly.
    """
    lengths = df.groupby("unique_id").size().sort_values(ascending=False)
    for uid in lengths.index:
        stamps = df.loc[df["unique_id"] == uid, "ds"]
        if len(stamps) < 3:
            continue
        inferred = pd.infer_freq(pd.DatetimeIndex(stamps))
        if inferred:
            return str(inferred)
    longest = str(lengths.index[0])
    sample = df.loc[df["unique_id"] == longest, "ds"].head(4)
    gaps = sample.diff().dropna().astype(str).tolist()
    raise ValueError(
        f"Could not infer a frequency. The longest series '{longest}' has "
        f"{int(lengths.iloc[0])} observations with spacing {gaps}. "
        f"Pass freq explicitly, e.g. {_EXAMPLE_ALIASES}."
    )


def validate_frequency(freq: str) -> str:
    """Validate an explicit pandas offset alias without changing its spelling."""
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid frequency '{freq}' ({exc}). Pass a pandas offset alias, "
            f"e.g. {_EXAMPLE_ALIASES}."
        ) from None
    if offset is None:
        raise ValueError(
            f"Invalid frequency '{freq}'. Pass a pandas offset alias, "
            f"e.g. {_EXAMPLE_ALIASES}."
        )
    return freq


def inspect_gaps(df: pd.DataFrame, freq: str, max_series: int = 20) -> dict[str, Any]:
    """Count missing and off-grid timestamps without modifying the panel.

    Leading and trailing periods are not gaps because the panel does not assert
    that every series must share a common start or end. A timestamp is off-grid
    when it cannot occur on the supplied anchored frequency (for example the
    15th of a month for ``MS``).
    """
    offset = pd.tseries.frequencies.to_offset(freq)
    assert offset is not None  # validated by :func:`validate_frequency`

    affected: list[dict[str, Any]] = []
    total_missing = 0
    total_off_grid = 0
    for uid, group in df.groupby("unique_id", sort=True):
        stamps = pd.DatetimeIndex(group["ds"])
        expected = pd.date_range(stamps.min(), stamps.max(), freq=offset)
        missing = expected.difference(stamps)
        off_grid = stamps.difference(expected)
        n_missing = len(missing)
        n_off_grid = len(off_grid)
        if not (n_missing or n_off_grid):
            continue
        total_missing += n_missing
        total_off_grid += n_off_grid
        affected.append(
            {
                "unique_id": str(uid),
                "n_missing_timestamps": n_missing,
                "n_off_grid_timestamps": n_off_grid,
                "first_missing": missing[0].isoformat() if n_missing else None,
                "first_off_grid": off_grid[0].isoformat() if n_off_grid else None,
            }
        )

    shown = affected[:max_series]
    return {
        "is_regular": not affected,
        "n_missing_timestamps": total_missing,
        "n_off_grid_timestamps": total_off_grid,
        "n_series_affected": len(affected),
        "series": shown,
        "truncated": len(shown) < len(affected),
        "n_omitted": len(affected) - len(shown),
    }


def require_regular(series: SeriesHandle) -> None:
    """Reject analysis of an irregular panel with a concrete remediation."""
    integrity = series.integrity
    if integrity.get("is_regular", True):
        return
    examples = [row["unique_id"] for row in integrity.get("series", [])[:5]]
    raise ValueError(
        f"Panel '{series.handle}' is irregular at frequency '{series.freq}': "
        f"{integrity.get('n_missing_timestamps', 0)} missing and "
        f"{integrity.get('n_off_grid_timestamps', 0)} off-grid timestamp(s) "
        f"across {integrity.get('n_series_affected', 0)} series, including "
        f"{examples}. Regularize the source explicitly, then call "
        "tsf_load_series again. Automatic filling or imputation would add an "
        "unaudited analytical assumption."
    )


def load_panel(
    path_str: str, freq: str | None = None, handle: str | None = None
) -> tuple[SeriesHandle, bool]:
    """Read, validate and register a panel. Returns the handle and whether it
    replaced an existing entry of the same name."""
    path = Path(path_str).expanduser()
    if not path.exists():
        raise ValueError(
            f"No such file: {path}. Pass an absolute path to a local .csv or "
            ".parquet file."
        )
    if path.is_dir():
        raise ValueError(f"{path} is a directory. Pass a single .csv or .parquet file.")

    df = _read_frame(path)
    _validate_columns(df, path)
    if df.empty:
        raise ValueError(f"{path.name} has the right columns but no rows.")
    df = _coerce(df)
    _check_duplicates(df)

    resolved_freq = validate_frequency(freq) if freq else infer_frequency(df)
    integrity = inspect_gaps(df, resolved_freq)
    name = handle or f"s{uuid.uuid4().hex[:6]}"
    replaced = name in _STORE

    series = SeriesHandle(
        handle=name,
        source=str(path.resolve()),
        df=df,
        freq=resolved_freq,
        sha256=sha256_file(path),
        loaded_at=datetime.now(timezone.utc).isoformat(),
        integrity=integrity,
    )
    put(series)
    return series, replaced


def load_future_regressors(
    path_str: str, series: SeriesHandle, h: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate the known future values needed by dynamic regression.

    The future frame must match the historical regressor schema and contain the
    exact next ``h`` timestamps for every series. Requiring an explicit file
    keeps assumptions about future inputs outside the forecasting server.
    """
    path = Path(path_str).expanduser()
    if not path.exists():
        raise ValueError(
            f"No such future regressors file: {path}. Pass an absolute path to "
            "a local .csv or .parquet file."
        )
    if path.is_dir():
        raise ValueError(
            f"{path} is a directory. Pass one future regressors CSV or Parquet file."
        )

    regressors = [c for c in series.df.columns if c not in REQUIRED_COLUMNS]
    if not regressors:
        raise ValueError(
            f"Panel '{series.handle}' has no historical regressor columns, so "
            "future_regressors_path must be omitted."
        )

    frame = _read_frame(path)
    expected_columns = [*FUTURE_REQUIRED_COLUMNS, *regressors]
    missing = [column for column in expected_columns if column not in frame.columns]
    unexpected = [column for column in frame.columns if column not in expected_columns]
    if missing or unexpected:
        raise ValueError(
            f"Future regressors columns do not match panel '{series.handle}'. "
            f"Expected exactly {expected_columns}; missing {missing}; unexpected "
            f"{unexpected}."
        )
    if frame.empty:
        raise ValueError(f"{path.name} has the right columns but no rows.")

    frame = frame.copy()
    missing_ids = frame["unique_id"].isna()
    if missing_ids.any():
        raise ValueError(
            "Future regressors column 'unique_id' contains "
            f"{int(missing_ids.sum())} missing value(s)."
        )
    frame["unique_id"] = frame["unique_id"].astype(str)
    blank_ids = frame["unique_id"].str.strip().eq("")
    if blank_ids.any():
        raise ValueError(
            "Future regressors column 'unique_id' contains "
            f"{int(blank_ids.sum())} blank value(s)."
        )
    try:
        frame["ds"] = pd.to_datetime(frame["ds"])
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Future regressors column 'ds' could not be parsed ({exc}). Use "
            "ISO-8601 dates."
        ) from None
    if frame["ds"].isna().any():
        raise ValueError(
            "Future regressors column 'ds' contains missing or invalid timestamps."
        )
    _coerce_numeric_columns(frame, regressors, label="Future regressor")
    _check_duplicates(frame)

    historical_ids = set(series.df["unique_id"].unique())
    future_ids = set(frame["unique_id"].unique())
    if historical_ids != future_ids:
        missing_ids_set = sorted(historical_ids - future_ids)
        unexpected_ids = sorted(future_ids - historical_ids)
        raise ValueError(
            "Future regressors series IDs do not match the loaded panel. "
            f"Missing IDs: {missing_ids_set}; unexpected IDs: {unexpected_ids}."
        )

    offset = pd.tseries.frequencies.to_offset(series.freq)
    assert offset is not None  # the series frequency was validated on load
    date_errors: list[str] = []
    for uid in sorted(historical_ids):
        history = series.df.loc[series.df["unique_id"] == uid, "ds"]
        observed = pd.DatetimeIndex(
            frame.loc[frame["unique_id"] == uid, "ds"].sort_values()
        )
        expected = pd.date_range(history.max() + offset, periods=h, freq=offset)
        if observed.equals(expected):
            continue
        date_errors.append(
            f"{uid}: expected {h} dates from {expected[0].date()} through "
            f"{expected[-1].date()}, found {len(observed)} from "
            f"{observed.min().date() if len(observed) else 'none'} through "
            f"{observed.max().date() if len(observed) else 'none'}"
        )
    if date_errors:
        raise ValueError(
            "Future regressors do not cover the exact forecast horizon. "
            + "; ".join(date_errors[:5])
        )

    ordered = frame.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    metadata = {
        "source": str(path.resolve()),
        "sha256": sha256_file(path),
        "columns": regressors,
        "n_rows": len(ordered),
        "start": ordered["ds"].min().isoformat(),
        "end": ordered["ds"].max().isoformat(),
    }
    return ordered, metadata


def summarize(series: SeriesHandle, replaced: bool = False) -> dict[str, Any]:
    """Compact JSON-safe description of a loaded panel."""
    df = series.df
    counts = df.groupby("unique_id").size()
    return {
        "handle": series.handle,
        "source": series.source,
        "sha256": series.sha256,
        "freq": series.freq,
        "n_series": series.n_series,
        "n_obs": series.n_obs,
        "start": df["ds"].min().isoformat(),
        "end": df["ds"].max().isoformat(),
        "obs_per_series": {
            "min": int(counts.min()),
            "median": int(counts.median()),
            "max": int(counts.max()),
        },
        "n_missing_y": int(df["y"].isna().sum()),
        "integrity": series.integrity,
        "regressor_columns": [c for c in df.columns if c not in REQUIRED_COLUMNS],
        "extra_columns": [c for c in df.columns if c not in REQUIRED_COLUMNS],
        "loaded_at": series.loaded_at,
        "replaced_existing_handle": replaced,
    }
