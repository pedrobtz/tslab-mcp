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
    df["unique_id"] = df["unique_id"].astype(str)
    try:
        df["ds"] = pd.to_datetime(df["ds"])
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Column 'ds' could not be parsed as timestamps ({exc}). "
            "Use an ISO-8601 date column, e.g. '2024-01-31'."
        ) from None
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    ordered: pd.DataFrame = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return ordered


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

    resolved_freq = freq or infer_frequency(df)
    name = handle or f"s{uuid.uuid4().hex[:6]}"
    replaced = name in _STORE

    series = SeriesHandle(
        handle=name,
        source=str(path.resolve()),
        df=df,
        freq=resolved_freq,
        sha256=sha256_file(path),
        loaded_at=datetime.now(timezone.utc).isoformat(),
    )
    put(series)
    return series, replaced


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
        "extra_columns": [c for c in df.columns if c not in REQUIRED_COLUMNS],
        "loaded_at": series.loaded_at,
        "replaced_existing_handle": replaced,
    }
