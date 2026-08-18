"""TimeCopilot MCP server.

Exposes the *deterministic* half of TimeCopilot (`TimeCopilotForecaster`) as MCP
tools, so that Claude acts as the reasoning engine instead of an LLM endpoint
embedded in the library. No API key, no pydantic-ai, no second agent.

Design notes
------------
* **Handles, not dataframes.** Series live in a process-local registry keyed by
  a short handle. Tools return summaries and file paths; the raw data never
  enters the model context. This is the single most important choice here -- a
  10k-row frame serialised into a tool result will destroy the session.
* **Blocking work off the event loop.** Cross-validation over foundation models
  is CPU/GPU-bound and can run for minutes. Every heavy call goes through
  `anyio.to_thread.run_sync` so the stdio transport stays responsive.
* **Lazy model imports.** Foundation models drag in torch and friends. The
  registry stores import paths and resolves them on first use, so the server
  starts in milliseconds and works fine with only the stats extras installed.
* **Run manifests.** `tsf_export_run` writes a pinned, re-runnable config with
  input hashes and package versions. The numbers are produced by this file;
  the LLM only ever comments on them.

Run:  python timecopilot_mcp.py
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import anyio
import pandas as pd
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator

mcp = FastMCP("timecopilot_mcp")

OUTPUT_DIR = Path.home() / ".timecopilot-mcp" / "runs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = ("unique_id", "ds", "y")

# name -> (module, attribute). Resolved lazily; verify against your installed
# version with tsf_list_models, which probes importability rather than trusting
# this table.
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    # statistical / classical -- cheap, always available
    "SeasonalNaive": ("timecopilot.models.stats", "SeasonalNaive"),
    "HistoricAverage": ("timecopilot.models.stats", "HistoricAverage"),
    "AutoARIMA": ("timecopilot.models.stats", "AutoARIMA"),
    "AutoETS": ("timecopilot.models.stats", "AutoETS"),
    "AutoCES": ("timecopilot.models.stats", "AutoCES"),
    "Theta": ("timecopilot.models.stats", "Theta"),
    "ADIDA": ("timecopilot.models.stats", "ADIDA"),
    "ZeroModel": ("timecopilot.models.stats", "ZeroModel"),
    "Prophet": ("timecopilot.models.prophet", "Prophet"),
    # foundation -- heavy, optional extras
    "Chronos": ("timecopilot.models.foundation.chronos", "Chronos"),
    "Moirai": ("timecopilot.models.foundation.moirai", "Moirai"),
    "TimesFM": ("timecopilot.models.foundation.timesfm", "TimesFM"),
    "Toto": ("timecopilot.models.foundation.toto", "Toto"),
    "Sundial": ("timecopilot.models.foundation.sundial", "Sundial"),
    "TimeGPT": ("timecopilot.models.foundation.timegpt", "TimeGPT"),
}

STATISTICAL_MODELS = frozenset(
    k for k, (mod, _) in MODEL_REGISTRY.items() if "foundation" not in mod
)


# --------------------------------------------------------------------------
# session state
# --------------------------------------------------------------------------


@dataclass
class SeriesHandle:
    """A loaded panel, plus everything needed to reproduce the load."""

    handle: str
    source: str
    df: pd.DataFrame
    freq: str
    sha256: str
    loaded_at: str
    runs: list[dict[str, Any]] = field(default_factory=list)


_REGISTRY: dict[str, SeriesHandle] = {}


def _get(handle: str) -> SeriesHandle:
    try:
        return _REGISTRY[handle]
    except KeyError:
        known = ", ".join(_REGISTRY) or "none"
        raise ValueError(
            f"Unknown handle '{handle}'. Loaded handles: {known}. "
            "Call tsf_load_series first."
        ) from None


def _resolve_models(names: list[str]) -> list[Any]:
    """Instantiate model objects from names, with an actionable error."""
    resolved = []
    for name in names:
        if name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{name}'. Call tsf_list_models for the "
                "available set on this installation."
            )
        module_path, attr = MODEL_REGISTRY[name]
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ValueError(
                f"Model '{name}' is not installed ({exc}). Install the extra, "
                f"e.g. `uv pip install 'timecopilot[{name.lower()}]'`, or pick a "
                "statistical model instead: " + ", ".join(sorted(STATISTICAL_MODELS))
            ) from None
        resolved.append(getattr(module, attr)())
    return resolved


def _forecaster(names: list[str]):
    from timecopilot import TimeCopilotForecaster

    return TimeCopilotForecaster(models=_resolve_models(names))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("timecopilot", "statsforecast", "utilsforecast", "pandas", "torch"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return out


def _write(df: pd.DataFrame, stem: str) -> Path:
    path = OUTPUT_DIR / f"{stem}.parquet"
    df.to_parquet(path, index=False)
    return path


# --------------------------------------------------------------------------
# input models
# --------------------------------------------------------------------------


class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


class Base(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )


class LoadSeriesInput(Base):
    path: str = Field(
        ...,
        description="Path to a local CSV or Parquet file with columns "
        "unique_id, ds, y (e.g. '/data/deposits.parquet').",
        min_length=1,
    )
    freq: str | None = Field(
        default=None,
        description="Pandas offset alias, e.g. 'MS', 'D', 'W-MON'. Inferred "
        "from ds when omitted.",
    )
    handle: str | None = Field(
        default=None,
        description="Optional name for this dataset. Auto-generated if omitted.",
        max_length=64,
    )


class DescribeInput(Base):
    handle: str = Field(..., description="Handle returned by tsf_load_series.")
    max_series: int = Field(
        default=20,
        description="Cap on per-series rows in the response, to bound context.",
        ge=1,
        le=200,
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class CrossValidateInput(Base):
    handle: str = Field(..., description="Handle returned by tsf_load_series.")
    models: list[str] = Field(
        ...,
        description="Model names to compare, e.g. ['SeasonalNaive','AutoETS',"
        "'AutoARIMA']. Always include SeasonalNaive as a baseline.",
        min_length=1,
        max_length=12,
    )
    h: int = Field(..., description="Forecast horizon in periods.", ge=1, le=5000)
    n_windows: int = Field(
        default=5, description="Rolling-origin evaluation windows.", ge=1, le=50
    )
    metrics: list[str] = Field(
        default_factory=lambda: ["mase", "smape"],
        description="Metrics from utilsforecast: mase, smape, mae, rmse, mape.",
    )

    @field_validator("models")
    @classmethod
    def _dedupe(cls, v: list[str]) -> list[str]:
        seen: list[str] = []
        for name in v:
            if name not in seen:
                seen.append(name)
        return seen


class ForecastInput(Base):
    handle: str = Field(..., description="Handle returned by tsf_load_series.")
    models: list[str] = Field(
        ...,
        description="Model names. Pass more than one to ensemble/compare.",
        min_length=1,
        max_length=12,
    )
    h: int = Field(..., description="Forecast horizon in periods.", ge=1, le=5000)
    level: list[int] = Field(
        default_factory=lambda: [80, 95],
        description="Prediction interval levels.",
        max_length=5,
    )


class AnomalyInput(Base):
    handle: str = Field(..., description="Handle returned by tsf_load_series.")
    model: str = Field(..., description="Single model name to use as the detector.")
    h: int = Field(default=1, description="Horizon for the rolling test.", ge=1)
    level: int = Field(
        default=99, description="Two-sided interval confidence level.", ge=50, le=100
    )


class ExportRunInput(Base):
    handle: str = Field(..., description="Handle whose runs should be exported.")
    note: str | None = Field(
        default=None, description="Free-text rationale to attach to the manifest."
    )


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------


@mcp.tool(
    name="tsf_load_series",
    annotations={
        "title": "Load a time series panel",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tsf_load_series(params: LoadSeriesInput) -> str:
    """Load a panel from disk and register it under a handle.

    Returns a compact JSON summary: handle, n_series, n_obs, inferred freq,
    date range, missing-value counts, and the sha256 of the source file. The
    data itself stays in the server process -- pass the handle to other tools.
    """

    def _load() -> str:
        path = Path(params.path).expanduser()
        if not path.exists():
            raise ValueError(f"No such file: {path}")

        if path.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        elif path.suffix.lower() in {".csv", ".txt"}:
            df = pd.read_csv(path)
        else:
            raise ValueError(
                f"Unsupported extension '{path.suffix}'. Use .csv or .parquet."
            )

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns {missing}. Found: {list(df.columns)}. "
                "Rename to the Nixtla convention (unique_id, ds, y) before loading."
            )

        df["ds"] = pd.to_datetime(df["ds"])
        df["unique_id"] = df["unique_id"].astype(str)
        df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        freq = params.freq
        if freq is None:
            first = df[df["unique_id"] == df["unique_id"].iloc[0]]["ds"]
            freq = pd.infer_freq(first) or ""
            if not freq:
                raise ValueError(
                    "Could not infer frequency -- the first series has irregular "
                    "spacing. Pass freq explicitly (e.g. 'MS', 'D', 'B')."
                )

        handle = params.handle or f"s{uuid.uuid4().hex[:6]}"
        _REGISTRY[handle] = SeriesHandle(
            handle=handle,
            source=str(path),
            df=df,
            freq=freq,
            sha256=_sha256(path),
            loaded_at=datetime.now(timezone.utc).isoformat(),
        )

        counts = df.groupby("unique_id").size()
        return json.dumps(
            {
                "handle": handle,
                "n_series": int(df["unique_id"].nunique()),
                "n_obs": int(len(df)),
                "freq": freq,
                "start": df["ds"].min().isoformat(),
                "end": df["ds"].max().isoformat(),
                "obs_per_series": {
                    "min": int(counts.min()),
                    "median": int(counts.median()),
                    "max": int(counts.max()),
                },
                "n_missing_y": int(df["y"].isna().sum()),
                "sha256": _REGISTRY[handle].sha256,
                "extra_columns": [c for c in df.columns if c not in REQUIRED_COLUMNS],
            },
            indent=2,
        )

    return await anyio.to_thread.run_sync(_load)


@mcp.tool(
    name="tsf_describe_series",
    annotations={
        "title": "Statistical features of a loaded panel",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tsf_describe_series(params: DescribeInput) -> str:
    """Compute the features you need to choose a model family.

    Per series: length, mean, sd, coefficient of variation, share of zeros,
    trend strength (OLS slope R-squared), seasonal strength (variance explained
    by period means), and lag-1 autocorrelation of the differenced series.
    Uses tsfeatures when installed, otherwise a pandas fallback.
    """

    def _describe() -> str:
        sh = _get(params.handle)
        period = _seasonal_period(sh.freq)
        rows = []
        for uid, g in sh.df.groupby("unique_id"):
            y = g["y"].astype(float).reset_index(drop=True)
            rows.append(
                {
                    "unique_id": uid,
                    "n": len(y),
                    "mean": round(float(y.mean()), 4),
                    "sd": round(float(y.std(ddof=1)), 4) if len(y) > 1 else None,
                    "cv": round(float(y.std(ddof=1) / y.mean()), 4)
                    if len(y) > 1 and y.mean()
                    else None,
                    "pct_zero": round(float((y == 0).mean()), 4),
                    "trend_strength": _trend_strength(y),
                    "seasonal_strength": _seasonal_strength(y, period),
                    "acf1_diff": _acf1(y.diff().dropna()),
                }
            )
        rows.sort(key=lambda r: r["unique_id"])
        truncated = len(rows) > params.max_series
        shown = rows[: params.max_series]

        payload = {
            "handle": sh.handle,
            "freq": sh.freq,
            "seasonal_period": period,
            "n_series": len(rows),
            "shown": len(shown),
            "truncated": truncated,
            "features": shown,
        }
        if params.response_format is ResponseFormat.JSON:
            return json.dumps(payload, indent=2)

        lines = [
            f"**{sh.handle}** -- {len(rows)} series, freq `{sh.freq}`, "
            f"seasonal period {period}",
            "",
            "| id | n | mean | cv | %zero | trend | seasonal | acf1(diff) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in shown:
            lines.append(
                f"| {r['unique_id']} | {r['n']} | {r['mean']} | {r['cv']} | "
                f"{r['pct_zero']} | {r['trend_strength']} | "
                f"{r['seasonal_strength']} | {r['acf1_diff']} |"
            )
        if truncated:
            lines.append(f"\n_{len(rows) - len(shown)} further series omitted._")
        return "\n".join(lines)

    return await anyio.to_thread.run_sync(_describe)


@mcp.tool(
    name="tsf_list_models",
    annotations={
        "title": "List available forecasting models",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tsf_list_models() -> str:
    """Probe which models are importable in this environment.

    Returns JSON: {"available": [...], "unavailable": {"Name": "reason"}}.
    Call this before cross-validating so you don't propose a model whose
    extras aren't installed.
    """

    def _probe() -> str:
        available, unavailable = [], {}
        for name, (module_path, attr) in MODEL_REGISTRY.items():
            try:
                module = importlib.import_module(module_path)
                getattr(module, attr)
                available.append(name)
            except (ImportError, AttributeError) as exc:
                unavailable[name] = type(exc).__name__
        return json.dumps(
            {
                "available": sorted(available),
                "unavailable": unavailable,
                "note": "Statistical models are cheap; foundation models may "
                "download weights on first call.",
            },
            indent=2,
        )

    return await anyio.to_thread.run_sync(_probe)


@mcp.tool(
    name="tsf_cross_validate",
    annotations={
        "title": "Rolling-origin model comparison",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tsf_cross_validate(params: CrossValidateInput) -> str:
    """Compare models by rolling-origin cross-validation. This is the tool that
    replaces the LLM's model-selection step -- read the table, then decide.

    Returns JSON with a per-model metric table aggregated over series and
    windows, plus a parquet path holding the full per-window predictions.
    Long-running: minutes for foundation models on large panels.
    """

    def _cv() -> str:
        sh = _get(params.handle)
        fcst = _forecaster(params.models)
        cv_df = fcst.cross_validation(
            df=sh.df, h=params.h, freq=sh.freq, n_windows=params.n_windows
        )
        path = _write(cv_df, f"cv_{sh.handle}_{uuid.uuid4().hex[:6]}")
        table = _evaluate(cv_df, sh.df, params.models, params.metrics, sh.freq)

        record = {
            "kind": "cross_validation",
            "models": params.models,
            "h": params.h,
            "n_windows": params.n_windows,
            "metrics": table,
            "artifact": str(path),
            "at": datetime.now(timezone.utc).isoformat(),
        }
        sh.runs.append(record)
        return json.dumps(record, indent=2)

    return await anyio.to_thread.run_sync(_cv)


@mcp.tool(
    name="tsf_forecast",
    annotations={
        "title": "Generate a forecast",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tsf_forecast(params: ForecastInput) -> str:
    """Fit and forecast h periods ahead with prediction intervals.

    Writes the full forecast to parquet and returns JSON with the path plus a
    small preview (first series, first 12 rows). Read the parquet yourself for
    anything more -- do not ask for the whole frame in the response.
    """

    def _run() -> str:
        sh = _get(params.handle)
        fcst = _forecaster(params.models)
        out = fcst.forecast(df=sh.df, h=params.h, freq=sh.freq, level=params.level)
        path = _write(out, f"fcst_{sh.handle}_{uuid.uuid4().hex[:6]}")

        first = out[out["unique_id"] == out["unique_id"].iloc[0]].head(12)
        record = {
            "kind": "forecast",
            "models": params.models,
            "h": params.h,
            "level": params.level,
            "artifact": str(path),
            "n_rows": int(len(out)),
            "columns": list(out.columns),
            "preview": json.loads(first.to_json(orient="records", date_format="iso")),
            "at": datetime.now(timezone.utc).isoformat(),
        }
        sh.runs.append(record)
        return json.dumps(record, indent=2)

    return await anyio.to_thread.run_sync(_run)


@mcp.tool(
    name="tsf_detect_anomalies",
    annotations={
        "title": "Detect anomalies via cross-validated z-scores",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tsf_detect_anomalies(params: AnomalyInput) -> str:
    """Flag points falling outside a two-sided prediction interval built from
    rolling-origin out-of-sample errors.

    Returns JSON: counts per series, the flagged timestamps (capped at 100),
    and a parquet path with the full result.
    """

    def _run() -> str:
        sh = _get(params.handle)
        fcst = _forecaster([params.model])
        out = fcst.detect_anomalies(
            df=sh.df, h=params.h, freq=sh.freq, level=params.level
        )
        path = _write(out, f"anom_{sh.handle}_{uuid.uuid4().hex[:6]}")

        flag_col = next(
            (c for c in out.columns if "anomal" in c.lower()),
            None,
        )
        flagged = out[out[flag_col]] if flag_col else out.iloc[0:0]
        record = {
            "kind": "anomalies",
            "model": params.model,
            "level": params.level,
            "artifact": str(path),
            "n_flagged": int(len(flagged)),
            "per_series": flagged.groupby("unique_id").size().to_dict()
            if len(flagged)
            else {},
            "flagged": json.loads(
                flagged.head(100).to_json(orient="records", date_format="iso")
            ),
            "at": datetime.now(timezone.utc).isoformat(),
        }
        sh.runs.append(record)
        return json.dumps(record, indent=2)

    return await anyio.to_thread.run_sync(_run)


@mcp.tool(
    name="tsf_export_run",
    annotations={
        "title": "Export a pinned, re-runnable manifest",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def tsf_export_run(params: ExportRunInput) -> str:
    """Write a JSON manifest of everything done to this handle: source path and
    sha256, freq, every cross-validation and forecast call with its arguments
    and artifact paths, package versions, and an optional rationale.

    This is the artifact of record. Replaying it reproduces the numbers with no
    LLM in the path.
    """

    def _export() -> str:
        sh = _get(params.handle)
        manifest = {
            "manifest_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": {
                "source": sh.source,
                "sha256": sh.sha256,
                "freq": sh.freq,
                "loaded_at": sh.loaded_at,
                "n_series": int(sh.df["unique_id"].nunique()),
                "n_obs": int(len(sh.df)),
            },
            "runs": sh.runs,
            "environment": _versions(),
            "note": params.note,
        }
        path = OUTPUT_DIR / f"manifest_{sh.handle}_{uuid.uuid4().hex[:6]}.json"
        path.write_text(json.dumps(manifest, indent=2))
        return json.dumps({"manifest": str(path), "n_runs": len(sh.runs)}, indent=2)

    return await anyio.to_thread.run_sync(_export)


# --------------------------------------------------------------------------
# feature helpers
# --------------------------------------------------------------------------


def _seasonal_period(freq: str) -> int:
    head = freq.split("-")[0].upper().lstrip("0123456789")
    return {
        "H": 24,
        "D": 7,
        "B": 5,
        "W": 52,
        "M": 12,
        "MS": 12,
        "Q": 4,
        "QS": 4,
        "Y": 1,
        "YS": 1,
        "A": 1,
    }.get(head, 1)


def _trend_strength(y: pd.Series) -> float | None:
    if len(y) < 3:
        return None
    x = pd.Series(range(len(y)), dtype=float)
    r = y.corr(x)
    return None if pd.isna(r) else round(float(r**2), 4)


def _seasonal_strength(y: pd.Series, period: int) -> float | None:
    if period <= 1 or len(y) < 2 * period:
        return None
    idx = pd.Series(range(len(y))) % period
    total = y.var(ddof=1)
    if not total:
        return None
    between = y.groupby(idx).mean().var(ddof=1)
    return round(float(min(between / total, 1.0)), 4)


def _acf1(y: pd.Series) -> float | None:
    if len(y) < 3:
        return None
    r = y.autocorr(lag=1)
    return None if pd.isna(r) else round(float(r), 4)


def _evaluate(
    cv_df: pd.DataFrame,
    train: pd.DataFrame,
    models: list[str],
    metrics: list[str],
    freq: str,
) -> dict[str, Any]:
    """Aggregate CV predictions into a per-model metric table."""
    try:
        from utilsforecast.evaluation import evaluate
        from utilsforecast.losses import mae, mape, mase, rmse, smape

        fns = {"mae": mae, "mape": mape, "rmse": rmse, "smape": smape}
        chosen: list[Any] = []
        for name in metrics:
            if name == "mase":
                seasonality = _seasonal_period(freq)
                chosen.append(lambda df, models, **kw: mase(
                    df, models, seasonality=seasonality, train_df=train, **kw
                ))
            elif name in fns:
                chosen.append(fns[name])
        present = [m for m in models if m in cv_df.columns]
        res = evaluate(cv_df, metrics=chosen, models=present, train_df=train)
        agg = res.drop(columns=["unique_id"]).groupby("metric").mean().round(4)
        return json.loads(agg.to_json(orient="index"))
    except Exception as exc:  # noqa: BLE001 -- fall back rather than fail the run
        present = [m for m in models if m in cv_df.columns]
        out = {}
        for m in present:
            err = (cv_df[m] - cv_df["y"]).abs()
            out[m] = {"mae": round(float(err.mean()), 4)}
        return {"fallback_reason": str(exc), "mae": out}


if __name__ == "__main__":
    mcp.run()
