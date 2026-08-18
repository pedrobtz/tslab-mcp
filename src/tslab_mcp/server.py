"""MCP server: tool registration and dispatch.

Deliberately thin. Every tool is a docstring written for the agent that will
read it, a schema from :mod:`tslab_mcp.schemas`, and a synchronous closure
handed to a worker thread (design invariant I2 -- forecasting blocks for
minutes and must never occupy the event loop, or the stdio transport stops
answering and the client drops the server mid-run). Business logic lives in the
sibling modules; if a tool body grows past ~30 lines, it belongs somewhere else.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
import pandas as pd
from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from . import (
    __version__,
    artifacts,
    backends,
    evaluation,
    features,
    manifest,
    models,
    registry,
    report,
)
from .schemas import (
    CrossValidateInput,
    DescribeSeriesInput,
    DetectAnomaliesInput,
    ExportReportInput,
    ExportRunInput,
    ForecastInput,
    ListModelsInput,
    LoadSeriesInput,
    ReportFormat,
    ResponseFormat,
)

mcp = MCPServer(
    name="tslab-mcp",
    version=__version__,
    instructions=(
        "Deterministic time-series forecasting. You are the reasoning engine: "
        "these tools compute numbers, you interpret them. The workflow is "
        "tsf_load_series -> tsf_describe_series -> tsf_list_models -> "
        "tsf_cross_validate -> tsf_forecast -> tsf_export_run. Panels stay in "
        "the server behind a handle; bulk results are written to parquet and "
        "only summarised in the response, so read the artifact path rather "
        "than asking for more rows."
    ),
)

def _read_only(title: str) -> ToolAnnotations:
    """Annotations for a tool that computes and writes artifacts but mutates
    no state the caller can observe through this server."""
    return ToolAnnotations(
        title=title,
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )

#: TimeCopilot names the flag column per model, as ``"{alias}-anomaly"``. The
#: alias is usually the model name but need not be, so the column is located by
#: suffix rather than assembled from the requested name.
_ANOMALY_SUFFIX = "-anomaly"


def _dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str)


def _preview(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    """JSON-safe preview rows, capped by ``limit``."""
    if limit <= 0 or df.empty:
        return []
    head = df.head(limit)
    rows: list[dict[str, Any]] = json.loads(
        head.to_json(orient="records", date_format="iso")
    )
    return rows


@mcp.tool(name="tsf_load_series", annotations=_read_only("Load a time series panel"))
async def tsf_load_series(params: LoadSeriesInput) -> str:
    """Read a CSV or Parquet panel from disk and register it under a handle.

    Call this first; every other tool takes the handle it returns. The file must
    be in Nixtla long format (unique_id, ds, y). Returns a compact JSON summary
    -- series count, inferred frequency, date range, missing values, and the
    SHA-256 of the source -- and nothing else: the data stays in the server so
    it never consumes your context.

    Read the summary before choosing a horizon. If obs_per_series.min is small,
    a long horizon or many CV windows will not fit.
    """

    def _run() -> str:
        series, replaced = registry.load_panel(params.path, params.freq, params.handle)
        return _dumps(registry.summarize(series, replaced))

    return await anyio.to_thread.run_sync(_run)


@mcp.tool(
    name="tsf_describe_series",
    annotations=_read_only("Statistical features of a loaded panel"),
)
async def tsf_describe_series(params: DescribeSeriesInput) -> str:
    """Compute the per-series features that decide which model family to try.

    Returns length, mean, sd, coefficient of variation, share of zeros, trend
    strength (R-squared against time), seasonal strength (variance explained by
    the period means), and lag-1 autocorrelation of the differenced series.

    Read it as evidence, not as an answer: high seasonal_strength argues for
    SeasonalNaive or AutoETS; a high pct_zero argues for the intermittent-demand
    models (ADIDA, IMAPA, CrostonClassic); high cv with low structure argues for
    keeping expectations modest. Cheap -- returns in under a second.
    """

    def _run() -> str:
        series = registry.get(params.handle)
        period = features.seasonal_period(series.freq)
        rows = features.panel_features(series.df, period)
        shown = rows[: params.max_series]
        omitted = len(rows) - len(shown)

        payload = {
            "handle": series.handle,
            "freq": series.freq,
            "seasonal_period": period,
            "n_series": len(rows),
            "shown": len(shown),
            "truncated": omitted > 0,
            "n_omitted": omitted,
            "features": shown,
        }

        # Features are the evidence behind the model-family decision, so they
        # belong in the run log rather than only in the conversation.
        entry = manifest.record(
            "features",
            seasonal_period=period,
            n_series=len(rows),
            shown=len(shown),
            truncated=omitted > 0,
            n_omitted=omitted,
            features=shown,
        )
        series.runs.append(entry)

        if params.response_format is ResponseFormat.JSON:
            return _dumps(payload)
        return _markdown_features(payload)

    return await anyio.to_thread.run_sync(_run)


def _markdown_features(payload: dict[str, Any]) -> str:
    """Render the feature payload as a table for direct reading."""
    header = (
        f"**{payload['handle']}** - {payload['n_series']} series, "
        f"freq `{payload['freq']}`, seasonal period {payload['seasonal_period']}"
    )
    lines = [
        header,
        "",
        "| id | n | mean | sd | cv | %zero | trend | seasonal | acf1(diff) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["features"]:
        cells = [
            row["unique_id"],
            row["n"],
            row["mean"],
            row["sd"],
            row["cv"],
            row["pct_zero"],
            row["trend_strength"],
            row["seasonal_strength"],
            row["acf1_diff"],
        ]
        rendered = " | ".join("-" if c is None else str(c) for c in cells)
        lines.append(f"| {rendered} |")
    if payload["truncated"]:
        lines.append(
            f"\n_{payload['n_omitted']} further series omitted; re-call with a "
            "higher max_series or response_format='json' if you need them._"
        )
    return "\n".join(lines)


@mcp.tool(
    name="tsf_list_models",
    annotations=_read_only("Probe available forecasting models"),
)
async def tsf_list_models(params: ListModelsInput) -> str:
    """Probe which models actually import in this environment.

    Call this before cross-validating so you never propose a model that cannot
    run here. Returns {available, statistical, foundation, unavailable}, where
    each unavailable entry carries the real reason -- some models are gated on
    the Python version, not merely absent.

    The first call imports TimeCopilot and can take ~30 seconds; later calls are
    instant.
    """

    def _run() -> str:
        return _dumps(models.probe(params.family.value, params.include_unavailable))

    return await anyio.to_thread.run_sync(_run)


@mcp.tool(
    name="tsf_cross_validate",
    annotations=_read_only("Rolling-origin model comparison"),
)
async def tsf_cross_validate(params: CrossValidateInput) -> str:
    """Compare models by rolling-origin cross-validation.

    This is the tool that replaces guesswork about model choice: it produces the
    evidence, you read the table and decide. Always include SeasonalNaive as the
    baseline -- a model that cannot beat it is not worth deploying.

    Returns a per-model metric table aggregated over series and windows, a
    best-first ranking per metric, and the parquet path holding every per-window
    prediction. LONG-RUNNING: seconds for statistical models, many minutes for
    foundation models on a large panel. Start with statistical models on the
    real horizon before reaching for anything pretrained.
    """

    def _run() -> str:
        series = registry.get(params.handle)
        names = list(params.models)
        metric_names = [metric.value for metric in params.metrics]

        models.validate_names(names)
        backend = backends.select(names)
        cv_df = backend.cross_validation(
            df=series.df,
            h=params.h,
            freq=series.freq,
            n_windows=params.n_windows,
            step_size=params.step_size,
        )
        path = artifacts.write_parquet(cv_df, "cv", series.handle)
        scores = evaluation.evaluate_cv(
            cv_df, series.df, names, metric_names, series.freq
        )

        entry = manifest.record(
            "cross_validation",
            models=names,
            backend=backend.name,
            h=params.h,
            n_windows=params.n_windows,
            step_size=params.step_size,
            metrics_requested=metric_names,
            artifact=str(path),
            n_rows=len(cv_df),
            **scores,
        )
        series.runs.append(entry)
        return _dumps(entry)

    return await anyio.to_thread.run_sync(_run)


@mcp.tool(name="tsf_forecast", annotations=_read_only("Generate a forecast"))
async def tsf_forecast(params: ForecastInput) -> str:
    """Fit on the full history and forecast h periods ahead with intervals.

    Use after tsf_cross_validate has justified the model choice. The full
    forecast goes to parquet; the response carries the path, the column list,
    the row count and a small preview. Read the parquet for anything more --
    raising max_preview_rows to dump the frame into the conversation is the one
    thing that reliably ruins a long session.

    LONG-RUNNING for foundation models.
    """

    def _run() -> str:
        series = registry.get(params.handle)
        names = list(params.models)

        models.validate_names(names)
        backend = backends.select(names)
        forecast_df = backend.forecast(
            df=series.df, h=params.h, freq=series.freq, level=params.level
        )
        path = artifacts.write_parquet(forecast_df, "fcst", series.handle)

        first_id = forecast_df["unique_id"].iloc[0]
        preview = _preview(
            forecast_df[forecast_df["unique_id"] == first_id], params.max_preview_rows
        )
        entry = manifest.record(
            "forecast",
            models=names,
            backend=backend.name,
            h=params.h,
            level=params.level,
            artifact=str(path),
            n_rows=len(forecast_df),
            columns=list(forecast_df.columns),
            preview_series=str(first_id),
            preview=preview,
            truncated=len(forecast_df) > len(preview),
        )
        series.runs.append(entry)
        return _dumps(entry)

    return await anyio.to_thread.run_sync(_run)


@mcp.tool(
    name="tsf_detect_anomalies",
    annotations=_read_only("Detect anomalies via cross-validated intervals"),
)
async def tsf_detect_anomalies(params: DetectAnomaliesInput) -> str:
    """Flag historical points that fall outside a cross-validated prediction
    interval.

    The detector model defines what "expected" means, so pick one that fits the
    series: a weak detector flags its own errors rather than real anomalies. Run
    tsf_describe_series or tsf_cross_validate first.

    Returns flagged counts per series, a capped list of flagged rows, and the
    parquet path with the full result.

    LONG-RUNNING, and the default is the expensive one: leaving n_windows unset
    refits the model once per observation across the whole history, which takes
    minutes even for a statistical model. Pass n_windows (e.g. 12) unless you
    genuinely need every point tested.
    """

    def _run() -> str:
        series = registry.get(params.handle)

        models.validate_names([params.model])
        backend = backends.select([params.model])
        result = backend.detect_anomalies(
            df=series.df,
            h=params.h,
            freq=series.freq,
            n_windows=params.n_windows,
            level=params.level,
        )
        path = artifacts.write_parquet(result, "anom", series.handle)

        columns = [c for c in result.columns if c.endswith(_ANOMALY_SUFFIX)]
        if columns:
            flag_column: str | None = columns[0]
            flagged = result[result[columns[0]].fillna(False).astype(bool)]
            note = None
        else:
            flag_column = None
            flagged = result.iloc[0:0]
            note = (
                f"No '*{_ANOMALY_SUFFIX}' column in the result; columns are "
                f"{list(result.columns)}. Counts are unavailable -- read the "
                "parquet artifact directly."
            )

        per_series = {
            str(uid): int(count)
            for uid, count in flagged.groupby("unique_id").size().items()
        }
        entry = manifest.record(
            "anomalies",
            model=params.model,
            backend=backend.name,
            h=params.h,
            n_windows=params.n_windows,
            level=params.level,
            flag_column=flag_column,
            artifact=str(path),
            n_rows_tested=len(result),
            n_flagged=len(flagged),
            per_series=per_series,
            flagged=_preview(flagged, params.max_flagged),
            truncated=len(flagged) > params.max_flagged,
            note=note,
        )
        series.runs.append(entry)
        return _dumps(entry)

    return await anyio.to_thread.run_sync(_run)


@mcp.tool(
    name="tsf_export_run",
    annotations=ToolAnnotations(
        title="Export a pinned, re-runnable manifest",
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=False,
    ),
)
async def tsf_export_run(params: ExportRunInput) -> str:
    """Write a JSON manifest of everything done to this handle.

    Records the source path and SHA-256, the frequency, every call with its
    arguments and artifact paths, the pinned package versions, and your note.
    This is the artifact of record: your prose in the conversation is lost, this
    file is not. Write the note -- say which model you picked, what the metric
    table showed, and what you rejected.

    Call it at the end of any analysis someone might have to defend or rerun.
    Writes a file, so it is not read-only.
    """

    def _run() -> str:
        series = registry.get(params.handle)
        payload = manifest.build(series, params.note)
        path = artifacts.write_json(payload, "manifest", series.handle)
        return _dumps(
            {
                "manifest": str(path),
                "n_runs": len(series.runs),
                "kinds": sorted({str(run["kind"]) for run in series.runs}),
                "environment": payload["environment"],
            }
        )

    return await anyio.to_thread.run_sync(_run)


@mcp.tool(
    name="tsf_export_report",
    annotations=ToolAnnotations(
        title="Export a readable report of the whole run",
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=False,
    ),
)
async def tsf_export_report(params: ExportReportInput) -> str:
    """Render every step of the analysis as a report someone can read.

    Covers the input and its hash, the features, each cross-validation with its
    metric table and ranking, the forecasts, any anomaly runs, and the pinned
    environment -- in the order they happened. HTML is self-contained, with no
    external stylesheet or script, so it opens correctly years later.

    Call it after tsf_export_run at the end of an analysis. Pass a note: the
    report headlines it as the rationale, and a table of numbers without the
    reasoning is what makes a reviewer ask for the whole thing again.

    Report from `manifest_path` instead of `handle` to re-render an older run --
    it needs nothing but the manifest file.
    """

    def _run() -> str:
        if params.manifest_path is not None:
            path = Path(params.manifest_path).expanduser()
            if not path.exists():
                raise ValueError(
                    f"No such manifest: {path}. Pass the path returned by "
                    "tsf_export_run, or use handle to report on this session."
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
            source = str(payload.get("input", {}).get("source", "manifest"))
            stem = Path(source).stem or "manifest"
        else:
            series = registry.get(str(params.handle))
            payload = manifest.build(series, params.note)
            stem = series.handle

        suffix = ".html" if params.format is ReportFormat.HTML else ".md"
        content = report.render(payload, params.format.value)
        written = artifacts.write_text(content, "report", stem, suffix)
        return _dumps(
            {
                "report": str(written),
                "format": params.format.value,
                "n_steps": len(payload.get("runs", [])),
                "steps": [str(run.get("kind")) for run in payload.get("runs", [])],
                "bytes": len(content.encode("utf-8")),
            }
        )

    return await anyio.to_thread.run_sync(_run)


def main() -> None:
    """Run the server over stdio.

    stdio only, by design: the panels are assumed sensitive and never leave the
    machine. The SDK's stdio transport serves the wire from a private duplicate
    of fd 1 and points fd 1 itself at stderr, so stray prints from the
    forecasting stack -- or from the worker processes statsforecast spawns --
    cannot corrupt the protocol.
    """
    mcp.run(transport="stdio")
