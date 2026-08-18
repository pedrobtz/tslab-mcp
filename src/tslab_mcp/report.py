"""Render a run manifest as a human-readable report.

A report is a *pure function of the manifest*: it reads no parquet, calls no
model, and touches no registry. That is the point -- the same manifest renders
to the same report months later, offline, with the server stopped, which is the
property that makes it worth showing to someone who has to sign off on a
forecast.

Two renderers share one extraction layer, so the Markdown and HTML versions
cannot drift apart in content, only in presentation. Neither needs a template
engine: the document is almost entirely generated tables.
"""

from __future__ import annotations

from html import escape
from typing import Any

Row = list[str]
Table = tuple[list[str], list[Row]]

#: Placeholder for a missing or non-finite value, in both renderers.
EMPTY = "—"

_KIND_TITLES = {
    "features": "Features",
    "cross_validation": "Cross-validation",
    "forecast": "Forecast",
    "anomalies": "Anomaly detection",
}


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    """Render a scalar for display, collapsing empties to a single marker."""
    if value is None or value == "":
        return EMPTY
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt(item) for item in value) if value else EMPTY
    if isinstance(value, dict):
        return ", ".join(f"{k}: {_fmt(v)}" for k, v in value.items()) or EMPTY
    return str(value)


def _pairs(items: list[tuple[str, Any]]) -> Table:
    """A two-column table of field/value pairs, skipping absent values."""
    rows = [[label, _fmt(value)] for label, value in items if value is not None]
    return ["Field", "Value"], rows


# --------------------------------------------------------------------------
# extraction: manifest -> tables (shared by both renderers)
# --------------------------------------------------------------------------


def _input_table(manifest: dict[str, Any]) -> Table:
    source = manifest.get("input", {})
    return _pairs(
        [
            ("Source", source.get("source")),
            ("SHA-256", source.get("sha256")),
            ("Frequency", source.get("freq")),
            ("Series", source.get("n_series")),
            ("Observations", source.get("n_obs")),
            ("Loaded at", source.get("loaded_at")),
        ]
    )


def _timeline_table(runs: list[dict[str, Any]]) -> Table:
    table: list[Row] = []
    for index, run in enumerate(runs, start=1):
        kind = str(run.get("kind", "?"))
        table.append(
            [
                str(index),
                _KIND_TITLES.get(kind, kind),
                _fmt(run.get("at")),
                _fmt(run.get("models") or run.get("model") or run.get("n_series")),
            ]
        )
    return ["#", "Step", "When (UTC)", "Models / scope"], table


def _features_table(run: dict[str, Any]) -> Table:
    headers = [
        "Series", "n", "Mean", "SD", "CV", "% zero",
        "Trend", "Seasonal", "ACF1(diff)",
    ]
    rows = [
        [
            _fmt(feature.get("unique_id")),
            _fmt(feature.get("n")),
            _fmt(feature.get("mean")),
            _fmt(feature.get("sd")),
            _fmt(feature.get("cv")),
            _fmt(feature.get("pct_zero")),
            _fmt(feature.get("trend_strength")),
            _fmt(feature.get("seasonal_strength")),
            _fmt(feature.get("acf1_diff")),
        ]
        for feature in run.get("features", [])
    ]
    return headers, rows


def _metrics_table(run: dict[str, Any]) -> Table:
    """Models as rows, metrics as columns, best-first where a ranking exists."""
    metrics: dict[str, dict[str, float]] = run.get("metrics", {})
    metric_names = list(metrics)
    models: list[str] = []
    for scores in metrics.values():
        for model in scores:
            if model not in models:
                models.append(model)

    ranking = run.get("ranking", {})
    if metric_names and metric_names[0] in ranking:
        order = ranking[metric_names[0]]
        models.sort(key=lambda m: order.index(m) if m in order else len(order))

    headers = ["Model", *metric_names]
    rows = [
        [model, *[_fmt(metrics[name].get(model)) for name in metric_names]]
        for model in models
    ]
    return headers, rows


def _preview_table(run: dict[str, Any]) -> Table:
    preview: list[dict[str, Any]] = run.get("preview") or run.get("flagged") or []
    if not preview:
        return [], []
    headers = list(preview[0])
    rows = [[_fmt(row.get(column)) for column in headers] for row in preview]
    return headers, rows


def _environment_table(manifest: dict[str, Any]) -> Table:
    environment = manifest.get("environment", {})
    return ["Package", "Version"], [
        [name, _fmt(value)] for name, value in environment.items()
    ]


def _run_details(run: dict[str, Any]) -> Table:
    """Per-run parameters, excluding the bulk payloads rendered separately."""
    skip = {"kind", "features", "metrics", "ranking", "preview", "flagged", "at"}
    return _pairs([(k.replace("_", " ").capitalize(), v)
                   for k, v in run.items() if k not in skip])


def _title(manifest: dict[str, Any]) -> str:
    source = manifest.get("input", {}).get("source", "")
    name = source.rsplit("/", 1)[-1] if source else "run"
    return f"Forecasting report — {name}"


# --------------------------------------------------------------------------
# Markdown renderer
# --------------------------------------------------------------------------


def _md_table(table: Table) -> list[str]:
    headers, rows = table
    if not headers or not rows:
        return ["_No rows._", ""]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    lines.append("")
    return lines


def render_markdown(manifest: dict[str, Any]) -> str:
    """Render a manifest as a Markdown report."""
    runs: list[dict[str, Any]] = manifest.get("runs", [])
    out: list[str] = [f"# {_title(manifest)}", ""]
    out.append(
        f"Generated {_fmt(manifest.get('created_at'))} by tslab-mcp "
        f"{_fmt(manifest.get('environment', {}).get('tslab_mcp'))}. "
        "Every number here was produced by a library call, not by a language "
        "model."
    )
    out.append("")

    if manifest.get("note"):
        out += ["## Rationale", "", f"> {manifest['note']}", ""]

    out += ["## Input data", "", *_md_table(_input_table(manifest))]
    out += ["## Steps", "", *_md_table(_timeline_table(runs))]

    for index, run in enumerate(runs, start=1):
        kind = str(run.get("kind", "?"))
        out += [f"## {index}. {_KIND_TITLES.get(kind, kind)}", ""]
        out += _md_table(_run_details(run))

        if kind == "features":
            out += _md_table(_features_table(run))
        elif kind == "cross_validation":
            out += _md_table(_metrics_table(run))
            if run.get("degraded_reason"):
                out += [f"**Degraded:** {run['degraded_reason']}", ""]
        else:
            preview = _preview_table(run)
            if preview[1]:
                label = "Flagged rows" if kind == "anomalies" else "Preview"
                out += [f"**{label}**", "", *_md_table(preview)]

    out += ["## Environment", "", *_md_table(_environment_table(manifest))]
    out += [
        "_Reproduce by replaying the steps above against the source file with "
        "the pinned versions; no language model is involved._",
        "",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------
# HTML renderer
# --------------------------------------------------------------------------

_CSS = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #5c6370; --line: #e2e5e9;
  --accent: #2f6feb; --head: #f6f8fa; --note: #f4f7ff;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1216; --fg: #e6e8eb; --muted: #9aa3ad; --line: #262c33;
    --accent: #6ea8ff; --head: #161b22; --note: #12203a;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2.5rem 1.25rem; background: var(--bg); color: var(--fg);
  font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
main { max-width: 60rem; margin: 0 auto; }
h1 { font-size: 1.7rem; margin: 0 0 .3rem; letter-spacing: -.01em; }
h2 {
  font-size: 1.15rem; margin: 2.5rem 0 .75rem; padding-bottom: .35rem;
  border-bottom: 1px solid var(--line);
}
p.lead { color: var(--muted); margin: 0 0 2rem; }
blockquote {
  margin: 0 0 1rem; padding: .85rem 1.1rem; background: var(--note);
  border-left: 3px solid var(--accent); border-radius: 0 4px 4px 0;
}
.scroll { overflow-x: auto; margin-bottom: 1rem; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td {
  text-align: left; padding: .45rem .7rem; border-bottom: 1px solid var(--line);
  white-space: nowrap;
}
th { background: var(--head); font-weight: 600; }
tbody tr:hover td { background: var(--head); }
td:first-child { font-variant-numeric: tabular-nums; }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.warn { color: #b3541e; font-weight: 600; }
footer { margin-top: 3rem; color: var(--muted); font-size: .85rem; }
"""


def _html_table(table: Table) -> str:
    headers, rows = table
    if not headers or not rows:
        return "<p><em>No rows.</em></p>"
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def render_html(manifest: dict[str, Any]) -> str:
    """Render a manifest as a self-contained HTML page.

    No external stylesheet, script, or font: the file opens correctly from a
    thumb drive in six months, which a CDN link would not.
    """
    runs: list[dict[str, Any]] = manifest.get("runs", [])
    title = _title(manifest)
    parts: list[str] = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{escape(title)}</title>",
        f"<style>{_CSS}</style></head><body><main>",
        f"<h1>{escape(title)}</h1>",
        '<p class="lead">Generated '
        f"{escape(_fmt(manifest.get('created_at')))} by tslab-mcp "
        f"{escape(_fmt(manifest.get('environment', {}).get('tslab_mcp')))}. "
        "Every number here was produced by a library call, not by a language "
        "model.</p>",
    ]

    if manifest.get("note"):
        parts.append("<h2>Rationale</h2>")
        parts.append(f"<blockquote>{escape(str(manifest['note']))}</blockquote>")

    parts.append("<h2>Input data</h2>")
    parts.append(_html_table(_input_table(manifest)))
    parts.append("<h2>Steps</h2>")
    parts.append(_html_table(_timeline_table(runs)))

    for index, run in enumerate(runs, start=1):
        kind = str(run.get("kind", "?"))
        parts.append(f"<h2>{index}. {escape(_KIND_TITLES.get(kind, kind))}</h2>")
        parts.append(_html_table(_run_details(run)))

        if kind == "features":
            parts.append(_html_table(_features_table(run)))
        elif kind == "cross_validation":
            parts.append(_html_table(_metrics_table(run)))
            if run.get("degraded_reason"):
                parts.append(
                    f'<p class="warn">Degraded: '
                    f"{escape(str(run['degraded_reason']))}</p>"
                )
        else:
            preview = _preview_table(run)
            if preview[1]:
                label = "Flagged rows" if kind == "anomalies" else "Preview"
                parts.append(f"<p><strong>{label}</strong></p>")
                parts.append(_html_table(preview))

    parts.append("<h2>Environment</h2>")
    parts.append(_html_table(_environment_table(manifest)))
    parts.append(
        "<footer>Reproduce by replaying the steps above against the source "
        "file with the pinned versions; no language model is involved."
        "</footer>"
    )
    parts.append("</main></body></html>")
    return "\n".join(parts)


def render(manifest: dict[str, Any], fmt: str) -> str:
    """Render ``manifest`` in ``fmt`` (``"html"`` or ``"markdown"``)."""
    if fmt == "html":
        return render_html(manifest)
    if fmt == "markdown":
        return render_markdown(manifest)
    raise ValueError(f"Unsupported report format '{fmt}'. Use 'html' or 'markdown'.")
