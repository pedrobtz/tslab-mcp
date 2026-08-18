# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --all-groups                      # set up (installs ~291 packages, ~2.3 GB)
uv run pytest                             # fast suite (~1.5s); excludes `slow` via addopts
uv run pytest -m slow                     # exercises TimeCopilot (~3.5 min, no weight downloads)
uv run pytest -m ''                       # everything
uv run pytest tests/test_features.py::test_acf1_within_bounds   # single test
uv run ruff check src tests
uv run mypy                               # strict on src/; first cold run takes ~10 min
uv build
```

Verify the wire protocol, not just the functions — a tool can pass its unit test
and still fail a client handshake:

```bash
npx @modelcontextprotocol/inspector uv run tslab-mcp
```

## What this is

An MCP server exposing deterministic time-series forecasting. The session
driving the tools is the reasoning engine; this package computes numbers and
never calls an LLM.

Two backends, chosen automatically in `backends.select()`:

- **statsforecast** (core, always installed) runs the eleven statistical models.
  ~340 MB, no torch.
- **TimeCopilot** (optional `foundation` extra) runs the pretrained models and
  Prophet. ~2 GB, and only reached when a request names one of its models.

A request whose models are all statistical never touches TimeCopilot. A mixed
request goes entirely to TimeCopilot, which carries the statistical models too —
letting it merge the frames avoids reimplementing an upstream join.

## Architecture

Four invariants drive nearly every design decision. Breaking one looks locally
harmless and is not.

**Handles, not dataframes.** Panels live in `registry.py`'s module-level
`_STORE`, keyed by a short handle. Tools take the handle and return summaries,
aggregates, and file paths. A CV frame is `n_series × h × n_windows × n_models`
rows — serialising one into a tool result exhausts the session's context on the
first call. Every bulk return path is capped by a schema-enforced maximum and
reports `truncated` plus the omitted count.

**Blocking work never touches the event loop.** Every tool body is a sync
closure dispatched through `anyio.to_thread.run_sync`. Cross-validation runs for
minutes; a blocked loop means the stdio transport stops answering and the client
drops the server mid-run.

**TimeCopilot is imported lazily, always.** `import timecopilot` costs ~34
seconds, and it is now optional, so a module-scope import would break the base
install outright. All imports of it live inside function bodies in `backends.py`
and `models.py`. The same trap bites throwaway scripts: put TimeCopilot imports
inside `main()`, or the script will appear to hang — TimeCopilot runs
statsforecast with `n_jobs=-1`, and its spawned workers re-import the entry
module. Our own statsforecast backend leaves `n_jobs` at its default of 1 to
avoid that entirely.

**The manifest is the artifact of record.** Every tool that computes appends a
`manifest.record(...)` entry to `SeriesHandle.runs` — including
`tsf_describe_series`, whose features are the evidence behind the model choice.
`tsf_export_run` pins those to the input SHA-256 and installed package versions.
If a decision exists only in conversation prose, the design has failed — put it
in the run record or the `note` field.

`report.py` renders a manifest to HTML or Markdown and is a **pure function of
the manifest**: it reads no parquet, calls no model, and touches no registry.
Keep it that way — that property is what lets a report regenerate offline years
later. Both renderers share one extraction layer so their content cannot drift.

### Module boundaries

`server.py` is deliberately thin: a docstring, a schema, a closure, a thread. If
a tool body passes ~30 lines, the logic belongs in a sibling module.

- `schemas.py` — all validation. Tool bodies never hand-check inputs. Field
  descriptions are read by an agent deciding how to call, so they carry an
  example and real constraints.
- `features.py` — pure functions, no I/O, no TimeCopilot import. The most
  testable module; keep its tests densest. Every helper returns `None`, never
  `nan` (`nan` is not valid JSON and would break the response).
- `evaluation.py`, `registry.py` — must degrade rather than fail. A ten-minute
  CV run is never lost to an upstream signature change; failures return a
  fallback table plus `degraded_reason`.
- `models.py` — the registry, split into `STATS_MODELS` (statsforecast, always
  available) and `TIMECOPILOT_MODELS` (needs the extra). `probe()` records the
  exception **message**, not just its type: several models fail with
  `"requires Python < 3.13"`, and the message is the actionable part.
- `backends.py` — both engines behind one protocol. They must return identical
  frame shapes, because the tool layer, run records and reports all depend on
  it; `detect_anomalies` is reimplemented for statsforecast and must keep
  emitting `{alias}-anomaly`.
- `seasonality.py` — the shared seasonal period. Do not route it through
  gluonts or TimeCopilot; the base install has neither.

### Errors are prompts for the agent's next action

Every raised message states what went wrong, the observed state, and the
specific next call — `Unknown handle 'x'. Loaded handles: … Call tsf_load_series
first.` Never let a bare traceback reach the client.

## Landmines verified against the installed versions

These were each checked by running them against the installed libraries. Several
contradict what the documentation implies, so trust this list over a plausible-
looking snippet.

- **The MCP SDK is `mcp` 2.0.** `mcp.server.fastmcp.FastMCP` does not exist. Use
  `MCPServer` from `mcp.server`. `ToolAnnotations` fields are **snake_case**
  (`read_only_hint`), `list_tools()` is async, and `CallToolResult.is_error`
  replaced `isError`.
- **`utilsforecast.evaluate` returns a `cutoff` column.** Aggregate over model
  columns explicitly — `.drop(columns=["unique_id"]).groupby("metric").mean()`
  tries to average a timestamp. Covered by
  `test_cutoff_column_is_not_averaged`.
- **Anomaly flags are per-model columns named `"{alias}-anomaly"`**, not a
  single `anomaly` column. Locate by suffix.
- **Seasonality is deliberately computed two ways.** `seasonality.py`
  reimplements the gluonts/M4 convention (`D → 1`, `W → 1`) and is used for both
  a model's `season_length` and MASE, so the two backends agree and no install
  needs gluonts. `features.py` keeps a richer table (`D → 7`, `W → 52`) for the
  descriptive seasonal-strength feature only. Verified to match gluonts exactly
  across 20 aliases; changing it silently changes every seasonal forecast.
- **`seasonal_strength` is STL-based and must stay trend independent.** The
  obvious "variance explained by calendar position" measure falls from 1.00 to
  0.03 on identical seasonality as the trend steepens, which tells an agent that
  a growing seasonal series is not seasonal. A parametrised regression test pins
  this. Note the ~0.3–0.5 noise floor, and that neither statsmodels nor
  tsfeatures rejects NaN input — tsfeatures scores such a series 1.0 — so the
  guards in `seasonal_strength` are load-bearing, not decoration.
- **stdout needs no protection.** The SDK's stdio transport already serves the
  wire from a private CLOEXEC duplicate of fd 1 and points fd 1 at stderr, so
  stray prints from the forecasting stack and its worker processes cannot
  corrupt the JSON-RPC framing.
- **`detect_anomalies` defaults `n_windows` to the maximum**, refitting once per
  observation. Always pass it.

## Dependency constraints

- `pandas` floor is `>=2.1.2`, not 2.2. On Python < 3.13 TimeCopilot pins
  `tabpfn-time-series`, which caps `pandas<2.2.0`; requiring 2.2 makes the
  project unresolvable on 3.10–3.12. The seasonal-period table therefore accepts
  both alias generations (`M` and `ME`, `H` and `h`).
- Python 3.13 is the dev target (`.python-version`): it loses only `TabPFN` and
  `Sundial` and gains pandas 2.2+.
- TimeCopilot publishes **no per-model extras** — every model ships in its base
  install, which is why it is all-or-nothing behind our `foundation` extra.
- The core depends on `statsforecast` and `utilsforecast` only; adding anything
  that pulls torch to the core defeats the point of the split.
