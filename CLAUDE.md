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

An MCP server wrapping `TimeCopilotForecaster` — the *deterministic* half of
TimeCopilot. The session driving the tools is the reasoning engine; this package
computes numbers and never calls an LLM. Nothing under `src/tslab_mcp/` may
import `pydantic_ai`, `openai`, or `anthropic` (they are present in the
dependency tree only because TimeCopilot depends on them unconditionally).

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
seconds. All imports of it live inside `models.py` function bodies. This is not
only about startup: TimeCopilot runs statsforecast with `n_jobs=-1` outside a
daemon process, and the spawned workers re-import the entry module — a
module-scope import would be paid *per worker, per call*. The same trap bites
throwaway scripts: put TimeCopilot imports inside `main()`, or the script will
appear to hang.

**The manifest is the artifact of record.** Every tool that computes appends a
`manifest.record(...)` entry to `SeriesHandle.runs`; `tsf_export_run` pins those
to the input SHA-256 and installed package versions. If a decision exists only
in conversation prose, the design has failed — put it in the run record or the
`note` field.

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
- `models.py` — `MODEL_REGISTRY` maps friendly names to import paths. `probe()`
  records the exception **message**, not just its type: several models fail with
  `"requires Python < 3.13"`, and the message is the actionable part.

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
- **Seasonality is deliberately computed two ways.** `features.py` uses its own
  table (`D → 7`) for the descriptive seasonal-strength feature;
  `evaluation.py` prefers TimeCopilot's `get_seasonality` (M4 convention,
  `D → 1`) for MASE, and only when `timecopilot` is already in `sys.modules`, so
  a lookup never triggers the 34s import. The response states which was used.
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
- TimeCopilot publishes **no per-model extras** — every model ships in the base
  install. Only `distributed` exists to mirror.
