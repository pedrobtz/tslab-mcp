# `tslab-mcp` — Build Brief

> **Status: built and verified, 2026-08-18.** The package ships as `tslab-mcp`
> (module `tslab_mcp`), v0.1.0. Section 13 has been replaced with the results of
> the verification pass it asked for, and the claims it got wrong are corrected
> in place below. Section 15 records the packaging reality that differs from
> what section 11 assumed.

---

## 1. Goal

Build a local MCP server that exposes the deterministic forecasting core of
[TimeCopilot](https://timecopilot.dev) as tools, so that a Claude session acts
as the reasoning engine for a time-series workflow — feature interpretation,
model selection, evaluation, narrative — while every number is produced by
ordinary, reproducible Python.

The deliverable is an installable Python package, `timecopilot-mcp`, runnable as
`uvx timecopilot-mcp` or via a `mcpServers` stanza, with no LLM API key
anywhere in its dependency tree.

---

## 2. Background: why not the library's own agent

TimeCopilot ships two entry points:

| Class | LLM required | Role |
|---|---|---|
| `TimeCopilot` | yes (via pydantic-ai) | agentic wrapper: interprets features, picks models, orchestrates CV, writes the explanation |
| `TimeCopilotForecaster` | no | plain forecasting layer: `.forecast()`, `.cross_validation()`, `.detect_anomalies()` over a list of model objects |

The LLM in `TimeCopilot` does four things: read statistical features, select a
model family, decide the validation setup, and explain the result. Inside a
Claude session those are exactly the things the session is already doing. Using
`TimeCopilot` there means nesting one agent inside another — two prompts, two
bills, two sources of nondeterminism, and an opaque middle layer that makes the
model-selection rationale unauditable.

So: **invert control.** `TimeCopilotForecaster` becomes the tool; the session
becomes the agent. The package is a thin, well-typed shell around the
deterministic class, designed for an agent to drive across many turns.

---

## 3. Non-goals

Explicitly out of scope. Do not add these unprompted.

- **No LLM calls of any kind.** Not for explanation, not for naming, not for
  fallbacks. If a tool needs judgement, it returns the evidence and lets the
  session judge.
  *Corrected:* the original wording demanded that `pydantic-ai` and `openai` be
  absent from the dependency graph. That is not achievable — TimeCopilot 0.0.30
  is a monolith and depends on both unconditionally (see §15). The enforceable
  and equivalent property is that **this package never imports or calls them**,
  and needs no API key. Nothing under `src/tslab_mcp/` references either.
- **No hosted transport.** stdio only. The data is assumed sensitive; it never
  leaves the machine, and the server does no outbound networking except model
  weight downloads performed by TimeCopilot itself.
- **No auto-selection tool.** Do not write `tsf_auto_forecast` that picks a
  model internally. That reintroduces the opaque layer this package exists to
  remove.
- **No plotting to the response.** A `plot` helper writing a PNG to disk is
  fine; returning image bytes through the tool result is not.
- **No database, no scheduler, no web UI.**

---

## 4. Design principles

Four invariants. Every design decision below follows from one of them; if a
proposed change violates one, reject it.

### I1 — Handles, not dataframes

Series live in a process-local registry keyed by a short opaque handle. Tools
accept the handle and return summaries, aggregates, and filesystem paths. Raw
frames never cross the tool boundary.

*Why:* a single cross-validation frame is `n_series × h × n_windows × n_models`
rows. Serialising one into a tool result exhausts the session's context on the
first call and makes every subsequent turn worse. Context is the scarcest
resource in an agentic loop; treat it as the primary budget.

*Corollary:* every tool that produces bulk output writes parquet and returns the
path plus a bounded preview. Previews are capped by an explicit parameter with
a schema-enforced maximum, never by "usually it's small."

### I2 — Blocking work never touches the event loop

Cross-validating four foundation models over a thousand series is minutes of
CPU/GPU time. FastMCP tools are `async def`, so the body must be a sync closure
dispatched through `anyio.to_thread.run_sync`.

*Why:* a blocked event loop means the stdio transport stops answering, the
client's keepalive fails, and the session loses the server mid-task with a
half-finished run and no artifact.

### I3 — The environment is discovered, not assumed

TimeCopilot's foundation models are optional extras. Module paths drift between
versions. The server must therefore *probe*: import each registered model
lazily, report what actually resolved, and turn an import failure into a
structured, actionable message naming the extra to install and the cheap
alternatives available right now.

*Why:* an agent that proposes `Moirai` on a box with only the stats extras
should learn that from a tool result in one turn, not from a traceback that
kills a ten-minute run.

### I4 — The manifest is the artifact of record

Every mutation of state appends to a run log on the handle. `tsf_export_run`
serialises source path, input SHA-256, inferred frequency, every call with its
arguments and output paths, and pinned package versions.

*Why:* the session's prose is commentary. The manifest is what someone reruns
in six months to reproduce the number, and what a reviewer reads to see which
models were compared and on what basis. In a regulated setting this is the
difference between a defensible model-selection record and "the assistant
suggested ARIMA."

---

## 5. Package layout

```
tslab-mcp/
├── pyproject.toml
├── README.md
├── src/tslab_mcp/
│   ├── __init__.py          # version only
│   ├── __main__.py          # console entry point -> server.main()
│   ├── server.py            # FastMCP instance, tool registration, main()
│   ├── registry.py          # SeriesHandle, the handle store, lookup errors
│   ├── models.py            # MODEL_REGISTRY, lazy resolution, probing
│   ├── schemas.py           # all pydantic input models + ResponseFormat
│   ├── features.py          # seasonal period, trend/seasonal strength, acf1
│   ├── evaluation.py        # CV frame -> metric table, with fallback
│   ├── artifacts.py         # parquet/JSON writing, output dir, hashing
│   └── manifest.py          # run records, version pinning, export
└── tests/
    ├── conftest.py          # synthetic panel fixtures
    ├── test_registry.py
    ├── test_features.py
    ├── test_evaluation.py
    ├── test_schemas.py
    └── test_tools.py        # in-process tool calls, stats models only
```

A single-file reference implementation exists (`timecopilot_mcp.py`, ~740
lines) covering the tool surface end to end. It was used as the source of truth
for behaviour and decomposed into the layout above.

*Corrected:* the reference does not run against the installed versions. Its
`from mcp.server.fastmcp import FastMCP` import no longer resolves (§13), and
its metric aggregation averages the `cutoff` column (§13). Treat it as a
behavioural sketch, not as working code.

---

## 6. Module contracts

**`registry.py`** — `SeriesHandle` is a dataclass holding handle, source path,
the dataframe, frequency, SHA-256, load timestamp, and a `runs` list. The store
is a module-level dict. `get(handle)` raises `ValueError` naming the known
handles when lookup fails. No global mutable state beyond this dict; keep it in
one place so a future `lifespan` refactor is mechanical.

**`models.py`** — `MODEL_REGISTRY: dict[str, tuple[str, str]]` maps a friendly
name to `(module_path, attribute)`. `resolve(names) -> list[object]` imports
lazily and instantiates. `probe() -> dict` attempts every entry and partitions
into available/unavailable with the exception type as reason. Statistical models
are tagged separately so error messages can suggest a cheap substitute.

**`schemas.py`** — one pydantic v2 model per tool. `model_config` sets
`str_strip_whitespace`, `validate_assignment`, `extra="forbid"`. Every field
carries a description written *for an agent reading the schema*, with an
example value and real constraints (`ge`, `le`, `min_length`, `max_length`).
Validation lives here; tool bodies never hand-check inputs.

**`features.py`** — pure functions over a `pd.Series`, no I/O, no TimeCopilot
import. This is the most testable module in the package; give it the densest
tests. Prefer `tsfeatures` when importable, fall back to the pandas
implementations, and record which path was taken in the response.

**`evaluation.py`** — takes a CV frame and returns a per-model metric table.
Wraps `utilsforecast.evaluation.evaluate`. Must degrade rather than fail: if the
metric call raises, return a plain MAE computed directly plus the reason string,
so a ten-minute CV run is never lost to a signature change upstream.

**`artifacts.py`** — output directory resolution (respect
`TIMECOPILOT_MCP_HOME`, default `~/.timecopilot-mcp/runs`), parquet writing with
collision-free stems, streaming SHA-256.

**`server.py`** — thin. Each tool is a docstring, a `@mcp.tool` decorator with
full annotations, and a closure dispatched to a thread. Business logic belongs
in the modules above; if a tool body exceeds ~30 lines, something is in the
wrong place.

---

## 7. Tool surface

| Tool | Purpose | Returns |
|---|---|---|
| `tsf_load_series` | Read CSV/parquet, validate the `unique_id`/`ds`/`y` contract, infer freq, register a handle | JSON summary + SHA-256 |
| `tsf_describe_series` | Per-series features for model-family selection | Markdown table or JSON, row-capped |
| `tsf_list_models` | Probe importable models | `{available, unavailable}` |
| `tsf_cross_validate` | Rolling-origin comparison across models | Metric table + parquet path |
| `tsf_forecast` | Fit and forecast with intervals | Parquet path + bounded preview |
| `tsf_detect_anomalies` | Cross-validated z-score flagging | Counts, capped flag list, path |
| `tsf_export_run` | Pin the whole session to a manifest | Manifest path |

Naming: `tsf_` prefix throughout, snake_case, action-oriented. All tools are
`readOnlyHint: true` except `tsf_export_run`, which writes a file and is not
idempotent. Nothing here is destructive; there is no delete tool by design —
artifact cleanup is the user's business, not the agent's.

### Contract requirements per tool

- Docstrings are read by the agent as the tool description. Write them as
  guidance for *when and why* to call, not just what the function does. Say
  explicitly when a call is long-running.
- Return JSON for anything the session will compute on; Markdown only where a
  human reads the response directly (`tsf_describe_series`), and make it a
  parameter.
- Timestamps ISO-8601, UTC, always.
- Every bulk return path is capped and reports `truncated: true` plus the count
  omitted, so the session knows to read the parquet instead of asking again.

---

## 8. Error handling policy

Errors are prompts for the agent's next action. Every raised message must
contain: what went wrong, the observed state, and the specific next call.

```
# bad
KeyError: 'unique_id'

# good
Missing required columns ['unique_id']. Found: ['id', 'date', 'value'].
Rename to the Nixtla convention (unique_id, ds, y) before loading.
```

Frequency inference failure names the offending series and gives example offset
aliases. Unknown model names point at `tsf_list_models`. Unresolvable imports
name the extra *and* list the statistical models available immediately. Never
let a bare traceback reach the client.

---

## 9. Determinism and lineage

- Seed anything stochastic and record the seed in the run entry.
- Record `importlib.metadata.version` for `timecopilot`, `statsforecast`,
  `utilsforecast`, `pandas`, `torch`, plus Python version and platform string.
- Hash the input file, not the dataframe — cheap, and it catches the case where
  the source changed under a rerun.
- The manifest must be sufficient to reconstruct the run without reading the
  conversation. If a decision only exists in the session's prose, the design has
  failed; add it to the run record or to the `note` field.

---

## 10. Testing

Fast tests only — CI must not download model weights.

- **Fixtures:** synthetic panels in `conftest.py` covering the shapes that break
  things: a clean seasonal monthly series, an intermittent series with many
  zeros, a short series below `2 × period`, a multi-series panel with ragged
  lengths, an irregular series that must fail frequency inference.
- **`features.py`:** property-style tests — seasonal strength in `[0, 1]`, trend
  strength 1.0 on a pure ramp, `None` rather than `nan` on degenerate input.
  Every helper must return `None`, never `nan`, since `nan` is not valid JSON.
- **`schemas.py`:** rejection tests for out-of-range `h`, empty model lists,
  extra fields, and duplicate model names.
- **`test_tools.py`:** call the tool functions in-process with stats models only
  (`SeasonalNaive`, `AutoETS`). Assert on the *shape* of the JSON, not on
  forecast values.
- **Contract test:** every registered tool has a non-empty docstring, a
  pydantic input model, and complete annotations. Cheap, and it stops the tool
  surface from rotting.
- Mark anything needing weights `@pytest.mark.slow`, excluded by default.

Verify the server starts and lists tools with the MCP Inspector before calling
it done.

---

## 11. Packaging

- `pyproject.toml`, hatchling, `src/` layout, `requires-python = ">=3.10"`
  (TimeCopilot's floor).
- Core deps: `mcp`, `pandas`, `pyarrow`, `anyio`, `pydantic>=2`, `timecopilot`.
- ~~Extras mirroring TimeCopilot's own, so `uvx timecopilot-mcp[chronos]`
  works.~~ *Corrected:* TimeCopilot publishes no per-model extras — every model
  is a mandatory dependency of the base install. Its only extra is
  `distributed`, which `tslab-mcp[distributed]` mirrors. See §15.
- `[project.scripts] tslab-mcp = "tslab_mcp.__main__:main"`.
- Ruff + mypy strict on `src/`. Full type coverage; this package's whole value
  is its schemas.
- README with the `mcpServers` JSON stanza and a worked example: load → describe
  → list models → cross-validate → forecast → export.

---

## 12. Build order

1. Skeleton: `pyproject.toml`, package dirs, `server.py` with `tsf_list_models`
   only. Confirm the Inspector connects and lists one tool.
2. `registry.py` + `artifacts.py` + `tsf_load_series`. Get the handle pattern
   right before anything depends on it.
3. `features.py` + `tsf_describe_series`, with its tests. Pure functions, no
   TimeCopilot dependency — this can be built and verified in isolation.
4. `tsf_cross_validate` + `evaluation.py`. The hardest piece; expect the metric
   signatures to need adjustment (see §13).
5. `tsf_forecast`, then `tsf_detect_anomalies`.
6. `manifest.py` + `tsf_export_run`.
7. README, contract tests, Inspector pass.

Each step ends green: tests pass, server starts, Inspector lists the new tool.

---

## 13. Verified against the installed library

Checked 2026-08-18 against `timecopilot==0.0.30`, `mcp==2.0.0`, `pandas==2.3.x`,
Python 3.13.8. Every item below was run, not read.

- **The MCP SDK moved.** `mcp.server.fastmcp.FastMCP` no longer exists in
  `mcp` 2.0. The class is `MCPServer`, imported from `mcp.server`. Consequences:
  tool annotations are a typed `ToolAnnotations` model with **snake_case**
  fields (`read_only_hint`, not `readOnlyHint`), `list_tools()` is a coroutine,
  `CallToolResult.is_error` replaced `isError`, and `run()` takes
  `transport="stdio"`. A single pydantic parameter still works and nests the
  schema under `params` via `$ref`.
- **Module paths.** `TimeCopilotForecaster` is importable from `timecopilot`.
  `timecopilot.models.stats` holds more than the brief assumed: `AutoARIMA`,
  `AutoETS`, `AutoCES`, `SeasonalNaive`, `HistoricAverage`, `Theta`,
  `DynamicOptimizedTheta`, `ADIDA`, `IMAPA`, `CrostonClassic`, `ZeroModel`.
  `timecopilot.models.prophet.Prophet` confirmed.
  The foundation set is `chronos.Chronos`, `flowstate.FlowState`,
  `moirai.Moirai`, `patchtst_fm.PatchTSTFM`, `sundial.Sundial`, `t0.T0`,
  `tabpfn.TabPFN`, `tirex.TiRex`, `timesfm.TimesFM`, `timegpt.TimeGPT`,
  `toto.Toto`. There is no `granite` module. `Sundial` and `TabPFN` raise
  `ImportError("... requires Python < 3.13")` on 3.13 — which is why `probe()`
  records the exception **message**, not just its type: the message is the
  actionable part.
- **Method signatures.** All parameters are positional-or-keyword; none are
  keyword-only. `forecast(df, h, freq=None, level=None, quantiles=None,
  num_partitions=None)`; `cross_validation(df, h, freq=None, n_windows=1,
  step_size=None, level=None, quantiles=None, num_partitions=None)`;
  `detect_anomalies(df, h=None, freq=None, n_windows=None, level=99,
  num_partitions=None)`; `TimeCopilotForecaster(models, fallback_model=None,
  clean_cache=False)`.
- **The anomaly output column.** Not `anomaly`. TimeCopilot emits **one flag
  column per model**, named `"{alias}-anomaly"`, and its own code locates them
  with `col.endswith("-anomaly")`. The reference implementation's substring
  search for `"anomal"` works by accident with one model and picks an arbitrary
  column with several. The tool locates the column by suffix and reports which
  one it used.
- **`utilsforecast` metric signatures.** `mase(df, models, seasonality,
  train_df, ...)` — both extra arguments are required and positional-or-keyword.
  `functools.partial(mase, seasonality=n)` composed with
  `evaluate(..., train_df=...)` works, and `evaluate` still labels the row
  `mase`. **The real breakage was elsewhere:** `evaluate` returns
  `unique_id`, `cutoff`, `metric`, and one column per model, so the reference's
  `res.drop(columns=["unique_id"]).groupby("metric").mean()` tries to average a
  timestamp column. Aggregate over the model columns explicitly.
- **A features helper.** There is none. `get_seasonality(freq)` exists at
  `timecopilot.models.utils.forecaster`, delegating to gluonts, and follows the
  M4 convention where **daily and weekly data are non-seasonal** (`D -> 1`,
  `W -> 1`). That is right for MASE and useless for a seasonal-strength feature,
  so the two are separated: `features.py` keeps its own table (`D -> 7`,
  `W -> 52`) for the descriptive feature, `evaluation.py` prefers TimeCopilot's
  for the metric, and the response states the seasonality it used.

### Two costs that shape the design

- **`import timecopilot` takes ~34 seconds** and pulls in torch, transformers,
  lightning and prophet. Nothing may import it at module scope. This is not only
  a startup concern: statsforecast runs with `n_jobs=-1` outside a daemon
  process, and its spawned workers re-import the entry module — an eager import
  would be paid once per worker, on every call.
- **`detect_anomalies` defaults to the maximum number of windows**, refitting
  once per observation across the whole history. The tool exposes `n_windows`
  and its docstring says to set it.

## 14. Definition of done

`uvx --from . timecopilot-mcp` starts; the Inspector lists seven tools with
complete schemas; a Claude Code session can load a CSV, read the features, pick
a model on the evidence, cross-validate, forecast, and export a manifest that
reruns to the same numbers with the server stopped.

---

## 15. Packaging reality

Recorded because it contradicts §11 and will contradict it again on the next
TimeCopilot release.

**TimeCopilot 0.0.30 is a monolith.** There are no per-model extras. Every
model — chronos, moirai, timesfm, toto, tirex, prophet — is an unconditional
dependency of the base install, along with `torch`, `transformers`,
`lightning`, `gluonts`, `xgboost`, `optuna`, and (relevant to §3) `pydantic-ai`
and `openai`. Installing this package pulls 291 packages and about 2.3 GB. The
only extra TimeCopilot publishes is `distributed`, which `tslab-mcp[distributed]`
mirrors.

That makes §3's "must not appear in the dependency graph" unachievable, so the
non-goal was restated as the property that can actually be held and tested:
`tslab_mcp` never imports `pydantic_ai`, `openai` or `anthropic`, and the server
needs no API key. `TimeGPT` is the one registered model that reaches a network
service, and it is labelled as such in `tsf_list_models`.

**The pandas floor is 2.1.2, not 2.2.** On Python < 3.13 TimeCopilot depends on
`tabpfn-time-series==1.0.3`, which pins `pandas>=2.1.2,<2.2.0`; on 3.13+ that
dependency drops out and TimeCopilot itself requires `pandas>=2.2.0`. Declaring
`pandas>=2.2.0` makes the project unresolvable on 3.10-3.12. The seasonal-period
table therefore accepts both alias generations (`M` and `ME`, `H` and `h`).

**Python 3.13 is the recommended target**: it loses only `TabPFN` and `Sundial`
and gains pandas 2.2+.

**stdout is already protected.** The `mcp` 2.0 stdio transport serves the wire
from a private `F_DUPFD_CLOEXEC` duplicate of fd 1 and points fd 1 itself at
stderr for the lifetime of the server. Stray prints from the forecasting stack,
and from the worker processes statsforecast spawns, cannot corrupt the JSON-RPC
framing. No hardening was needed in this package.
