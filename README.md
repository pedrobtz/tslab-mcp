# tslab-mcp

An MCP server that exposes deterministic time-series forecasting as tools, so
your agent is the reasoning engine and every number comes from ordinary,
reproducible Python.

No LLM is called anywhere in this package. No API key is required (unless you
ask for `TimeGPT`, which calls the Nixtla API).

## Why

Some forecasting libraries ship an agent that reads features, picks a model,
and explains the result with an LLM in the loop. Calling one of those from your
own agent nests an agent inside an agent — two prompts, two bills, two sources
of nondeterminism, and an opaque middle layer that makes the model-selection
rationale unauditable.

So control is inverted here: the forecasting library is the tool, and your
agent is the one reasoning. It reads the features, argues for a model family,
cross-validates the candidates, and writes the rationale into a manifest. Every
number on the way is produced by a library call you can rerun without an LLM in
the path.

That split carries into how the package itself is built. The base install runs
eleven statistical models — `AutoARIMA`, `AutoETS`, `Theta`, `CrostonClassic`
and friends — through [statsforecast](https://github.com/Nixtla/statsforecast):
roughly 340 MB, no PyTorch, and it starts in seconds. An optional `foundation`
extra adds [TimeCopilot](https://timecopilot.dev)'s pretrained models —
Chronos, Moirai, TimesFM, TiRex, Toto and others — plus Prophet, for when a
statistical baseline isn't enough. A request that only names statistical models
never imports TimeCopilot or torch; a request that names even one foundation
model runs entirely through TimeCopilot, which carries the statistical models
too. Either way `tsf_list_models` reports what's actually installed before you
commit to a model.

## Install

Requires Python 3.10+ (3.13 recommended, see [Python version](#python-version)).

```bash
uvx tslab-mcp                     # run without installing
uv tool install tslab-mcp         # or install the CLI
```

The base install runs the eleven statistical models through
[statsforecast](https://github.com/Nixtla/statsforecast): roughly 340 MB, no
PyTorch, and it starts instantly. For the pretrained foundation models — Chronos,
Moirai, TimesFM, Toto, TiRex — and Prophet, add the extra:

```bash
uvx --from 'tslab-mcp[foundation]' tslab-mcp
```

> The `foundation` extra pulls TimeCopilot, which brings torch, transformers and
> lightning: roughly 2 GB on first install, and the first tool call that touches
> it spends ~30 seconds importing. Both are one-off, and neither is paid unless
> you ask for a model that needs them.

### From GitHub

`uv` and `uvx` both accept a git URL in place of a package name, which installs
the current `main` without waiting for a release
(see [submit-pypi.md](submit-pypi.md) for the PyPI status):

```bash
uvx --from git+https://github.com/pedrobtz/tslab-mcp tslab-mcp
uv tool install git+https://github.com/pedrobtz/tslab-mcp        # or install the CLI

# with the foundation extra
uvx --from 'tslab-mcp[foundation] @ git+https://github.com/pedrobtz/tslab-mcp' tslab-mcp
```

Pin a ref for anything other than casual testing — the branch head can move
under you otherwise. A commit works today; a version tag will too once one is
cut:

```bash
uv tool install "git+https://github.com/pedrobtz/tslab-mcp@136824c1cc2a"
```

### From a checkout

```bash
git clone https://github.com/pedrobtz/tslab-mcp
cd tslab-mcp
uv sync                              # base
uv sync --extra foundation           # with the pretrained models
uv run tslab-mcp
```

## Configure

Add the server to your MCP client's configuration. The file differs per client —
often `.mcp.json` in the project root — but the entry itself is the same shape:

```json
{
  "mcpServers": {
    "tslab": {
      "command": "uvx",
      "args": ["tslab-mcp"],
      "env": {
        "TSLAB_MCP_HOME": "~/.tslab-mcp"
      }
    }
  }
}
```

`TSLAB_MCP_HOME` sets where artifacts are written; it defaults to
`~/.tslab-mcp`, and run outputs land in `<home>/runs`.

Transport is stdio only, by design: your data is assumed sensitive and never
leaves the machine. The server makes no outbound requests except the model
weight downloads TimeCopilot itself performs for foundation models, and the
Nixtla API calls `TimeGPT` makes if you ask for it specifically.

### GitHub Copilot

Copilot discovers MCP servers from an `mcp.json` file and exposes their tools in
**agent mode** — the tools do not appear in ask or edit mode.

**VS Code.** Put the server in `.vscode/mcp.json` to share it with the repo, or
run **MCP: Open User Configuration** from the Command Palette to keep it in your
own profile across every workspace. Note the key is `servers`, not `mcpServers`:

```json
{
  "servers": {
    "tslab": {
      "type": "stdio",
      "command": "uvx",
      "args": ["tslab-mcp"],
      "env": {
        "TSLAB_MCP_HOME": "${userHome}/.tslab-mcp"
      }
    }
  }
}
```

From a checkout, point it at the working tree instead:

```json
{
  "servers": {
    "tslab": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "${workspaceFolder}", "tslab-mcp"]
    }
  }
}
```

Then: open Chat, switch the mode selector to **Agent**, and use the **Tools**
button to confirm the eight `tsf_*` tools are listed and enabled. `MCP: List
Servers` shows the server's status and its logs, which is where a failed start
is explained. Copilot caps how many tools can be active at once, so if you run
several MCP servers you may need to deselect some to fit all eight.

**Visual Studio.** Same JSON shape, in `.mcp.json` at the solution root (or
`%USERPROFILE%\.mcp.json` for all solutions), then enable the tools from the
Copilot Chat agent-mode tool picker.

**JetBrains, Eclipse, and Xcode.** Open the Copilot Chat agent-mode tool picker,
choose **Edit MCP configuration**, and add the same `servers` entry to the
`mcp.json` it opens.

> **Copilot coding agent** (the cloud agent on github.com) is a poor fit for this
> server: it runs your MCP servers inside an ephemeral GitHub Actions
> environment, which means paying the ~2 GB TimeCopilot install on every run, and
> it has no access to local data files. Use it from your editor instead.

## Tools

| Tool | Purpose | Returns |
|---|---|---|
| `tsf_load_series` | Read CSV/Parquet, validate the `unique_id`/`ds`/`y` contract, infer frequency, register a handle | JSON summary + SHA-256 |
| `tsf_describe_series` | Per-series features for choosing a model family | Markdown table or JSON, row-capped |
| `tsf_list_models` | Probe which models actually import here | `{available, statistical, foundation, unavailable}` |
| `tsf_cross_validate` | Rolling-origin comparison across models | Metric table, ranking, parquet path |
| `tsf_forecast` | Fit and forecast with prediction intervals | Parquet path + bounded preview |
| `tsf_detect_anomalies` | Cross-validated interval flagging | Counts, capped flag list, parquet path |
| `tsf_export_run` | Pin the session to a re-runnable manifest | Manifest path |
| `tsf_export_report` | Render every step as a readable report | HTML or Markdown path |

Everything except the two `tsf_export_*` tools is marked read-only; nothing here
deletes, so cleaning up `~/.tslab-mcp/runs` is your business, not the agent's.

## Starting a session

The tools do not enforce an order, so the opening prompt is what turns eight
callable functions into an analysis. Something like this works well:

> Use the tslab tools to forecast the series in
> `/Users/me/data/deposits.csv`, 12 months ahead.
>
> Work in this order and show your reasoning at each step:
>
> 1. Load the file and tell me what you found — how many series, what frequency,
>    any gaps or missing values.
> 2. Describe the features, and say which model families they argue for, and why.
> 3. Check which models are actually installed before proposing any.
> 4. Cross-validate your shortlist against a SeasonalNaive baseline over 4
>    windows. Statistical models only for now.
> 5. Forecast with the winner, with 80% and 95% intervals.
> 6. Export a run manifest and an HTML report, and put the model-selection
>    rationale in the note: what you chose, what the metric table showed, and
>    what you rejected.
>
> Summarise results and give me the parquet paths — don't paste whole frames
> into the chat.

Four things in that prompt are doing real work:

- **An absolute path.** Relative paths resolve against the *server's* working
  directory, which your MCP client chooses and you generally cannot predict.
- **A horizon that matches the decision.** `h` drives both the forecast and how
  much history each CV window consumes; 12 monthly steps is a year of planning,
  not an arbitrary default.
- **"Statistical models only for now."** Without it, an agent may reach for a
  foundation model and spend several minutes downloading weights to answer a
  question `AutoETS` would have settled in seconds. Lift the restriction once
  the cheap models have set a floor.
- **Asking for the rationale in the manifest note.** The chat transcript is
  disposable; the manifest is the part someone can rerun and audit. If the
  reasoning only exists in the conversation, it is effectively lost.

Shorter openers, when you know what you want:

> Load `/Users/me/data/sales.parquet` and describe the features. Don't forecast
> yet — I want to see what we're dealing with first.

> Compare SeasonalNaive, AutoETS and AutoARIMA on the loaded `deposits` handle,
> over 6 windows at h=12, then tell me whether anything beats the baseline by
> enough to be worth the extra complexity.

Statistical-only calls answer in seconds. The first call that names a
foundation model spends ~30 seconds importing TimeCopilot before it does
anything else — that pause is expected, not a hang, and it only happens if the
`foundation` extra is installed and a request actually reaches for one.

## A worked session

Start from a CSV in Nixtla long format:

```csv
unique_id,ds,y
branch_01,2018-01-01,1043.2
branch_01,2018-02-01,1102.7
...
```

**1. Load it.** The panel stays in the server process; the handle is all the
session carries.

```json
{"handle": "deposits", "n_series": 12, "n_obs": 864, "freq": "MS",
 "start": "2018-01-01T00:00:00", "end": "2023-12-01T00:00:00",
 "obs_per_series": {"min": 72, "median": 72, "max": 72},
 "n_missing_y": 0, "sha256": "9f2c…"}
```

**2. Describe it.** These are the numbers you reason over.

```
| id        | n  | mean   | cv    | %zero | trend | seasonal | acf1(diff) |
|-----------|----|--------|-------|-------|-------|----------|------------|
| branch_01 | 72 | 1180.4 | 0.112 | 0.0   | 0.83  | 0.62     | -0.31      |
```

High seasonal strength and a clear trend argue for `AutoETS` and `AutoARIMA`
over a naive baseline; a high `%zero` would have argued for `ADIDA` or
`CrostonClassic` instead.

`seasonal` is an STL strength — the seasonal component measured against what
remains once the trend is removed — so a growing series still reports its
seasonality honestly. It carries a noise floor of roughly 0.3–0.5: scores in
that band mean "no evidence", not "mildly seasonal".

**3. Check what is installed** with `tsf_list_models`, so you never propose a
model this machine cannot run.

**4. Cross-validate the candidates** — always including `SeasonalNaive`, since a
model that cannot beat it is not worth deploying:

```json
{"kind": "cross_validation", "models": ["SeasonalNaive", "AutoETS", "AutoARIMA"],
 "h": 12, "n_windows": 4, "seasonality_used_for_mase": 12,
 "metrics": {"mase": {"SeasonalNaive": 1.0, "AutoETS": 0.71, "AutoARIMA": 0.68}},
 "ranking": {"mase": ["AutoARIMA", "AutoETS", "SeasonalNaive"]},
 "artifact": "~/.tslab-mcp/runs/cv_deposits_3f1a9c02.parquet"}
```

**5. Forecast** with the winner. The full frame goes to parquet; the response
carries the path, the columns, and a short preview.

**6. Export the run and the report.** Write down *why*, in the note — it is the
only part of your reasoning that outlives the conversation:

```json
{"manifest": "~/.tslab-mcp/runs/manifest_deposits_77b0e415.json", "n_runs": 3,
 "kinds": ["cross_validation", "forecast"]}
```

The manifest holds the source path and hash, the frequency, every call with its
arguments and artifact paths, the pinned versions of whatever's actually
installed — statsforecast, pandas and Python always; TimeCopilot and torch too
if the `foundation` extra is in — and your note. It is sufficient to reproduce
the numbers with the server stopped.

`tsf_export_report` turns that same manifest into something a person reads —
features, metric tables ordered best-first, forecasts, anomalies and the
environment, in the order they happened:

```json
{"report": "~/.tslab-mcp/runs/report_deposits_5c31d0a7.html",
 "format": "html", "n_steps": 3,
 "steps": ["features", "cross_validation", "forecast"]}
```

The report is a *pure function of the manifest*: it reads no parquet and calls
no model, so `tsf_export_report` with `manifest_path` re-renders a run from
months ago with nothing loaded. The HTML embeds its own CSS and references no
external script, stylesheet or font, so it still opens correctly offline.

## Design

Four invariants, and the reasons they exist:

**Handles, not dataframes.** One cross-validation frame is
`n_series × h × n_windows × n_models` rows. Serialising it into a tool result
exhausts the session's context on the first call and makes every later turn
worse. Tools take a handle and return summaries, aggregates, and file paths;
every bulk path is capped and reports what it omitted, so the session knows to
read the parquet rather than ask again.

**Blocking work never touches the event loop.** Cross-validating several models
over a large panel is minutes of CPU. Every tool body is a synchronous closure
dispatched through `anyio.to_thread.run_sync`, so the stdio transport keeps
answering and the client does not drop the server mid-run.

**The environment is discovered, not assumed.** Models are imported lazily and
probed, never assumed present. `tsf_list_models` reports what actually resolved
here, so asking for `Chronos` without the extra returns a message naming the
extra rather than a traceback ten minutes into a run.

The backend is chosen by what you ask for: a request whose models are all
statistical runs through statsforecast, and only a request that needs a
pretrained model reaches for TimeCopilot. Statistical runs therefore never
import torch, and the server starts instantly either way.

statsforecast is left at its default `n_jobs=1` deliberately. Its parallel mode
spawns worker processes that re-import the entry module, which inside an MCP
server buys contention and a stdout hazard rather than speed.

**The manifest is the artifact of record.** Prose in the conversation is
commentary. The manifest is what someone reruns in six months, and what a
reviewer reads to see which models were compared and on what basis.

## Python version

TimeCopilot gates several models on the interpreter version, and on Python < 3.13
it pins `tabpfn-time-series`, which caps pandas below 2.2.

| Python | Models | pandas |
|---|---|---|
| 3.13 | everything except `TabPFN` and `Sundial` | ≥ 2.2 |
| 3.10–3.12 | adds `TabPFN`, `Sundial` | < 2.2 |

3.13 is the recommended target. Either way, `tsf_list_models` reports what
actually resolved, with the reason for anything that did not.

## Development

```bash
uv sync --all-groups
uv run pytest                  # fast suite
uv run pytest -m slow          # exercises TimeCopilot; slower, no weight downloads
uv run ruff check src tests
uv run mypy
```

Inspect the tool surface with the MCP Inspector:

```bash
npx @modelcontextprotocol/inspector uv run tslab-mcp
```

## License

MIT
