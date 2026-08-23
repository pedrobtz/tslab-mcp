# Use TSLab with a GitHub Copilot custom agent

This repository includes everything needed to run a dedicated forecasting
agent in GitHub Copilot Chat for VS Code:

- [`.github/agents/tslab-forecaster.agent.md`](../.github/agents/tslab-forecaster.agent.md)
  defines the agent's role, tool permissions, and evidence-first workflow.
- [`.vscode/mcp.json`](../.vscode/mcp.json) starts this checkout as the `tslab`
  MCP server.
- [`src/tslab_mcp/agent_workflow.md`](../src/tslab_mcp/agent_workflow.md) is the
  full runtime contract sent to every MCP client.
- [`examples/air_passengers.csv`](../examples/air_passengers.csv) provides a
  small end-to-end test dataset.

The MCP configuration makes the tools available. The custom-agent profile tells
Copilot how and when to use them. Both pieces are required for the guided
experience.

## Quickstart from this checkout

### 1. Install the prerequisites

Install:

- Git and [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.10 or newer
- VS Code with GitHub Copilot Chat and custom agents enabled

Clone and prepare the project:

```bash
git clone https://github.com/pedrobtz/tslab-mcp
cd tslab-mcp
uv sync
code .
```

The base installation is sufficient for the example and does not install
TimeCopilot, PyTorch, or foundation-model weights.

### 2. Start the MCP server

The committed [`.vscode/mcp.json`](../.vscode/mcp.json) already contains the
portable configuration for this checkout.

1. Open the VS Code Command Palette.
2. Run **MCP: List Servers**.
3. Select **tslab** and start it if it is not already running.
4. Use **Show Output** from the same menu if startup fails.

The first start may take longer while `uv` prepares the environment. VS Code
must be able to find `uv` on its `PATH`.

### 3. Select the custom agent

1. Open GitHub Copilot Chat.
2. Open the agent selector at the bottom of the chat view.
3. Choose **TSLab Forecaster**.
4. Open the tool picker and confirm that all nine `tsf_*` tools from **tslab**
   are enabled.

If the agent appears but its tools do not, confirm that the MCP server ID is
exactly `tslab`. The profile deliberately grants `tslab/*`, not every MCP tool
installed in VS Code.

### 4. Run the AirPassengers example

Get the dataset's absolute path from the repository root:

```bash
uv run python -c "from pathlib import Path; print(Path('examples/air_passengers.csv').resolve())"
```

Paste this into the **TSLab Forecaster** chat, replacing `<ABSOLUTE_PATH>`:

```text
Analyze <ABSOLUTE_PATH> end to end. Use handle `air_passengers`, forecast the
next 12 monthly periods, compare suitable statistical models against
SeasonalNaive using rolling-origin cross-validation and MASE, validate the
selection, generate 80% and 95% prediction intervals, check historical
anomalies, and export the manifest and Markdown report. Do not use foundation
models. Explain the winning-versus-baseline evidence and return every artifact
path.
```

The agent should load and describe the 144 monthly observations, discover the
installed models, cross-validate candidates, pass its proposal through
`tsf_select_model`, forecast with that validated model, detect anomalies, and
export the reproducible record. Scores can vary with forecasting-library
versions, so judge the run by that sequence rather than by a hard-coded winner.

For prompts that expose each stage individually, use the longer
[`examples/air-passengers-chat.md`](../examples/air-passengers-chat.md)
walkthrough.

## Use the agent in another repository

The copy-ready [`integrations/github-copilot`](../integrations/github-copilot)
bundle contains both required files with their correct destination paths. If
the target repository does not already have either file, run this from the
`tslab-mcp` checkout:

```bash
cp -R integrations/github-copilot/bundle/. /path/to/target-repository/
```

The bundled server entry uses `uvx` rather than this checkout's local `uv run`
command:

```json
{
  "servers": {
    "tslab": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/pedrobtz/tslab-mcp",
        "tslab-mcp"
      ],
      "env": {
        "TSLAB_MCP_HOME": "${userHome}/.tslab-mcp"
      }
    }
  }
}
```

If `.vscode/mcp.json` already exists in the target, do not overwrite it. Copy
only the agent profile and merge the bundled `tslab` entry into the existing
top-level `servers` object. The bundle's own
[`README.md`](../integrations/github-copilot/README.md) gives the two safe copy
paths.

Pin the Git URL to a tag or commit for reproducible team use. After the package
is published, the command can be shortened to `uvx tslab-mcp`.

The supplied profile targets VS Code because this stdio server is designed to
read local data and retain an in-memory handle during the session. GitHub's
cloud coding agent runs in an ephemeral environment and cannot see files that
exist only on your workstation.

## Common problems

- **No TSLab Forecaster in the selector:** verify that the profile is under
  `.github/agents/` and reload the VS Code window.
- **No `tsf_*` tools:** switch to the custom agent, start `tslab` with **MCP:
  List Servers**, and check its output log.
- **`uv` command not found:** start VS Code from a shell where `uv` works, or
  make `uv` available on the graphical application's `PATH`.
- **File not found:** pass an absolute CSV or Parquet path. The server process
  cannot resolve paths relative to the chat message.
- **Unknown handle after a restart:** handles live in the MCP server process;
  load the data again after restarting it.
- **Unexpected large downloads:** keep the model family statistical. Foundation
  models require the optional `foundation` installation and may download model
  weights.
