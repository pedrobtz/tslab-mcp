# TSLab GitHub Copilot agent bundle

The [`bundle`](bundle) directory is a copy-ready GitHub Copilot integration for
another repository. It contains:

```text
bundle/
├── .github/agents/tslab-forecaster.agent.md
└── .vscode/mcp.json
```

The agent profile defines the evidence-first forecasting workflow and restricts
its tools to read/search plus `tslab/*`. The MCP configuration runs the current
GitHub version of `tslab-mcp` with `uvx`, so the target repository does not need
to contain this package's source.

## Copy into a repository

If the target has neither of these files already, copy the bundle contents from
the `tslab-mcp` checkout:

```bash
cp -R integrations/github-copilot/bundle/. /path/to/target-repository/
```

Then open the target repository in VS Code, run **MCP: List Servers**, start
`tslab`, and select **TSLab Forecaster** in Copilot Chat.

## If the target already has Copilot configuration

Do not overwrite an existing `.vscode/mcp.json`. Instead:

1. Copy only
   `bundle/.github/agents/tslab-forecaster.agent.md` into the target's
   `.github/agents/` directory.
2. Merge the `tslab` object from `bundle/.vscode/mcp.json` into the existing
   top-level `servers` object.

If the target already has an agent with the same filename, compare and rename
one profile before copying it.

For installation, verification, the AirPassengers prompt, and troubleshooting,
see the full [GitHub Copilot guide](../../docs/github-copilot.md).
