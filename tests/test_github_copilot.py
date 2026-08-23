"""The checked-in Copilot agent and MCP configuration stay usable together."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_vscode_mcp_config_starts_this_checkout_portably():
    config = json.loads((ROOT / ".vscode" / "mcp.json").read_text())
    server = config["servers"]["tslab"]

    assert server["type"] == "stdio"
    assert server["command"] == "uv"
    assert server["args"] == [
        "run",
        "--directory",
        "${workspaceFolder}",
        "tslab-mcp",
    ]
    assert server["env"]["TSLAB_MCP_HOME"] == "${userHome}/.tslab-mcp"


def test_copilot_agent_is_vscode_scoped_to_tslab():
    profile = (
        ROOT / ".github" / "agents" / "tslab-forecaster.agent.md"
    ).read_text()

    assert profile.startswith("---\n")
    assert "target: vscode" in profile
    assert '  - "tslab/*"' in profile
    assert "  - execute" not in profile
    assert "Treat the workflow instructions" in profile


def test_copilot_agent_preserves_the_complete_tool_order():
    profile = (
        ROOT / ".github" / "agents" / "tslab-forecaster.agent.md"
    ).read_text()
    tools = [
        "tsf_load_series",
        "tsf_describe_series",
        "tsf_list_models",
        "tsf_cross_validate",
        "tsf_select_model",
        "tsf_forecast",
        "tsf_detect_anomalies",
        "tsf_export_run",
        "tsf_export_report",
    ]

    positions = [profile.index(tool) for tool in tools]
    assert positions == sorted(positions)


def test_copyable_bundle_keeps_the_canonical_agent_profile():
    canonical = (
        ROOT / ".github" / "agents" / "tslab-forecaster.agent.md"
    ).read_text()
    bundled = (
        ROOT
        / "integrations"
        / "github-copilot"
        / "bundle"
        / ".github"
        / "agents"
        / "tslab-forecaster.agent.md"
    ).read_text()

    assert bundled == canonical


def test_copyable_bundle_runs_the_server_without_a_checkout():
    path = (
        ROOT
        / "integrations"
        / "github-copilot"
        / "bundle"
        / ".vscode"
        / "mcp.json"
    )
    server = json.loads(path.read_text())["servers"]["tslab"]

    assert server["command"] == "uvx"
    assert server["args"] == [
        "--from",
        "git+https://github.com/pedrobtz/tslab-mcp",
        "tslab-mcp",
    ]
