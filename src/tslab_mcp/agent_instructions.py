"""Canonical instructions sent to the external MCP agent."""

from __future__ import annotations

from importlib.resources import files

AGENT_INSTRUCTIONS = (
    files("tslab_mcp")
    .joinpath("agent_workflow.md")
    .read_text(encoding="utf-8")
    .strip()
)
