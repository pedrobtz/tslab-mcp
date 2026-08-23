"""The external agent receives a stable end-to-end workflow contract."""

from __future__ import annotations

from tslab_mcp.agent_instructions import AGENT_INSTRUCTIONS
from tslab_mcp.server import mcp


def test_server_exposes_the_canonical_agent_instructions():
    assert mcp.instructions == AGENT_INSTRUCTIONS
    assert len(AGENT_INSTRUCTIONS) > 4_000


def test_complete_workflow_tools_appear_in_execution_order():
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
    positions = [AGENT_INSTRUCTIONS.index(tool) for tool in tools]
    assert positions == sorted(positions)


def test_selection_policy_requires_evidence_and_a_baseline():
    required = [
        "SeasonalNaive",
        "MASE as the primary",
        "Do not select a model from feature intuition alone",
        "If no challenger qualifies",
        "same selected model",
        "never claim that a more complex model is better",
    ]
    for phrase in required:
        assert phrase in AGENT_INSTRUCTIONS
