"""Model-resolution contracts that do not require the optional extra."""

from __future__ import annotations

import sys
from types import ModuleType

from tslab_mcp import models


def test_timecopilot_statistical_wrapper_receives_public_alias(monkeypatch):
    timecopilot = ModuleType("timecopilot")
    tc_models = ModuleType("timecopilot.models")
    tc_stats = ModuleType("timecopilot.models.stats")

    class FakeAutoCES:
        def __init__(self, alias=None):
            self.alias = alias

    tc_stats.AutoCES = FakeAutoCES  # type: ignore[attr-defined]
    tc_models.stats = tc_stats  # type: ignore[attr-defined]
    timecopilot.models = tc_models  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "timecopilot", timecopilot)
    monkeypatch.setitem(sys.modules, "timecopilot.models", tc_models)
    monkeypatch.setitem(sys.modules, "timecopilot.models.stats", tc_stats)

    resolved = models.resolve_timecopilot(["AutoCES"])

    assert len(resolved) == 1
    assert resolved[0].alias == "AutoCES"
