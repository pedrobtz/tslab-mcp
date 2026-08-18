"""Smoke tests, so the CI pipeline has something to run before the real code lands."""

import tslab_mcp


def test_version_is_exposed() -> None:
    assert tslab_mcp.__version__
