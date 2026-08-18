"""Synthetic fixtures covering the panel shapes that break things."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from tslab_mcp import artifacts, registry


def pytest_collection_modifyitems(config, items):
    """Skip, rather than fail, the tests that need the optional extra.

    TimeCopilot is no longer a core dependency, so a base install legitimately
    cannot run these. Skipping keeps `pytest -m slow` meaningful in both
    environments instead of erroring in one of them.
    """
    if importlib.util.find_spec("timecopilot") is not None:
        return
    skip = pytest.mark.skip(reason="needs the 'foundation' extra (timecopilot)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Every test gets an empty handle store and its own artifact directory."""
    monkeypatch.setenv(artifacts.HOME_ENV_VAR, str(tmp_path / "home"))
    registry.clear()
    yield
    registry.clear()


def _panel(uid: str, periods: int, freq: str = "MS", **kwargs) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": uid,
            "ds": pd.date_range("2018-01-01", periods=periods, freq=freq),
            "y": kwargs["y"],
        }
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20240818)


@pytest.fixture
def seasonal_monthly(rng) -> pd.DataFrame:
    """A clean monthly series: trend plus a strong 12-period cycle."""
    n = 72
    season = 12 * np.sin(2 * np.pi * np.arange(n) / 12)
    y = 100 + 0.6 * np.arange(n) + season + rng.normal(0, 1.5, n)
    return _panel("clean", n, y=y)


@pytest.fixture
def intermittent(rng) -> pd.DataFrame:
    """Mostly zeros, occasional demand -- the Croston/ADIDA shape."""
    n = 72
    y = np.where(rng.random(n) < 0.25, rng.integers(1, 9, n), 0).astype(float)
    return _panel("intermittent", n, y=y)


@pytest.fixture
def too_short(rng) -> pd.DataFrame:
    """Shorter than 2 x seasonal period: seasonal strength must be None."""
    n = 8
    return _panel("short", n, y=100 + rng.normal(0, 1, n))


@pytest.fixture
def constant() -> pd.DataFrame:
    """Zero variance: every strength/correlation feature must be None."""
    n = 36
    return _panel("flat", n, y=np.full(n, 5.0))


@pytest.fixture
def ragged_panel(seasonal_monthly, intermittent, too_short) -> pd.DataFrame:
    """Multiple series of different lengths in one frame."""
    return pd.concat([seasonal_monthly, intermittent, too_short], ignore_index=True)


@pytest.fixture
def irregular() -> pd.DataFrame:
    """Deliberately unequal spacing: frequency inference must fail."""
    return pd.DataFrame(
        {
            "unique_id": "irregular",
            "ds": pd.to_datetime(
                ["2020-01-01", "2020-01-03", "2020-02-17", "2020-02-19", "2020-07-01"]
            ),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


@pytest.fixture
def csv_path(tmp_path, seasonal_monthly) -> str:
    path = tmp_path / "clean.csv"
    seasonal_monthly.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def panel_path(tmp_path, ragged_panel) -> str:
    path = tmp_path / "panel.parquet"
    ragged_panel.to_parquet(path, index=False)
    return str(path)


@pytest.fixture
def loaded(csv_path):
    """A registered single-series handle."""
    series, _ = registry.load_panel(csv_path, handle="clean")
    return series
