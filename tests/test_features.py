"""Property-style tests for the feature helpers.

These functions decide which model family an agent proposes, and they feed a
JSON response, so the two things that matter are that the numbers stay inside
their defined ranges and that degenerate input yields ``None`` rather than
``nan``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from tslab_mcp.features import (
    acf1,
    panel_features,
    seasonal_period,
    seasonal_strength,
    series_features,
    trend_strength,
)


@pytest.mark.parametrize(
    ("freq", "expected"),
    [
        ("MS", 12), ("M", 12), ("ME", 12),
        ("QS", 4), ("Q", 4),
        ("YS", 1), ("Y", 1), ("A", 1),
        ("D", 7), ("B", 5),
        ("W-MON", 52), ("W", 52),
        ("H", 24), ("h", 24),
        ("min", 1440),
        ("", 1), ("nonsense", 1),
    ],
)
def test_seasonal_period_covers_both_alias_generations(freq, expected):
    assert seasonal_period(freq) == expected


def test_trend_strength_is_one_on_a_pure_ramp():
    assert trend_strength(pd.Series(np.arange(50, dtype=float))) == 1.0


def test_trend_strength_is_near_zero_on_noise(rng):
    value = trend_strength(pd.Series(rng.normal(0, 1, 500)))
    assert value is not None
    assert value < 0.1


def test_trend_strength_none_when_too_short():
    assert trend_strength(pd.Series([1.0, 2.0])) is None


def test_trend_strength_none_on_constant():
    assert trend_strength(pd.Series([3.0] * 20)) is None


def test_seasonal_strength_in_unit_interval(seasonal_monthly):
    value = seasonal_strength(seasonal_monthly["y"], 12)
    assert value is not None
    assert 0.0 <= value <= 1.0


def test_seasonal_strength_high_on_a_pure_cycle():
    y = pd.Series(np.tile([0.0, 5.0, 10.0, 5.0], 20))
    value = seasonal_strength(y, 4)
    assert value is not None
    assert value > 0.9


def test_seasonal_strength_none_below_two_full_cycles():
    assert seasonal_strength(pd.Series(np.arange(11, dtype=float)), 12) is None


def test_seasonal_strength_none_when_period_is_one():
    assert seasonal_strength(pd.Series(np.arange(50, dtype=float)), 1) is None


def test_seasonal_strength_none_on_zero_variance():
    assert seasonal_strength(pd.Series([2.0] * 48), 12) is None


def test_acf1_none_on_constant_and_short():
    assert acf1(pd.Series([1.0] * 30)) is None
    assert acf1(pd.Series([1.0, 2.0])) is None


def test_acf1_within_bounds(rng):
    value = acf1(pd.Series(rng.normal(0, 1, 300)))
    assert value is not None
    assert -1.0 <= value <= 1.0


def test_series_features_are_json_serialisable_and_never_nan(constant):
    row = series_features("flat", constant["y"], constant["ds"], 12)
    encoded = json.dumps(row)
    assert "NaN" not in encoded
    for key, value in row.items():
        if isinstance(value, float):
            assert not math.isnan(value), key


def test_short_series_features_stay_json_clean(too_short):
    row = series_features("short", too_short["y"], too_short["ds"], 12)
    assert row["seasonal_strength"] is None
    assert "NaN" not in json.dumps(row)


def test_panel_features_one_row_per_series_sorted(ragged_panel):
    rows = panel_features(ragged_panel, 12)
    ids = [row["unique_id"] for row in rows]
    assert ids == sorted(ids)
    assert set(ids) == set(ragged_panel["unique_id"].unique())


def test_panel_features_report_true_lengths(ragged_panel):
    rows = {row["unique_id"]: row["n"] for row in panel_features(ragged_panel, 12)}
    expected = ragged_panel.groupby("unique_id").size().to_dict()
    assert rows == expected


def test_intermittent_series_shows_its_zeros(intermittent):
    row = series_features("i", intermittent["y"], intermittent["ds"], 12)
    assert row["pct_zero"] > 0.5
