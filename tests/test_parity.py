"""Do the two backends produce the same numbers?

The package's whole claim is reproducibility, and the statsforecast backend is a
reimplementation of the path TimeCopilot used to own. If those disagree, a
manifest exported from a base install does not reproduce on an install with the
``foundation`` extra — which would make the artifact of record a fiction.

TimeCopilot's statistical models delegate to the same statsforecast classes, so
these should agree exactly. Marked slow: it needs the extra, and importing
TimeCopilot costs ~30 seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tslab_mcp.backends import StatsForecastBackend, TimeCopilotBackend
from tslab_mcp.evaluation import evaluate_cv

pytestmark = pytest.mark.slow

FREQ = "MS"
H = 3


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    """One short, strongly seasonal series: enough to expose a period mismatch,
    small enough that TimeCopilot's leg of the comparison stays quick."""
    rng = np.random.default_rng(4)
    n = 36
    ds = pd.date_range("2020-01-01", periods=n, freq=FREQ)
    y = 200 + 2 * np.arange(n) + 40 * np.sin(2 * np.pi * np.arange(n) / 12)
    y = y + rng.normal(0, 3, n)
    return pd.DataFrame({"unique_id": "a", "ds": ds, "y": y})


@pytest.fixture(scope="module")
def backends() -> tuple[StatsForecastBackend, TimeCopilotBackend]:
    names = ["SeasonalNaive"]
    return StatsForecastBackend(names), TimeCopilotBackend(names)


def test_forecast_point_values_agree(panel, backends):
    stats, timecopilot = backends
    a = stats.forecast(panel, H, FREQ, None).sort_values("ds").reset_index(drop=True)
    b = (
        timecopilot.forecast(panel, H, FREQ, None)
        .sort_values("ds")
        .reset_index(drop=True)
    )
    assert len(a) == len(b) == H
    np.testing.assert_allclose(
        a["SeasonalNaive"].to_numpy(float),
        b["SeasonalNaive"].to_numpy(float),
        rtol=0,
        atol=1e-9,
    )


def test_prediction_intervals_agree(panel, backends):
    stats, timecopilot = backends
    a = stats.forecast(panel, H, FREQ, [80]).sort_values("ds").reset_index(drop=True)
    b = (
        timecopilot.forecast(panel, H, FREQ, [80])
        .sort_values("ds")
        .reset_index(drop=True)
    )
    for column in ("SeasonalNaive-lo-80", "SeasonalNaive-hi-80"):
        assert column in a.columns and column in b.columns
        np.testing.assert_allclose(
            a[column].to_numpy(float), b[column].to_numpy(float), rtol=0, atol=1e-9
        )


def test_cross_validation_metrics_agree(panel, backends):
    """The number an agent actually reads when choosing a model."""
    stats, timecopilot = backends
    cv_a = stats.cross_validation(panel, H, FREQ, 2, None)
    cv_b = timecopilot.cross_validation(panel, H, FREQ, 2, None)

    metrics = ["mase", "smape"]
    a = evaluate_cv(cv_a, panel, ["SeasonalNaive"], metrics, FREQ)
    b = evaluate_cv(cv_b, panel, ["SeasonalNaive"], metrics, FREQ)

    assert a["seasonality_used_for_mase"] == b["seasonality_used_for_mase"] == 12
    for metric in metrics:
        assert a["metrics"][metric]["SeasonalNaive"] == pytest.approx(
            b["metrics"][metric]["SeasonalNaive"], abs=1e-9
        )


def test_anomaly_output_shape_agrees(panel, backends):
    """The reimplemented detector must keep the column name the report reads."""
    stats, timecopilot = backends
    a = stats.detect_anomalies(panel, 1, FREQ, 3, 99)
    b = timecopilot.detect_anomalies(panel, 1, FREQ, 3, 99)

    flag = "SeasonalNaive-anomaly"
    assert flag in a.columns, list(a.columns)
    assert flag in b.columns, list(b.columns)
    assert a[flag].dtype == bool
    assert len(a) == len(b)
