"""Metric aggregation, including the degradation path.

A cross-validation run can cost minutes, so the contract is that evaluation
never raises: a broken metric call must still return a usable table plus the
reason it degraded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tslab_mcp import evaluation


@pytest.fixture
def train(rng) -> pd.DataFrame:
    dates = pd.date_range("2018-01-01", periods=60, freq="MS")
    return pd.DataFrame(
        {"unique_id": "a", "ds": dates, "y": 100 + rng.normal(0, 5, 60)}
    )


@pytest.fixture
def cv_frame(train, rng) -> pd.DataFrame:
    """A realistic CV frame: cutoff column, actuals, two model columns."""
    tail = train.tail(12).reset_index(drop=True)
    return pd.DataFrame(
        {
            "unique_id": "a",
            "ds": tail["ds"],
            "cutoff": train["ds"].iloc[47],
            "y": tail["y"],
            "SeasonalNaive": tail["y"] + rng.normal(0, 3, 12),
            "AutoETS": tail["y"] + rng.normal(0, 1, 12),
        }
    )


def test_metric_table_has_one_entry_per_metric(cv_frame, train):
    result = evaluation.evaluate_cv(
        cv_frame, train, ["SeasonalNaive", "AutoETS"], ["mase", "smape"], "MS"
    )
    assert set(result["metrics"]) == {"mase", "smape"}
    assert "degraded_reason" not in result


def test_cutoff_column_is_not_averaged(cv_frame, train):
    """Regression guard: ``evaluate`` carries ``cutoff`` through, and averaging
    a timestamp either explodes or produces nonsense."""
    result = evaluation.evaluate_cv(
        cv_frame, train, ["SeasonalNaive", "AutoETS"], ["mae"], "MS"
    )
    assert set(result["metrics"]["mae"]) == {"SeasonalNaive", "AutoETS"}


def test_scores_are_plain_floats(cv_frame, train):
    result = evaluation.evaluate_cv(cv_frame, train, ["AutoETS"], ["mae"], "MS")
    value = result["metrics"]["mae"]["AutoETS"]
    assert isinstance(value, float)
    assert value > 0


def test_ranking_is_best_first(cv_frame, train):
    result = evaluation.evaluate_cv(
        cv_frame, train, ["SeasonalNaive", "AutoETS"], ["mae"], "MS"
    )
    # AutoETS was built with a third of SeasonalNaive's error.
    assert result["ranking"]["mae"][0] == "AutoETS"


def test_missing_model_columns_are_reported_not_fatal(cv_frame, train):
    result = evaluation.evaluate_cv(
        cv_frame, train, ["SeasonalNaive", "Chronos"], ["mae"], "MS"
    )
    assert result["models_evaluated"] == ["SeasonalNaive"]
    assert result["models_without_predictions"] == ["Chronos"]


def test_no_usable_columns_degrades_with_a_reason(cv_frame, train):
    result = evaluation.evaluate_cv(cv_frame, train, ["Chronos"], ["mae"], "MS")
    assert result["metrics"] == {}
    assert "Chronos" not in result["models_evaluated"]
    assert "degraded_reason" in result


def test_upstream_failure_falls_back_to_mae(cv_frame, train, monkeypatch):
    def _explode(*args, **kwargs):
        raise TypeError("mase() got an unexpected keyword argument")

    monkeypatch.setattr(evaluation, "_metric_functions", _explode)
    result = evaluation.evaluate_cv(
        cv_frame, train, ["SeasonalNaive", "AutoETS"], ["mase"], "MS"
    )
    assert "degraded_reason" in result
    assert "unexpected keyword" in result["degraded_reason"]
    assert set(result["metrics"]["mae"]) == {"SeasonalNaive", "AutoETS"}


def test_unsupported_metric_degrades_rather_than_raising(cv_frame, train):
    result = evaluation.evaluate_cv(cv_frame, train, ["AutoETS"], ["bogus"], "MS")
    assert "degraded_reason" in result
    assert "mae" in result["metrics"]


def test_seasonality_falls_back_without_timecopilot_imported(monkeypatch):
    monkeypatch.delitem(__import__("sys").modules, "timecopilot", raising=False)
    assert evaluation.evaluation_seasonality("MS") == 12
    assert evaluation.evaluation_seasonality("D") == 7


def test_nan_scores_are_dropped_from_the_table(train, rng):
    tail = train.tail(6).reset_index(drop=True)
    frame = pd.DataFrame(
        {
            "unique_id": "a",
            "ds": tail["ds"],
            "cutoff": train["ds"].iloc[53],
            "y": tail["y"],
            "AutoETS": np.nan,
        }
    )
    result = evaluation.evaluate_cv(frame, train, ["AutoETS"], ["mae"], "MS")
    assert "AutoETS" not in result["metrics"].get("mae", {})
