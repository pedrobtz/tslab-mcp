"""Backend contract tests across every advertised statistical model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tslab_mcp import backends, evaluation
from tslab_mcp.models import STATS_MODELS


def test_every_statistical_model_forecasts_with_default_intervals(
    seasonal_monthly,
):
    names = list(STATS_MODELS)
    result = backends.StatsForecastBackend(names).forecast(
        seasonal_monthly, h=2, freq="MS", level=[80, 95]
    )
    backends.validate_forecast_frame(result, names, [80, 95])
    for name in names:
        assert name in result
        for level in (80, 95):
            assert f"{name}-lo-{level}" in result
            assert f"{name}-hi-{level}" in result


def test_every_statistical_model_cross_validates_under_its_public_name(
    seasonal_monthly,
):
    names = list(STATS_MODELS)
    result = backends.StatsForecastBackend(names).cross_validation(
        seasonal_monthly, h=2, freq="MS", n_windows=2, step_size=None
    )
    backends.validate_prediction_frame(result, names, "Cross-validation")
    scores = evaluation.evaluate_cv(
        result, seasonal_monthly, names, ["mae"], "MS"
    )
    assert set(scores["metrics"]["mae"]) == set(names)
    assert "models_without_predictions" not in scores


def test_every_statistical_model_emits_a_canonical_anomaly_flag(
    seasonal_monthly,
):
    names = list(STATS_MODELS)
    result = backends.StatsForecastBackend(names).detect_anomalies(
        seasonal_monthly, h=1, freq="MS", n_windows=2, level=90
    )
    backends.validate_prediction_frame(result, names, "Anomaly detection")
    for name in names:
        assert f"{name}-anomaly" in result


@pytest.mark.parametrize("name", ["ADIDA", "IMAPA", "CrostonClassic"])
def test_intermittent_models_record_conformal_interval_configuration(name):
    config = backends.prediction_interval_configuration(
        "statsforecast", [name], h=6, levels=[80, 95]
    )
    assert config is not None
    assert config["conformal_models"] == [name]
    assert config["conformal"] == {
        "method": "conformal_distribution",
        "n_windows": 2,
        "h": 6,
    }


def test_history_preflight_rejects_short_seasonal_series(too_short):
    with pytest.raises(ValueError, match="SeasonalNaive needs at least 12"):
        backends.validate_history(
            too_short,
            ["SeasonalNaive"],
            freq="MS",
            h=3,
            operation="forecast",
        )


def test_history_preflight_accounts_for_cv_windows(seasonal_monthly):
    short = seasonal_monthly.head(15)
    with pytest.raises(ValueError, match="needs at least 18"):
        backends.validate_history(
            short,
            ["SeasonalNaive"],
            freq="MS",
            h=2,
            operation="cross-validation",
            n_windows=3,
            step_size=2,
        )


def test_default_anomaly_windows_reserve_model_training_history(seasonal_monthly):
    windows = backends.resolve_anomaly_windows(
        seasonal_monthly,
        ["SeasonalNaive"],
        freq="MS",
        h=1,
        requested=None,
    )
    assert windows == len(seasonal_monthly) - 12


@pytest.mark.parametrize(
    ("name", "minimum"),
    [
        ("HistoricAverage", 2),
        ("AutoARIMA", 4),
        ("AutoETS", 7),
        ("AutoCES", 5),
        ("Theta", 4),
        ("DynamicOptimizedTheta", 4),
        ("SeasonalNaive", 12),
    ],
)
def test_model_specific_training_minima(name, minimum):
    assert backends.minimum_training_length(name, "MS", h=1) == minimum


@pytest.mark.filterwarnings("ignore:divide by zero encountered in scalar divide")
@pytest.mark.filterwarnings(
    "ignore:Too few residuals after burn-in for sigma estimate"
)
def test_default_anomaly_windows_run_for_every_statistical_model():
    n = 18
    values = 20 + 0.3 * np.arange(n) + 2 * np.sin(2 * np.pi * np.arange(n) / 12)
    frame = pd.DataFrame(
        {
            "unique_id": "a",
            "ds": pd.date_range("2020-01-01", periods=n, freq="MS"),
            "y": values,
        }
    )
    for name in STATS_MODELS:
        result = backends.StatsForecastBackend([name]).detect_anomalies(
            frame, h=1, freq="MS", n_windows=None, level=90
        )
        backends.validate_anomaly_frame(result, name, 90)


def test_non_finite_predictions_are_rejected():
    result = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": pd.date_range("2026-01-01", periods=2, freq="D"),
            "Model": [1.0, np.nan],
        }
    )
    with pytest.raises(ValueError, match="non-finite"):
        backends.validate_prediction_frame(result, ["Model"], "Forecast")


def test_missing_structural_columns_are_rejected():
    result = pd.DataFrame({"Model": [1.0]})
    with pytest.raises(ValueError, match="structural column"):
        backends.validate_prediction_frame(result, ["Model"], "Forecast")


def test_missing_requested_intervals_are_rejected():
    result = pd.DataFrame(
        {
            "unique_id": ["a"],
            "ds": pd.to_datetime(["2026-01-01"]),
            "Model": [1.0],
        }
    )
    with pytest.raises(ValueError, match="interval column"):
        backends.validate_forecast_frame(result, ["Model"], [80])


def test_non_finite_anomaly_intervals_are_rejected():
    result = pd.DataFrame(
        {
            "unique_id": ["a"],
            "ds": pd.to_datetime(["2026-01-01"]),
            "cutoff": pd.to_datetime(["2025-12-31"]),
            "y": [1.0],
            "Model": [1.0],
            "Model-lo-90": [np.nan],
            "Model-hi-90": [2.0],
            "Model-anomaly": [False],
        }
    )
    with pytest.raises(ValueError, match="non-finite prediction intervals"):
        backends.validate_anomaly_frame(result, "Model", 90)


def test_anomaly_flag_must_be_boolean():
    result = pd.DataFrame(
        {
            "unique_id": ["a"],
            "ds": pd.to_datetime(["2026-01-01"]),
            "cutoff": pd.to_datetime(["2025-12-31"]),
            "y": [1.0],
            "Model": [1.0],
            "Model-lo-90": [0.0],
            "Model-hi-90": [2.0],
            "Model-anomaly": [0],
        }
    )
    with pytest.raises(ValueError, match="non-boolean"):
        backends.validate_anomaly_frame(result, "Model", 90)
