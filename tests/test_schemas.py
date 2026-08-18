"""Rejection tests for the tool input models.

The schemas are the package's guard rail: every constraint here is one an agent
would otherwise discover as a traceback ten minutes into a run.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tslab_mcp.schemas import (
    CrossValidateInput,
    DescribeSeriesInput,
    DetectAnomaliesInput,
    ExportRunInput,
    ForecastInput,
    ListModelsInput,
    LoadSeriesInput,
    Metric,
    ResponseFormat,
)

ALL_INPUTS = [
    LoadSeriesInput,
    DescribeSeriesInput,
    ListModelsInput,
    CrossValidateInput,
    ForecastInput,
    DetectAnomaliesInput,
    ExportRunInput,
]


@pytest.mark.parametrize("model", ALL_INPUTS)
def test_extra_fields_are_forbidden(model):
    """A misspelled argument must fail loudly, not be silently dropped."""
    fields = {
        "path": "/tmp/x.csv", "handle": "h", "models": ["AutoETS"],
        "model": "AutoETS", "h": 4,
    }
    kwargs = {k: v for k, v in fields.items() if k in model.model_fields}
    with pytest.raises(ValidationError, match="typo_field"):
        model(**kwargs, typo_field=1)


@pytest.mark.parametrize("model", ALL_INPUTS)
def test_every_field_documents_itself(model):
    """Descriptions are what the agent reads; none may be missing."""
    for name, field in model.model_fields.items():
        assert field.description, f"{model.__name__}.{name} has no description"


def test_whitespace_is_stripped():
    assert LoadSeriesInput(path="  /tmp/a.csv  ").path == "/tmp/a.csv"


def test_empty_path_rejected():
    with pytest.raises(ValidationError):
        LoadSeriesInput(path="")


def test_load_defaults_leave_inference_to_the_server():
    params = LoadSeriesInput(path="/tmp/a.csv")
    assert params.freq is None
    assert params.handle is None


def test_describe_defaults_to_markdown():
    params = DescribeSeriesInput(handle="h")
    assert params.response_format is ResponseFormat.MARKDOWN
    assert params.max_series == 20


@pytest.mark.parametrize("value", [0, 201])
def test_max_series_is_bounded(value):
    with pytest.raises(ValidationError):
        DescribeSeriesInput(handle="h", max_series=value)


@pytest.mark.parametrize("value", [0, 5001])
def test_horizon_is_bounded(value):
    with pytest.raises(ValidationError):
        CrossValidateInput(handle="h", models=["AutoETS"], h=value)


def test_empty_model_list_rejected():
    with pytest.raises(ValidationError):
        CrossValidateInput(handle="h", models=[], h=4)


def test_too_many_models_rejected():
    with pytest.raises(ValidationError):
        CrossValidateInput(handle="h", models=[f"m{i}" for i in range(13)], h=4)


def test_duplicate_models_are_collapsed_in_order():
    params = CrossValidateInput(
        handle="h", models=["AutoETS", "SeasonalNaive", "AutoETS"], h=4
    )
    assert params.models == ["AutoETS", "SeasonalNaive"]


def test_duplicate_metrics_are_collapsed():
    params = CrossValidateInput(
        handle="h", models=["AutoETS"], h=4, metrics=["mae", "mae", "mase"]
    )
    assert params.metrics == [Metric.MAE, Metric.MASE]


def test_unknown_metric_rejected():
    with pytest.raises(ValidationError):
        CrossValidateInput(handle="h", models=["AutoETS"], h=4, metrics=["r2"])


def test_cross_validate_defaults():
    params = CrossValidateInput(handle="h", models=["AutoETS"], h=4)
    assert params.n_windows == 5
    assert params.step_size is None
    assert params.metrics == [Metric.MASE, Metric.SMAPE]


@pytest.mark.parametrize("value", [0, 51])
def test_n_windows_is_bounded(value):
    with pytest.raises(ValidationError):
        CrossValidateInput(handle="h", models=["AutoETS"], h=4, n_windows=value)


@pytest.mark.parametrize("level", [[0], [100], [-5], [80, 150]])
def test_interval_levels_must_be_percentages(level):
    with pytest.raises(ValidationError, match="out of range"):
        ForecastInput(handle="h", models=["AutoETS"], h=4, level=level)


def test_levels_are_sorted_and_deduped():
    params = ForecastInput(handle="h", models=["AutoETS"], h=4, level=[95, 80, 95])
    assert params.level == [80, 95]


def test_empty_level_list_is_allowed_for_point_forecasts():
    assert ForecastInput(handle="h", models=["AutoETS"], h=4, level=[]).level == []


@pytest.mark.parametrize("value", [-1, 101])
def test_preview_rows_are_bounded(value):
    with pytest.raises(ValidationError):
        ForecastInput(handle="h", models=["AutoETS"], h=4, max_preview_rows=value)


@pytest.mark.parametrize("value", [49, 100])
def test_anomaly_level_is_bounded(value):
    with pytest.raises(ValidationError):
        DetectAnomaliesInput(handle="h", model="AutoETS", level=value)


def test_anomaly_defaults():
    params = DetectAnomaliesInput(handle="h", model="SeasonalNaive")
    assert (params.h, params.level, params.max_flagged) == (1, 99, 100)


def test_anomaly_takes_a_single_model_not_a_list():
    with pytest.raises(ValidationError):
        DetectAnomaliesInput(handle="h", model=["AutoETS"])


def test_export_note_is_optional_but_bounded():
    assert ExportRunInput(handle="h").note is None
    with pytest.raises(ValidationError):
        ExportRunInput(handle="h", note="x" * 4001)


def test_list_models_defaults_to_the_full_probe():
    params = ListModelsInput()
    assert params.family.value == "all"
    assert params.include_unavailable is True


def test_list_models_rejects_an_unknown_family():
    with pytest.raises(ValidationError):
        ListModelsInput(family="neural")
