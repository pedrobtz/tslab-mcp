"""In-process tool calls.

Tools are exercised through the same functions the server registers, asserting
on the *shape* of the response rather than on forecast values -- the numbers
belong to TimeCopilot and are not this package's contract.

Anything that imports TimeCopilot is marked ``slow``: the import alone costs
about thirty seconds. The default suite runs without it.
"""

from __future__ import annotations

import inspect
import json
import typing
from pathlib import Path

import pytest
from pydantic import BaseModel

from tslab_mcp import artifacts, registry
from tslab_mcp.schemas import (
    CrossValidateInput,
    DescribeSeriesInput,
    DetectAnomaliesInput,
    ExportRunInput,
    ForecastInput,
    ListModelsInput,
    LoadSeriesInput,
    ResponseFormat,
)
from tslab_mcp.server import (
    mcp,
    tsf_cross_validate,
    tsf_describe_series,
    tsf_detect_anomalies,
    tsf_export_run,
    tsf_forecast,
    tsf_list_models,
    tsf_load_series,
)

TOOL_FUNCTIONS = [
    tsf_load_series,
    tsf_describe_series,
    tsf_list_models,
    tsf_cross_validate,
    tsf_forecast,
    tsf_detect_anomalies,
    tsf_export_run,
]

EXPECTED_TOOL_NAMES = {
    "tsf_load_series",
    "tsf_describe_series",
    "tsf_list_models",
    "tsf_cross_validate",
    "tsf_forecast",
    "tsf_detect_anomalies",
    "tsf_export_run",
}


# --------------------------------------------------------------------------
# contract: cheap, and it stops the tool surface from rotting
# --------------------------------------------------------------------------


async def test_all_seven_tools_are_registered():
    names = {tool.name for tool in await mcp.list_tools()}
    assert names == EXPECTED_TOOL_NAMES


async def test_every_tool_describes_itself_for_an_agent():
    for tool in await mcp.list_tools():
        assert tool.description, tool.name
        assert len(tool.description) > 120, f"{tool.name} description is too thin"


async def test_every_tool_declares_annotations():
    for tool in await mcp.list_tools():
        assert tool.annotations is not None, tool.name
        assert tool.annotations.title, tool.name
        assert tool.annotations.open_world_hint is False, tool.name


async def test_only_export_run_is_declared_writing():
    writing = {
        tool.name
        for tool in await mcp.list_tools()
        if tool.annotations and tool.annotations.read_only_hint is False
    }
    assert writing == {"tsf_export_run"}


async def test_every_tool_exposes_a_pydantic_schema():
    for tool in await mcp.list_tools():
        schema = tool.input_schema
        assert schema["type"] == "object"
        assert "params" in schema["properties"], tool.name
        assert schema["$defs"], tool.name


@pytest.mark.parametrize("fn", TOOL_FUNCTIONS, ids=lambda f: f.__name__)
def test_tool_signatures_are_complete(fn):
    """One pydantic input model in, a string out, fully annotated."""
    assert inspect.iscoroutinefunction(fn)
    hints = typing.get_type_hints(fn)
    assert hints["return"] is str
    parameters = [p for name, p in inspect.signature(fn).parameters.items()]
    assert len(parameters) == 1
    model = hints[parameters[0].name]
    assert issubclass(model, BaseModel)
    assert model.model_config["extra"] == "forbid"


# --------------------------------------------------------------------------
# tools that need no forecasting backend
# --------------------------------------------------------------------------


async def test_load_returns_a_json_summary(csv_path):
    payload = json.loads(await tsf_load_series(LoadSeriesInput(path=csv_path)))
    assert payload["n_series"] == 1
    assert payload["n_obs"] == 72
    assert payload["freq"] == "MS"
    assert len(payload["sha256"]) == 64
    assert payload["handle"] in registry.handles()


async def test_load_honours_an_explicit_handle(csv_path):
    payload = json.loads(
        await tsf_load_series(LoadSeriesInput(path=csv_path, handle="deposits"))
    )
    assert payload["handle"] == "deposits"


async def test_load_error_is_actionable(tmp_path):
    with pytest.raises(ValueError, match="No such file"):
        await tsf_load_series(LoadSeriesInput(path=str(tmp_path / "nope.csv")))


async def test_describe_markdown_is_a_table(loaded):
    text = await tsf_describe_series(DescribeSeriesInput(handle="clean"))
    assert text.startswith("**clean**")
    assert "| id | n |" in text
    assert "seasonal period 12" in text


async def test_describe_json_shape(loaded):
    payload = json.loads(
        await tsf_describe_series(
            DescribeSeriesInput(handle="clean", response_format=ResponseFormat.JSON)
        )
    )
    assert payload["truncated"] is False
    assert payload["n_series"] == 1
    row = payload["features"][0]
    assert row["n"] == 72
    assert 0.0 <= row["seasonal_strength"] <= 1.0


async def test_describe_caps_rows_and_says_so(panel_path):
    registry.load_panel(panel_path, handle="panel")
    payload = json.loads(
        await tsf_describe_series(
            DescribeSeriesInput(
                handle="panel", max_series=1, response_format=ResponseFormat.JSON
            )
        )
    )
    assert payload["truncated"] is True
    assert payload["n_omitted"] == 2
    assert len(payload["features"]) == 1


async def test_describe_markdown_notes_the_omission(panel_path):
    registry.load_panel(panel_path, handle="panel")
    text = await tsf_describe_series(
        DescribeSeriesInput(handle="panel", max_series=1)
    )
    assert "further series omitted" in text


async def test_describe_unknown_handle_lists_the_known_ones(loaded):
    with pytest.raises(ValueError, match="clean"):
        await tsf_describe_series(DescribeSeriesInput(handle="ghost"))


async def test_export_writes_a_manifest_with_a_pinned_environment(loaded):
    payload = json.loads(
        await tsf_export_run(ExportRunInput(handle="clean", note="nothing run yet"))
    )
    path = Path(payload["manifest"])
    assert path.exists()
    assert path.parent == artifacts.output_dir()

    manifest = json.loads(path.read_text())
    assert manifest["manifest_version"] == 1
    assert manifest["note"] == "nothing run yet"
    assert manifest["input"]["sha256"] == loaded.sha256
    assert manifest["input"]["freq"] == "MS"
    assert manifest["environment"]["tslab_mcp"] == "0.1.0"
    assert "python" in manifest["environment"]


async def test_export_is_repeatable_without_collision(loaded):
    first = json.loads(await tsf_export_run(ExportRunInput(handle="clean")))
    second = json.loads(await tsf_export_run(ExportRunInput(handle="clean")))
    assert first["manifest"] != second["manifest"]
    assert Path(first["manifest"]).exists()


# --------------------------------------------------------------------------
# tools that drive TimeCopilot
# --------------------------------------------------------------------------


@pytest.mark.slow
async def test_list_models_finds_the_statistical_family():
    payload = json.loads(await tsf_list_models(ListModelsInput()))
    assert "SeasonalNaive" in payload["available"]
    assert "AutoETS" in payload["statistical"]
    assert payload["n_available"] >= 10
    assert "unavailable" in payload


@pytest.mark.slow
async def test_list_models_can_filter_to_one_family():
    payload = json.loads(
        await tsf_list_models(ListModelsInput(family="statistical"))
    )
    assert payload["foundation"] == []


@pytest.mark.slow
async def test_list_models_reports_reasons_not_just_types():
    payload = json.loads(await tsf_list_models(ListModelsInput()))
    for reason in payload["unavailable"].values():
        assert ":" in reason


@pytest.mark.slow
async def test_cross_validate_returns_a_metric_table_and_artifact(loaded):
    payload = json.loads(
        await tsf_cross_validate(
            CrossValidateInput(
                handle="clean",
                models=["SeasonalNaive", "AutoETS"],
                h=6,
                n_windows=2,
                metrics=["mase", "mae"],
            )
        )
    )
    assert payload["kind"] == "cross_validation"
    assert set(payload["metrics"]) == {"mase", "mae"}
    assert set(payload["metrics"]["mae"]) == {"SeasonalNaive", "AutoETS"}
    assert payload["ranking"]["mae"]
    assert Path(payload["artifact"]).exists()
    assert "degraded_reason" not in payload
    assert len(registry.get("clean").runs) == 1


@pytest.mark.slow
async def test_forecast_writes_parquet_and_previews(loaded):
    payload = json.loads(
        await tsf_forecast(
            ForecastInput(
                handle="clean", models=["SeasonalNaive"], h=6, max_preview_rows=3
            )
        )
    )
    assert payload["kind"] == "forecast"
    assert payload["n_rows"] == 6
    assert len(payload["preview"]) == 3
    assert payload["truncated"] is True
    assert "SeasonalNaive" in payload["columns"]
    assert any("lo-95" in column for column in payload["columns"])
    assert Path(payload["artifact"]).exists()


@pytest.mark.slow
async def test_forecast_preview_can_be_switched_off(loaded):
    payload = json.loads(
        await tsf_forecast(
            ForecastInput(
                handle="clean", models=["SeasonalNaive"], h=4, max_preview_rows=0
            )
        )
    )
    assert payload["preview"] == []


@pytest.mark.slow
async def test_detect_anomalies_counts_flags(loaded):
    payload = json.loads(
        await tsf_detect_anomalies(
            DetectAnomaliesInput(
                handle="clean", model="SeasonalNaive", n_windows=4, level=99
            )
        )
    )
    assert payload["kind"] == "anomalies"
    assert payload["note"] is None
    assert isinstance(payload["n_flagged"], int)
    assert payload["n_flagged"] <= payload["n_rows_tested"]
    assert Path(payload["artifact"]).exists()


@pytest.mark.slow
async def test_unknown_model_points_at_the_probe(loaded):
    with pytest.raises(ValueError, match="tsf_list_models"):
        await tsf_forecast(
            ForecastInput(handle="clean", models=["NotAModel"], h=3)
        )


@pytest.mark.slow
async def test_manifest_captures_the_whole_session(loaded):
    await tsf_cross_validate(
        CrossValidateInput(
            handle="clean", models=["SeasonalNaive"], h=6, n_windows=2, metrics=["mae"]
        )
    )
    await tsf_forecast(ForecastInput(handle="clean", models=["SeasonalNaive"], h=6))
    payload = json.loads(
        await tsf_export_run(ExportRunInput(handle="clean", note="picked the baseline"))
    )
    assert payload["n_runs"] == 2
    assert payload["kinds"] == ["cross_validation", "forecast"]

    manifest = json.loads(Path(payload["manifest"]).read_text())
    kinds = [run["kind"] for run in manifest["runs"]]
    assert kinds == ["cross_validation", "forecast"]
    assert manifest["environment"]["timecopilot"]
    for run in manifest["runs"]:
        assert Path(run["artifact"]).exists()
        assert run["at"].endswith("+00:00")
