"""Loading, validation, and the handle store."""

from __future__ import annotations

import pandas as pd
import pytest

from tslab_mcp import registry


def test_load_csv_registers_a_handle(csv_path):
    series, replaced = registry.load_panel(csv_path, handle="clean")
    assert replaced is False
    assert registry.get("clean") is series
    assert series.freq == "MS"
    assert series.n_series == 1
    assert len(series.sha256) == 64


def test_load_parquet_panel(panel_path):
    series, _ = registry.load_panel(panel_path)
    assert series.n_series == 3
    assert series.handle in registry.handles()


def test_summary_is_json_safe_and_complete(loaded):
    summary = registry.summarize(loaded)
    expected = {
        "handle", "source", "sha256", "freq", "n_series", "n_obs", "start",
        "end", "obs_per_series", "n_missing_y", "extra_columns", "loaded_at",
        "replaced_existing_handle",
    }
    assert expected <= set(summary)
    assert summary["n_obs"] == 72


def test_unknown_handle_names_the_known_ones(loaded):
    with pytest.raises(ValueError, match="clean"):
        registry.get("nope")


def test_unknown_handle_on_empty_store_says_none():
    with pytest.raises(ValueError, match="none"):
        registry.get("nope")


def test_reusing_a_handle_reports_the_replacement(csv_path):
    registry.load_panel(csv_path, handle="dup")
    _, replaced = registry.load_panel(csv_path, handle="dup")
    assert replaced is True


def test_missing_columns_error_names_what_was_found(tmp_path):
    path = tmp_path / "wrong.csv"
    pd.DataFrame({"id": ["a"], "date": ["2020-01-01"], "value": [1.0]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError) as excinfo:
        registry.load_panel(str(path))
    message = str(excinfo.value)
    assert "unique_id" in message
    assert "['id', 'date', 'value']" in message


def test_missing_file_error_is_actionable(tmp_path):
    with pytest.raises(ValueError, match="No such file"):
        registry.load_panel(str(tmp_path / "absent.csv"))


def test_unsupported_extension_lists_the_supported_ones(tmp_path):
    path = tmp_path / "data.xlsx"
    path.write_text("not really excel")
    with pytest.raises(ValueError, match=r"\.parquet"):
        registry.load_panel(str(path))


def test_duplicate_timestamps_are_rejected(tmp_path, seasonal_monthly):
    path = tmp_path / "dupes.csv"
    pd.concat([seasonal_monthly, seasonal_monthly.head(3)]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        registry.load_panel(str(path))


def test_empty_frame_is_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    pd.DataFrame(columns=["unique_id", "ds", "y"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no rows"):
        registry.load_panel(str(path))


def test_irregular_spacing_fails_with_an_example_alias(tmp_path, irregular):
    path = tmp_path / "irregular.csv"
    irregular.to_csv(path, index=False)
    with pytest.raises(ValueError) as excinfo:
        registry.load_panel(str(path))
    message = str(excinfo.value)
    assert "irregular" in message
    assert "'MS'" in message


def test_explicit_freq_overrides_failed_inference(tmp_path, irregular):
    path = tmp_path / "irregular.csv"
    irregular.to_csv(path, index=False)
    series, _ = registry.load_panel(str(path), freq="D")
    assert series.freq == "D"


def test_frequency_inferred_from_the_longest_series(panel_path):
    series, _ = registry.load_panel(panel_path)
    assert series.freq == "MS"


def test_rows_are_sorted_by_id_then_time(panel_path):
    series, _ = registry.load_panel(panel_path)
    df = series.df
    assert df.equals(df.sort_values(["unique_id", "ds"]).reset_index(drop=True))
    assert pd.api.types.is_datetime64_any_dtype(df["ds"])


def test_clear_empties_the_store(loaded):
    registry.clear()
    assert registry.handles() == []
