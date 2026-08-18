"""Report rendering.

The renderers are pure functions of a manifest dict, so they are tested against
hand-built manifests rather than a live run — including the shapes a real
session produces but a happy-path fixture would miss: a degraded metric table,
an empty run log, and missing optional keys.
"""

from __future__ import annotations

import json

import pytest

from tslab_mcp import report


@pytest.fixture
def manifest() -> dict:
    return {
        "manifest_version": 1,
        "created_at": "2026-08-18T12:00:00+00:00",
        "input": {
            "source": "/data/deposits.csv",
            "sha256": "9f2c" + "0" * 60,
            "freq": "MS",
            "loaded_at": "2026-08-18T11:59:00+00:00",
            "n_series": 3,
            "n_obs": 216,
        },
        "runs": [
            {
                "kind": "features",
                "at": "2026-08-18T11:59:10+00:00",
                "seasonal_period": 12,
                "n_series": 3,
                "shown": 3,
                "truncated": False,
                "n_omitted": 0,
                "features": [
                    {
                        "unique_id": "branch_01", "n": 72, "mean": 1207.42,
                        "sd": 137.39, "cv": 0.1138, "pct_zero": 0.0,
                        "trend_strength": 0.7673, "seasonal_strength": 0.1356,
                        "acf1_diff": 0.202,
                    },
                    {
                        "unique_id": "branch_02", "n": 72, "mean": 2210.33,
                        "sd": 132.04, "cv": 0.0597, "pct_zero": 0.0,
                        "trend_strength": 0.7589, "seasonal_strength": None,
                        "acf1_diff": 0.08,
                    },
                ],
            },
            {
                "kind": "cross_validation",
                "at": "2026-08-18T12:00:00+00:00",
                "models": ["SeasonalNaive", "AutoETS", "AutoARIMA"],
                "h": 12,
                "n_windows": 3,
                "metrics_requested": ["mase", "smape"],
                "artifact": "/runs/cv_deposits_a1.parquet",
                "seasonality_used_for_mase": 12,
                "metrics": {
                    "mase": {
                        "SeasonalNaive": 1.0427, "AutoETS": 0.3325,
                        "AutoARIMA": 0.3424,
                    },
                    "smape": {
                        "SeasonalNaive": 0.0189, "AutoETS": 0.0059,
                        "AutoARIMA": 0.0061,
                    },
                },
                "ranking": {
                    "mase": ["AutoETS", "AutoARIMA", "SeasonalNaive"],
                    "smape": ["AutoETS", "AutoARIMA", "SeasonalNaive"],
                },
            },
            {
                "kind": "forecast",
                "at": "2026-08-18T12:01:00+00:00",
                "models": ["AutoETS"],
                "h": 12,
                "level": [80, 95],
                "artifact": "/runs/fcst_deposits_b2.parquet",
                "n_rows": 36,
                "preview": [
                    {"unique_id": "branch_01", "ds": "2024-01-01", "AutoETS": 1436.06},
                ],
            },
        ],
        "environment": {"python": "3.13.8", "tslab_mcp": "0.1.0", "pandas": "2.3.3"},
        "note": "Chose AutoETS: lowest MASE, beating the baseline.",
    }


# --------------------------------------------------------------------------
# shared content guarantees
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["markdown", "html"])
def test_every_step_appears(manifest, fmt):
    out = report.render(manifest, fmt)
    for expected in ("Features", "Cross-validation", "Forecast"):
        assert expected in out


@pytest.mark.parametrize("fmt", ["markdown", "html"])
def test_rationale_is_carried_through(manifest, fmt):
    assert "lowest MASE" in report.render(manifest, fmt)


@pytest.mark.parametrize("fmt", ["markdown", "html"])
def test_lineage_is_present(manifest, fmt):
    out = report.render(manifest, fmt)
    assert "/data/deposits.csv" in out
    assert manifest["input"]["sha256"] in out
    assert "3.13.8" in out


@pytest.mark.parametrize("fmt", ["markdown", "html"])
def test_metric_values_are_rendered(manifest, fmt):
    out = report.render(manifest, fmt)
    assert "1.0427" in out
    assert "0.3325" in out


@pytest.mark.parametrize("fmt", ["markdown", "html"])
def test_artifact_paths_survive(manifest, fmt):
    out = report.render(manifest, fmt)
    assert "cv_deposits_a1.parquet" in out
    assert "fcst_deposits_b2.parquet" in out


@pytest.mark.parametrize("fmt", ["markdown", "html"])
def test_empty_run_log_still_renders(manifest, fmt):
    manifest["runs"] = []
    out = report.render(manifest, fmt)
    assert "/data/deposits.csv" in out


@pytest.mark.parametrize("fmt", ["markdown", "html"])
def test_missing_optional_keys_do_not_break(fmt):
    assert report.render({}, fmt)


@pytest.mark.parametrize("fmt", ["markdown", "html"])
def test_degraded_reason_is_surfaced(manifest, fmt):
    manifest["runs"][1]["degraded_reason"] = "TypeError: bad signature"
    out = report.render(manifest, fmt)
    assert "bad signature" in out


def test_unknown_format_is_rejected(manifest):
    with pytest.raises(ValueError, match="html"):
        report.render(manifest, "pdf")


# --------------------------------------------------------------------------
# format-specific
# --------------------------------------------------------------------------


def test_models_are_ordered_best_first(manifest):
    """The metric table leads with the winner, not with the requested order."""
    headers, rows = report._metrics_table(manifest["runs"][1])
    assert headers == ["Model", "mase", "smape"]
    assert [row[0] for row in rows] == ["AutoETS", "AutoARIMA", "SeasonalNaive"]


def test_none_features_render_as_a_marker_not_none(manifest):
    text = report.render_markdown(manifest)
    assert "None" not in text
    assert report.EMPTY in text


def test_markdown_tables_are_well_formed(manifest):
    for line in report.render_markdown(manifest).splitlines():
        if line.startswith("|") and "---" not in line:
            assert line.endswith("|")


def test_html_is_self_contained(manifest):
    html = report.render_html(manifest)
    assert html.startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")
    assert "<style>" in html
    # No external asset may be referenced: the file must open years later.
    for marker in ("http://", "https://", "<script", "src="):
        assert marker not in html


def test_html_escapes_injected_markup(manifest):
    manifest["note"] = "<script>alert('x')</script> & <b>bold</b>"
    html = report.render_html(manifest)
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_html_escapes_values_inside_tables(manifest):
    manifest["input"]["source"] = "/data/<img onerror=x>.csv"
    html = report.render_html(manifest)
    assert "<img onerror" not in html
    assert "&lt;img" in html


def test_renderers_agree_on_the_numbers(manifest):
    md = report.render_markdown(manifest)
    html = report.render_html(manifest)
    for value in ("1.0427", "0.3325", "0.0059", "216", "branch_01"):
        assert value in md and value in html


def test_report_is_deterministic(manifest):
    assert report.render_html(manifest) == report.render_html(manifest)
    assert report.render_markdown(json.loads(json.dumps(manifest))) == (
        report.render_markdown(manifest)
    )
