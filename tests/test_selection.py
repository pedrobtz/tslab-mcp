"""Deterministic validation of the agent's model-selection proposal."""

from __future__ import annotations

import pytest

from tslab_mcp import selection


def _cv(**scores):
    return {
        "kind": "cross_validation",
        "at": "2026-08-22T00:00:00+00:00",
        "artifact": "/tmp/cv.parquet",
        "metrics": {"mase": scores},
    }


def test_lowest_mase_challenger_is_validated():
    decision = selection.evaluate_proposal(
        _cv(SeasonalNaive=1.0, AutoETS=0.8, AutoARIMA=0.9),
        proposed_model="AutoETS",
        metric="mase",
        min_relative_improvement=0.0,
        finalize_baseline=False,
    )
    assert decision["status"] == "validated_challenger"
    assert decision["selected_model"] == "AutoETS"
    assert decision["relative_improvement"] == pytest.approx(0.2)
    assert decision["requires_additional_comparison"] is False


def test_nonwinning_proposal_is_rejected_with_recommendation():
    decision = selection.evaluate_proposal(
        _cv(SeasonalNaive=1.0, AutoETS=0.8, AutoARIMA=0.9),
        proposed_model="AutoARIMA",
        metric="mase",
        min_relative_improvement=0.0,
        finalize_baseline=False,
    )
    assert decision["status"] == "rejected_proposal"
    assert decision["selected_model"] is None
    assert decision["recommended_model"] == "AutoETS"


def test_any_exact_score_winner_is_accepted():
    decision = selection.evaluate_proposal(
        _cv(SeasonalNaive=1.0, AutoETS=0.8, AutoARIMA=0.8),
        proposed_model="AutoARIMA",
        metric="mase",
        min_relative_improvement=0.0,
        finalize_baseline=False,
    )
    assert decision["status"] == "validated_challenger"
    assert decision["selected_model"] == "AutoARIMA"
    assert decision["tied_best_challengers"] == ["AutoETS", "AutoARIMA"]


def test_first_baseline_win_requests_a_broader_comparison():
    decision = selection.evaluate_proposal(
        _cv(SeasonalNaive=0.7, AutoETS=0.8),
        proposed_model="SeasonalNaive",
        metric="mase",
        min_relative_improvement=0.0,
        finalize_baseline=False,
    )
    assert decision["status"] == "needs_additional_comparison"
    assert decision["selected_model"] is None
    assert decision["requires_additional_comparison"] is True


def test_baseline_can_be_finalized_when_no_challenger_qualifies():
    decision = selection.evaluate_proposal(
        _cv(SeasonalNaive=0.7, AutoETS=0.8),
        proposed_model="SeasonalNaive",
        metric="mase",
        min_relative_improvement=0.0,
        finalize_baseline=True,
    )
    assert decision["status"] == "baseline_fallback"
    assert decision["selected_model"] == "SeasonalNaive"


def test_material_improvement_threshold_is_enforced():
    decision = selection.evaluate_proposal(
        _cv(SeasonalNaive=1.0, AutoETS=0.97),
        proposed_model="AutoETS",
        metric="mase",
        min_relative_improvement=0.05,
        finalize_baseline=False,
    )
    assert decision["challenger_qualifies"] is False
    assert decision["status"] == "needs_additional_comparison"


def test_selection_requires_cv_and_seasonal_naive():
    with pytest.raises(ValueError, match="tsf_cross_validate"):
        selection.latest_cross_validation([])
    with pytest.raises(ValueError, match="SeasonalNaive"):
        selection.evaluate_proposal(
            _cv(AutoETS=0.8), "AutoETS", "mase", 0.0, False
        )


def test_forecast_must_match_selection_unless_explicitly_overridden():
    runs = [
        _cv(SeasonalNaive=1.0, AutoETS=0.8),
        {
            "kind": "model_selection",
            "status": "validated_challenger",
            "selected_model": "AutoETS",
        },
    ]
    assert selection.require_forecast_matches_selection(
        runs, ["AutoETS"], False
    ) == "AutoETS"
    with pytest.raises(ValueError, match="allow_selection_override"):
        selection.require_forecast_matches_selection(
            runs, ["AutoARIMA"], False
        )
    assert selection.require_forecast_matches_selection(
        runs, ["AutoARIMA"], True
    ) == "AutoETS"


def test_new_cv_invalidates_an_older_selection_checkpoint():
    runs = [
        _cv(SeasonalNaive=1.0, AutoETS=0.8),
        {
            "kind": "model_selection",
            "status": "validated_challenger",
            "selected_model": "AutoETS",
        },
        _cv(SeasonalNaive=0.9, AutoARIMA=0.7),
    ]
    assert selection.validated_model_after_latest_cv(runs) is None


def test_unresolved_selection_blocks_forecast_without_override():
    runs = [
        _cv(SeasonalNaive=1.0, AutoETS=0.8),
        {
            "kind": "model_selection",
            "status": "rejected_proposal",
            "selected_model": None,
        },
    ]
    with pytest.raises(ValueError, match="did not validate a model"):
        selection.require_forecast_matches_selection(runs, ["AutoETS"], False)
    assert selection.require_forecast_matches_selection(
        runs, ["AutoETS"], True
    ) is None
