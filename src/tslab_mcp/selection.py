"""Deterministic model-selection policy over recorded CV evidence."""

from __future__ import annotations

import math
from typing import Any

BASELINE_MODEL = "SeasonalNaive"


def latest_cross_validation(runs: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    """Return the index and payload of the latest recorded CV run."""
    for index in range(len(runs) - 1, -1, -1):
        run = runs[index]
        if run.get("kind") == "cross_validation":
            return index, run
    raise ValueError(
        "No cross-validation run is recorded for this handle. Call "
        "tsf_cross_validate before tsf_select_model."
    )


def _finite_scores(cv_run: dict[str, Any], metric: str) -> dict[str, float]:
    metrics = cv_run.get("metrics", {})
    raw_scores = metrics.get(metric)
    if not isinstance(raw_scores, dict) or not raw_scores:
        available = sorted(metrics) if isinstance(metrics, dict) else []
        raise ValueError(
            f"The latest CV run has no finite '{metric}' table. Available "
            f"metrics: {available}. Run cross-validation requesting '{metric}'."
        )

    scores: dict[str, float] = {}
    for model, value in raw_scores.items():
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            scores[str(model)] = score
    if not scores:
        raise ValueError(
            f"The latest CV run contains no finite '{metric}' model scores."
        )
    return scores


def evaluate_proposal(
    cv_run: dict[str, Any],
    proposed_model: str,
    metric: str,
    min_relative_improvement: float,
    finalize_baseline: bool,
) -> dict[str, Any]:
    """Validate an agent proposal against the numerical winner and baseline."""
    scores = _finite_scores(cv_run, metric)
    if BASELINE_MODEL not in scores:
        raise ValueError(
            f"The latest CV '{metric}' table does not contain {BASELINE_MODEL}. "
            "Run cross-validation again with the required baseline."
        )
    if proposed_model not in scores:
        raise ValueError(
            f"Proposed model '{proposed_model}' has no finite '{metric}' score "
            f"in the latest CV run. Scored models: {list(scores)}."
        )

    baseline_score = scores[BASELINE_MODEL]
    challengers = [
        (name, score) for name, score in scores.items() if name != BASELINE_MODEL
    ]
    best_challenger: str | None = None
    best_score: float | None = None
    tied_best: list[str] = []
    if challengers:
        best_score = min(score for _, score in challengers)
        tied_best = [name for name, score in challengers if score == best_score]
        best_challenger = tied_best[0]

    absolute_improvement = (
        baseline_score - best_score if best_score is not None else None
    )
    relative_improvement = (
        absolute_improvement / abs(baseline_score)
        if absolute_improvement is not None and baseline_score != 0
        else 0.0
    )
    challenger_qualifies = bool(
        best_score is not None
        and best_score < baseline_score
        and relative_improvement >= min_relative_improvement
    )

    selected_model: str | None = None
    recommended_model = best_challenger if challenger_qualifies else BASELINE_MODEL
    requires_additional_comparison = False
    if challenger_qualifies and proposed_model in tied_best:
        status = "validated_challenger"
        selected_model = proposed_model
        reason = (
            f"{proposed_model} is tied for the lowest {metric} and improves on "
            f"{BASELINE_MODEL} by {relative_improvement:.2%}."
        )
    elif challenger_qualifies:
        status = "rejected_proposal"
        reason = (
            f"{proposed_model} is not tied for the lowest {metric}. Propose "
            f"{recommended_model} or another exact-score tie."
        )
    elif not finalize_baseline:
        status = "needs_additional_comparison"
        requires_additional_comparison = True
        reason = (
            f"No challenger improves on {BASELINE_MODEL} by the required "
            f"{min_relative_improvement:.2%}. Run one justified broader CV "
            "comparison, or finalize the baseline when no candidates remain."
        )
    elif proposed_model == BASELINE_MODEL:
        status = "baseline_fallback"
        selected_model = BASELINE_MODEL
        reason = (
            f"No challenger met the required improvement; {BASELINE_MODEL} "
            "is validated as the final fallback."
        )
    else:
        status = "rejected_proposal"
        reason = (
            f"No challenger met the required improvement. Finalize with "
            f"{BASELINE_MODEL}, not {proposed_model}."
        )

    return {
        "proposed_model": proposed_model,
        "proposed_score": scores[proposed_model],
        "primary_metric": metric,
        "baseline_model": BASELINE_MODEL,
        "baseline_score": baseline_score,
        "best_challenger": best_challenger,
        "best_challenger_score": best_score,
        "tied_best_challengers": tied_best,
        "absolute_improvement": absolute_improvement,
        "relative_improvement": relative_improvement,
        "min_relative_improvement": min_relative_improvement,
        "challenger_qualifies": challenger_qualifies,
        "status": status,
        "selected_model": selected_model,
        "recommended_model": recommended_model,
        "requires_additional_comparison": requires_additional_comparison,
        "reason": reason,
        "cv_artifact": cv_run.get("artifact"),
        "cv_at": cv_run.get("at"),
    }


def validated_model_after_latest_cv(runs: list[dict[str, Any]]) -> str | None:
    """Return a validated selection only when it follows the latest CV run."""
    try:
        cv_index, _ = latest_cross_validation(runs)
    except ValueError:
        return None
    for run in reversed(runs[cv_index + 1 :]):
        if run.get("kind") != "model_selection":
            continue
        selected = run.get("selected_model")
        if run.get("status") in {"validated_challenger", "baseline_fallback"}:
            return str(selected) if selected is not None else None
    return None


def require_forecast_matches_selection(
    runs: list[dict[str, Any]], models: list[str], allow_override: bool
) -> str | None:
    """Prevent a forecast or detector diverging from the latest checkpoint."""
    try:
        cv_index, _ = latest_cross_validation(runs)
    except ValueError:
        return None
    checkpoint = next(
        (
            run
            for run in reversed(runs[cv_index + 1 :])
            if run.get("kind") == "model_selection"
        ),
        None,
    )
    if checkpoint is None:
        return None

    status = str(checkpoint.get("status"))
    raw_selected = checkpoint.get("selected_model")
    selected = str(raw_selected) if raw_selected is not None else None
    if allow_override:
        return selected
    if selected is None or status not in {
        "validated_challenger",
        "baseline_fallback",
    }:
        raise ValueError(
            f"The latest structured selection has status '{status}' and did "
            "not validate a model. Complete its retry/fallback instruction, "
            "or set allow_selection_override=true for an explicit "
            "user-directed forecast."
        )
    if models == [selected]:
        return selected
    raise ValueError(
        f"The latest structured selection validated '{selected}', but the "
        f"model request was {models}. Use only ['{selected}'], run a new "
        "cross-validation and selection, or set allow_selection_override=true "
        "for an explicit user-directed override."
    )
