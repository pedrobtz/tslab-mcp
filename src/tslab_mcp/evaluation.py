"""Turn a cross-validation frame into a per-model metric table.

Wraps ``utilsforecast.evaluation.evaluate``. The contract is that this module
*degrades rather than fails*: a ten-minute cross-validation run must never be
lost because a metric signature moved upstream, so any failure falls back to a
mean-absolute-error computed directly from the frame, with the reason attached.
"""

from __future__ import annotations

import sys
from functools import partial
from typing import Any

import pandas as pd

from .features import seasonal_period

#: Columns ``evaluate`` adds or carries through that are not model predictions.
_NON_MODEL_COLUMNS = frozenset({"unique_id", "ds", "cutoff", "metric", "y"})


def evaluation_seasonality(freq: str) -> int:
    """Seasonality used for MASE.

    Prefers TimeCopilot's own ``get_seasonality`` so the number matches what the
    library reports elsewhere (it follows the gluonts/M4 convention, where daily
    data is non-seasonal). Falls back to this package's richer table otherwise.

    The preference is conditional on TimeCopilot already being imported: it is,
    whenever a real cross-validation produced the frame, and paying a ~30 second
    import just to look up an integer would be absurd anywhere else.
    """
    if "timecopilot" not in sys.modules:
        return seasonal_period(freq)
    try:
        from timecopilot.models.utils.forecaster import get_seasonality
    except Exception:  # noqa: BLE001 -- fall back rather than fail the run
        return seasonal_period(freq)
    try:
        return int(get_seasonality(freq))
    except Exception:  # noqa: BLE001 -- unknown alias
        return seasonal_period(freq)


def _metric_functions(
    metrics: list[str], seasonality: int, train_df: pd.DataFrame
) -> list[Any]:
    """Build the callables ``evaluate`` expects, binding MASE's extra args.

    ``mase`` takes ``seasonality`` and ``train_df`` as required arguments;
    ``partial`` binds the first, and ``evaluate`` supplies the second.
    """
    from utilsforecast.losses import mae, mape, mase, rmse, smape

    simple = {"mae": mae, "mape": mape, "rmse": rmse, "smape": smape}
    functions: list[Any] = []
    for name in metrics:
        if name == "mase":
            functions.append(partial(mase, seasonality=seasonality))
        elif name in simple:
            functions.append(simple[name])
        else:
            raise ValueError(
                f"Unsupported metric '{name}'. Choose from: mase, smape, mae, "
                "rmse, mape."
            )
    return functions


def _present_models(cv_df: pd.DataFrame, models: list[str]) -> list[str]:
    """Model names that actually produced a column in the CV frame."""
    return [name for name in models if name in cv_df.columns]


def _aggregate(result: pd.DataFrame, models: list[str]) -> dict[str, dict[str, float]]:
    """Mean each metric across series and windows, keyed metric -> model -> value.

    Only model columns are averaged: ``evaluate`` carries ``cutoff`` through
    when the CV frame has one, and averaging a timestamp is meaningless.
    """
    columns = [c for c in models if c in result.columns]
    aggregated = result.groupby("metric")[columns].mean().round(6)
    return {
        str(metric): {
            model: float(value)
            for model, value in row.items()
            if pd.notna(value)
        }
        for metric, row in aggregated.iterrows()
    }


def _fallback_mae(
    cv_df: pd.DataFrame, models: list[str]
) -> dict[str, dict[str, float]]:
    """Mean absolute error computed straight off the frame."""
    scores: dict[str, float] = {}
    for model in _present_models(cv_df, models):
        errors = (cv_df[model] - cv_df["y"]).abs()
        mean_error = errors.mean()
        if pd.notna(mean_error):
            scores[model] = round(float(mean_error), 6)
    return {"mae": scores}


def _ranking(table: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    """Models ordered best-first per metric. Every metric here is lower-better."""
    return {
        metric: [model for model, _ in sorted(scores.items(), key=lambda kv: kv[1])]
        for metric, scores in table.items()
        if scores
    }


def evaluate_cv(
    cv_df: pd.DataFrame,
    train_df: pd.DataFrame,
    models: list[str],
    metrics: list[str],
    freq: str,
) -> dict[str, Any]:
    """Aggregate CV predictions into a metric table plus a per-metric ranking.

    Never raises: an upstream failure is reported in ``degraded_reason`` beside
    a fallback MAE table.
    """
    present = _present_models(cv_df, models)
    missing = [name for name in models if name not in present]
    seasonality = evaluation_seasonality(freq)

    payload: dict[str, Any] = {
        "models_evaluated": present,
        "seasonality_used_for_mase": seasonality,
        "n_rows_evaluated": len(cv_df),
    }
    if missing:
        payload["models_without_predictions"] = missing

    if not present:
        payload["metrics"] = {}
        payload["degraded_reason"] = (
            "No requested model produced a prediction column. Frame columns: "
            f"{list(cv_df.columns)}."
        )
        return payload

    try:
        from utilsforecast.evaluation import evaluate

        result = evaluate(
            cv_df,
            metrics=_metric_functions(metrics, seasonality, train_df),
            models=present,
            train_df=train_df,
        )
        table = _aggregate(result, present)
    except Exception as exc:  # noqa: BLE001 -- never lose a finished CV run
        table = _fallback_mae(cv_df, present)
        payload["degraded_reason"] = (
            f"{type(exc).__name__}: {exc}. Fell back to a directly computed "
            "MAE; the full per-window predictions are in the parquet artifact."
        )

    payload["metrics"] = table
    payload["ranking"] = _ranking(table)
    return payload
