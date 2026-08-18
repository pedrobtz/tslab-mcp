"""Statistical features used to pick a model family.

Pure functions over pandas objects: no I/O, no TimeCopilot import, no global
state. Every helper returns ``None`` rather than ``nan`` on degenerate input,
because ``nan`` is not valid JSON and would break the tool response.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

#: Seasonal period per pandas offset alias family. Covers both the legacy
#: aliases (``M``, ``H``, ``A``) and the pandas>=2.2 spellings (``ME``, ``h``,
#: ``YE``), since either may reach us from user input or ``infer_freq``.
_SEASONAL_PERIODS: dict[str, int] = {
    "Y": 1, "YE": 1, "YS": 1, "A": 1, "AS": 1, "BY": 1, "BYE": 1, "BYS": 1,
    "Q": 4, "QE": 4, "QS": 4, "BQ": 4, "BQE": 4, "BQS": 4,
    "M": 12, "ME": 12, "MS": 12, "BM": 12, "BME": 12, "BMS": 12,
    "SM": 24, "SME": 24, "SMS": 24,
    "W": 52,
    "D": 7, "C": 7,
    "B": 5,
    "H": 24, "BH": 24, "CBH": 24,
    "T": 1440, "MIN": 1440,
    "S": 3600,
    "L": 1000,
}

#: Feature keys every row carries, in response order.
FEATURE_KEYS = (
    "unique_id",
    "n",
    "start",
    "end",
    "mean",
    "sd",
    "cv",
    "pct_zero",
    "trend_strength",
    "seasonal_strength",
    "acf1_diff",
)


def _clean(value: Any) -> float | None:
    """Round to a JSON-safe float, mapping ``nan``/``inf``/non-numbers to ``None``.

    ``nan`` is not valid JSON, so it must never reach a tool response.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number) or number in (float("inf"), float("-inf")):
        return None
    return round(number, 4)


def seasonal_period(freq: str) -> int:
    """Observations per seasonal cycle for a pandas offset alias.

    Falls back to 1 (no seasonality) for anything unrecognised, which makes
    :func:`seasonal_strength` return ``None`` rather than a fabricated number.
    """
    if not freq:
        return 1
    head = freq.split("-")[0].strip()
    head = head.lstrip("0123456789")
    if head in ("min", "ms", "us", "ns"):  # case-sensitive pandas>=2.2 aliases
        return {"min": 1440, "ms": 1000, "us": 1000, "ns": 1000}[head]
    return _SEASONAL_PERIODS.get(head.upper(), 1)


def trend_strength(y: pd.Series) -> float | None:
    """R-squared of an OLS fit against time: 0 = flat, 1 = a pure ramp."""
    if len(y) < 3:
        return None
    values = y.astype("float64")
    if values.std(ddof=1) == 0:  # a flat line has no trend to explain
        return None
    x = pd.Series(range(len(values)), dtype="float64", index=values.index)
    correlation = values.corr(x)
    if correlation is None or pd.isna(correlation):
        return None
    return _clean(correlation**2)


def seasonal_strength(y: pd.Series, period: int) -> float | None:
    """Strength of seasonality after removing the trend, in ``[0, 1]``.

    Uses the STL formulation ``1 - Var(remainder) / Var(seasonal + remainder)``
    (Hyndman), which measures the seasonal component against what is left once
    the trend is taken out. The obvious alternative -- share of variance
    explained by the calendar-position means -- is *not* trend independent: on a
    series with identical seasonality it falls from 1.00 to 0.03 as the trend
    steepens, because the trend inflates the denominator while pulling the
    position means toward the series mean. Growing business series are exactly
    the case that trends and is seasonal, so that measure fails where it matters.

    Read the number with its noise floor in mind: STL attributes some variance
    to a seasonal component by chance, so pure noise scores roughly 0.3-0.5 here.
    Values in that band mean "no evidence", not "mildly seasonal"; only clearly
    higher values are worth acting on. That failure direction is the safe one in
    this workflow -- a false positive is caught by cross-validation, whereas a
    false negative means the seasonal model is never tried at all.

    ``None`` -- never ``nan`` -- when the question is not answerable: a
    non-seasonal frequency, fewer than two full cycles, missing values, or no
    variance to explain.
    """
    if period <= 1 or len(y) < 2 * period:
        return None
    values = y.astype("float64").reset_index(drop=True)
    # Guard explicitly: statsmodels does not reject NaN input, it silently
    # returns a strength (tsfeatures returns 1.0, i.e. "perfectly seasonal",
    # for a series with a hole in it).
    if values.isna().any():
        return None
    total = values.var(ddof=1)
    if pd.isna(total) or total <= 0:
        return None
    return _stl_seasonal_strength(values, period)


def _stl_seasonal_strength(values: pd.Series, period: int) -> float | None:
    """STL seasonal strength, or ``None`` if the decomposition cannot be trusted.

    ``statsmodels`` is always installed -- statsforecast depends on it -- so this
    needs no optional-dependency dance. The seasonal smoother is set longer than
    the series, which approximates R's ``s.window="periodic"`` and keeps trend
    from leaking into the seasonal component (it roughly halves the score on a
    trend-only series).
    """
    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError:  # pragma: no cover -- statsforecast guarantees it
        return None

    periodic_window = 2 * len(values) + 1  # odd, and longer than the series
    try:
        fitted = STL(
            values, period=period, seasonal=periodic_window, robust=True
        ).fit()
    except Exception:  # noqa: BLE001 -- a feature is never worth failing a tool
        return None

    remainder, seasonal = fitted.resid, fitted.seasonal
    denominator = (seasonal + remainder).var(ddof=1)
    if pd.isna(denominator) or denominator <= 0:
        return None
    strength = 1 - remainder.var(ddof=1) / denominator
    return _clean(min(max(strength, 0.0), 1.0))


def acf1(y: pd.Series) -> float | None:
    """Lag-1 autocorrelation; ``None`` when undefined (constant or too short)."""
    if len(y) < 3:
        return None
    values = y.astype("float64")
    if values.std(ddof=1) == 0:
        return None
    return _clean(values.autocorr(lag=1))


def series_features(
    uid: str, y: pd.Series, ds: pd.Series, period: int
) -> dict[str, Any]:
    """Feature row for a single series. ``y`` and ``ds`` must be aligned."""
    values = y.astype("float64").reset_index(drop=True)
    mean = values.mean()
    sd = values.std(ddof=1) if len(values) > 1 else None
    has_mean = mean is not None and not pd.isna(mean) and mean != 0
    cv = sd / mean if sd is not None and has_mean else None
    return {
        "unique_id": uid,
        "n": len(values),
        "start": ds.min().isoformat(),
        "end": ds.max().isoformat(),
        "mean": _clean(mean),
        "sd": _clean(sd),
        "cv": _clean(cv),
        "pct_zero": _clean((values == 0).mean()),
        "trend_strength": trend_strength(values),
        "seasonal_strength": seasonal_strength(values, period),
        "acf1_diff": acf1(values.diff().dropna()),
    }


def panel_features(df: pd.DataFrame, period: int) -> list[dict[str, Any]]:
    """Feature rows for every series in a long-format panel, sorted by id."""
    rows = [
        series_features(str(uid), group["y"], group["ds"], period)
        for uid, group in df.groupby("unique_id", sort=True)
    ]
    rows.sort(key=lambda row: str(row["unique_id"]))
    return rows
