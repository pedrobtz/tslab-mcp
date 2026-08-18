"""Seasonal period used by *models and metrics*.

This deliberately reimplements the gluonts/M4 convention that TimeCopilot uses
(via `gluonts.time_feature.get_seasonality`) rather than importing it, because
the statistical backend must produce identical numbers on an install where
neither gluonts nor TimeCopilot exists. Getting this wrong changes every
seasonal forecast and every MASE, silently.

Note the convention's sharp edge: **daily and weekly data are non-seasonal**
(`D -> 1`, `W -> 1`). That is right for MASE and for matching the literature,
and it is *not* what you want when describing a series, so
:func:`tslab_mcp.features.seasonal_period` keeps a separate, richer table
(`D -> 7`, `W -> 52`) for the descriptive feature. The two disagreeing is
intentional; the tools report which one they used.
"""

from __future__ import annotations

import pandas as pd

#: Base periods keyed by normalised pandas offset name, matching
#: ``gluonts.time_feature.seasonality.DEFAULT_SEASONALITIES``.
_BASE_SEASONALITIES: dict[str, int] = {
    "S": 3600,
    "s": 3600,
    "T": 1440,
    "min": 1440,
    "H": 24,
    "h": 24,
    "D": 1,
    "W": 1,
    "M": 12,
    "ME": 12,
    "B": 5,
    "Q": 4,
    "QE": 4,
}

_PANDAS_RENAMED_PERIOD_ENDS = pd.__version__ >= "2.2"


def _normalise(freq_name: str) -> str:
    """Collapse an offset name to the key used in the seasonality table.

    Start and end frequencies are treated alike (``MS`` and ``ME`` are both
    monthly), which is why the trailing ``S`` is stripped -- except on ``S``
    itself, which means seconds.
    """
    base = freq_name.split("-")[0]
    if len(base) >= 2 and base.endswith("S"):
        base = base[:-1]
        if _PANDAS_RENAMED_PERIOD_ENDS:
            base += "E"
    return base


def model_seasonality(freq: str) -> int:
    """Observations per seasonal cycle for a pandas offset alias.

    Used for a model's ``season_length`` and for MASE, so both backends agree.
    A multiple that does not divide the base period (``5h`` against a 24-hour
    day) falls back to 1 rather than inventing a fractional cycle.
    """
    if not freq:
        return 1
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except (ValueError, TypeError):
        return 1
    if offset is None:
        return 1

    base = _BASE_SEASONALITIES.get(_normalise(offset.name), 1)
    seasonality, remainder = divmod(base, offset.n)
    return seasonality if not remainder else 1
