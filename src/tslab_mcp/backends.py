"""Execution backends: statsforecast by default, TimeCopilot when asked for.

The statistical models this server exposes are statsforecast models — TimeCopilot
wraps the very same classes. Calling statsforecast directly makes the common case
a ~340 MB install instead of ~2 GB, with no torch, and is what the base package
does. TimeCopilot is an optional extra that adds the pretrained foundation models
(and Prophet).

Both backends return frames in the same shape, because the tool layer, the run
records and the reports all depend on it:

* ``forecast`` -> ``unique_id, ds, {alias}[, {alias}-lo-N, {alias}-hi-N ...]``
* ``cross_validation`` -> ``unique_id, ds, cutoff, y, {alias}...``
* ``detect_anomalies`` -> the cross-validation columns plus ``{alias}-anomaly``

statsforecast defaults to ``n_jobs=1``, and this deliberately leaves it there.
Its parallel mode spawns worker processes that re-import the entry module, which
inside an MCP server buys contention and a stdout hazard rather than speed.
"""

from __future__ import annotations

import inspect
from typing import Any, Protocol

import pandas as pd

from .seasonality import model_seasonality


class Backend(Protocol):
    """What the tool layer needs from a forecasting engine."""

    name: str

    def forecast(
        self, df: pd.DataFrame, h: int, freq: str, level: list[int] | None
    ) -> pd.DataFrame: ...

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        n_windows: int,
        step_size: int | None,
    ) -> pd.DataFrame: ...

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        n_windows: int | None,
        level: int,
    ) -> pd.DataFrame: ...


class StatsForecastBackend:
    """Runs statistical models through statsforecast, with no torch in sight."""

    name = "statsforecast"

    def __init__(self, model_names: list[str]) -> None:
        self.model_names = model_names

    def _engine(self, freq: str) -> Any:
        """Build a StatsForecast over the requested models for this frequency.

        Rebuilt per call because statsforecast takes ``freq`` and the models'
        ``season_length`` up front, and both depend on the panel.
        """
        from statsforecast import StatsForecast
        from statsforecast import models as sf_models

        season_length = model_seasonality(freq)
        instances = []
        for name in self.model_names:
            cls = getattr(sf_models, name)
            # Some models are seasonal and some are not; ask rather than assume,
            # so a signature change upstream cannot silently drop the period.
            if "season_length" in inspect.signature(cls).parameters:
                instances.append(cls(season_length=season_length))
            else:
                instances.append(cls())
        return StatsForecast(models=instances, freq=freq)

    def forecast(
        self, df: pd.DataFrame, h: int, freq: str, level: list[int] | None
    ) -> pd.DataFrame:
        out: pd.DataFrame = self._engine(freq).forecast(
            h=h, df=df, level=level or None
        )
        return out

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        n_windows: int,
        step_size: int | None,
    ) -> pd.DataFrame:
        out: pd.DataFrame = self._engine(freq).cross_validation(
            h=h, df=df, n_windows=n_windows, step_size=step_size or h
        )
        return out

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        n_windows: int | None,
        level: int,
    ) -> pd.DataFrame:
        """Cross-validate with intervals, then flag actuals that fall outside.

        statsforecast has no anomaly API, but TimeCopilot's is exactly this, so
        the output — including the ``{alias}-anomaly`` column name — matches
        what the other backend produces.
        """
        windows = n_windows if n_windows is not None else _max_windows(df, h)
        result: pd.DataFrame = self._engine(freq).cross_validation(
            h=h, df=df, n_windows=windows, step_size=h, level=[level]
        )
        for alias in self.model_names:
            lower, upper = f"{alias}-lo-{level}", f"{alias}-hi-{level}"
            if lower in result.columns and upper in result.columns:
                outside = (result["y"] < result[lower]) | (result["y"] > result[upper])
                result[f"{alias}-anomaly"] = outside.fillna(False)
        return result


class TimeCopilotBackend:
    """Runs any registered model, including the pretrained foundation ones."""

    name = "timecopilot"

    def __init__(self, model_names: list[str]) -> None:
        self.model_names = model_names

    def _forecaster(self) -> Any:
        from timecopilot import TimeCopilotForecaster

        from .models import resolve_timecopilot

        return TimeCopilotForecaster(models=resolve_timecopilot(self.model_names))

    def forecast(
        self, df: pd.DataFrame, h: int, freq: str, level: list[int] | None
    ) -> pd.DataFrame:
        out: pd.DataFrame = self._forecaster().forecast(
            df=df, h=h, freq=freq, level=level or None
        )
        return out

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        n_windows: int,
        step_size: int | None,
    ) -> pd.DataFrame:
        out: pd.DataFrame = self._forecaster().cross_validation(
            df=df, h=h, freq=freq, n_windows=n_windows, step_size=step_size
        )
        return out

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        n_windows: int | None,
        level: int,
    ) -> pd.DataFrame:
        out: pd.DataFrame = self._forecaster().detect_anomalies(
            df=df, h=h, freq=freq, n_windows=n_windows, level=level
        )
        return out


def _max_windows(df: pd.DataFrame, h: int) -> int:
    """How many non-overlapping windows the shortest series can support."""
    shortest = int(df.groupby("unique_id").size().min())
    return max(1, (shortest - h) // h)


def select(model_names: list[str]) -> Backend:
    """Choose the lightest backend that can run every requested model.

    Statistical-only requests go to statsforecast. As soon as one model needs
    TimeCopilot, the whole request goes there — it carries the statistical
    models too, and letting it merge the frames avoids reimplementing a join
    that already exists upstream.
    """
    from .models import STATS_MODELS, require_timecopilot

    if all(name in STATS_MODELS for name in model_names):
        return StatsForecastBackend(model_names)
    require_timecopilot([n for n in model_names if n not in STATS_MODELS])
    return TimeCopilotBackend(model_names)
