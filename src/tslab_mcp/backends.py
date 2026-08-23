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
from pandas.api.types import is_bool_dtype

from .seasonality import model_seasonality

#: Models with no native interval implementation. StatsForecast can give them
#: deterministic conformal intervals when the configuration is supplied at
#: construction time.
CONFORMAL_INTERVAL_MODELS = frozenset({"ADIDA", "IMAPA", "CrostonClassic"})
CONFORMAL_N_WINDOWS = 2
CONFORMAL_METHOD = "conformal_distribution"

#: Verified lower bounds for finite point and native-interval output in
#: StatsForecast 2.1.1. Seasonal and conformal requirements are applied below
#: because they depend on the request frequency or horizon.
MINIMUM_TRAINING_OBSERVATIONS = {
    "AutoARIMA": 4,
    "AutoETS": 7,
    "AutoCES": 5,
    "Theta": 4,
    "DynamicOptimizedTheta": 4,
}


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

    def _engine(
        self, freq: str, h: int | None = None, with_intervals: bool = False
    ) -> Any:
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
            parameters = inspect.signature(cls).parameters
            kwargs: dict[str, Any] = {}
            # Some models are seasonal and some are not; ask rather than assume,
            # so a signature change upstream cannot silently drop the period.
            if "season_length" in parameters:
                kwargs["season_length"] = season_length
            # Canonical aliases are the public contract. AutoCES otherwise emits
            # ``CES``, which makes evaluation treat its predictions as missing.
            if "alias" in parameters:
                kwargs["alias"] = name
            if with_intervals and name in CONFORMAL_INTERVAL_MODELS:
                if h is None:  # pragma: no cover - internal contract
                    raise AssertionError("h is required for conformal intervals")
                from statsforecast.utils import ConformalIntervals

                kwargs["prediction_intervals"] = ConformalIntervals(
                    n_windows=CONFORMAL_N_WINDOWS,
                    h=h,
                    method=CONFORMAL_METHOD,
                )
            instances.append(cls(**kwargs))
        return StatsForecast(models=instances, freq=freq)

    def forecast(
        self, df: pd.DataFrame, h: int, freq: str, level: list[int] | None
    ) -> pd.DataFrame:
        out: pd.DataFrame = self._engine(
            freq, h=h, with_intervals=bool(level)
        ).forecast(
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
        windows = resolve_anomaly_windows(
            df, self.model_names, freq, h, requested=n_windows
        )
        result: pd.DataFrame = self._engine(
            freq, h=h, with_intervals=True
        ).cross_validation(
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


def minimum_training_length(
    model_name: str, freq: str, h: int, with_intervals: bool = False
) -> int:
    """Conservative training observations required by a model invocation."""
    required = MINIMUM_TRAINING_OBSERVATIONS.get(model_name, 2)
    if model_name == "SeasonalNaive":
        required = max(required, model_seasonality(freq))
    if with_intervals and model_name in CONFORMAL_INTERVAL_MODELS:
        required = max(required, 2 * h + 1)
    return required


def resolve_anomaly_windows(
    df: pd.DataFrame,
    model_names: list[str],
    freq: str,
    h: int,
    requested: int | None,
) -> int:
    """Resolve the default without consuming history needed to fit the model."""
    if requested is not None:
        return requested
    shortest = int(df.groupby("unique_id").size().min())
    reserve = max(
        minimum_training_length(name, freq, h, with_intervals=True)
        for name in model_names
    )
    return max(1, (shortest - reserve) // h)


def validate_history(
    df: pd.DataFrame,
    model_names: list[str],
    freq: str,
    h: int,
    operation: str,
    n_windows: int | None = None,
    step_size: int | None = None,
    with_intervals: bool = False,
) -> None:
    """Fail before fitting when a model/window request cannot use every series."""
    if operation not in {"forecast", "cross-validation", "anomaly detection"}:
        raise ValueError(f"Unknown history-validation operation: {operation}")
    test_size = 0
    if n_windows is not None:
        step = step_size or h
        test_size = h + step * (n_windows - 1)

    counts = df.groupby("unique_id").size()
    violations: list[str] = []
    for name in model_names:
        training = minimum_training_length(name, freq, h, with_intervals)
        required = test_size + training
        short = counts[counts < required]
        if short.empty:
            continue
        examples = ", ".join(f"{uid}={int(n)}" for uid, n in short.head(5).items())
        suffix = ", ..." if len(short) > 5 else ""
        violations.append(
            f"{name} needs at least {required} observations; {examples}{suffix}"
        )

    if violations:
        settings = f"h={h}"
        if n_windows is not None:
            settings += f", n_windows={n_windows}, step_size={step_size or h}"
        raise ValueError(
            f"Insufficient history for {operation} ({settings}, freq='{freq}'). "
            + " ".join(violations)
            + " Reduce the horizon/windows, load more history, or choose a "
            "lower-history model such as HistoricAverage or ZeroModel."
        )


def _non_finite_counts(
    df: pd.DataFrame, columns: list[str]
) -> dict[str, int]:
    invalid: dict[str, int] = {}
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        mask = values.isna() | values.isin([float("inf"), float("-inf")])
        if mask.any():
            invalid[column] = int(mask.sum())
    return invalid


def validate_prediction_frame(
    df: pd.DataFrame,
    model_names: list[str],
    operation: str,
    required_columns: tuple[str, ...] = ("unique_id", "ds"),
) -> None:
    """Enforce canonical, finite prediction columns at the backend boundary."""
    if df.empty:
        raise ValueError(f"{operation} returned no rows.")
    missing_structure = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_structure:
        raise ValueError(
            f"{operation} did not return required structural column(s) "
            f"{missing_structure}. Frame columns: {list(df.columns)}. No "
            "artifact or successful run record was created."
        )
    missing = [name for name in model_names if name not in df.columns]
    if missing:
        raise ValueError(
            f"{operation} did not return canonical prediction column(s) "
            f"{missing}. Frame columns: {list(df.columns)}. Call "
            "tsf_list_models and report this backend alias mismatch."
        )
    invalid = _non_finite_counts(df, model_names)
    if invalid:
        raise ValueError(
            f"{operation} produced non-finite predictions: {invalid}. No "
            "artifact or successful run record was created. Load more history, "
            "reduce the horizon/windows, or choose another model."
        )


def validate_forecast_frame(
    df: pd.DataFrame, model_names: list[str], levels: list[int]
) -> None:
    """Validate points and every explicitly requested prediction interval."""
    validate_prediction_frame(df, model_names, "Forecast")
    expected = [
        f"{name}-{side}-{level}"
        for name in model_names
        for level in levels
        for side in ("lo", "hi")
    ]
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise ValueError(
            f"Forecast did not provide requested interval column(s) {missing}. "
            "Retry with level=[] for point forecasts or choose a model with "
            "interval support."
        )
    invalid = _non_finite_counts(df, expected)
    if invalid:
        raise ValueError(
            f"Forecast produced non-finite prediction intervals: {invalid}. "
            "No artifact or successful run record was created."
        )


def validate_anomaly_frame(df: pd.DataFrame, model_name: str, level: int) -> None:
    """Validate every value used to decide and report an anomaly."""
    validate_prediction_frame(
        df,
        [model_name],
        "Anomaly detection",
        required_columns=("unique_id", "ds", "cutoff", "y"),
    )
    intervals = [f"{model_name}-lo-{level}", f"{model_name}-hi-{level}"]
    missing = [column for column in intervals if column not in df.columns]
    if missing:
        raise ValueError(
            f"Anomaly detection did not provide interval column(s) {missing}. "
            "No artifact or successful run record was created."
        )
    invalid = _non_finite_counts(df, intervals)
    if invalid:
        raise ValueError(
            f"Anomaly detection produced non-finite prediction intervals: "
            f"{invalid}. No artifact or successful run record was created."
        )

    flag = f"{model_name}-anomaly"
    if flag not in df.columns:
        raise ValueError(
            f"Anomaly detection did not return canonical flag column '{flag}'. "
            f"Frame columns: {list(df.columns)}. No artifact or successful run "
            "record was created."
        )
    if df[flag].isna().any() or not is_bool_dtype(df[flag].dtype):
        raise ValueError(
            f"Anomaly detection returned a non-boolean or missing value in "
            f"'{flag}'. No artifact or successful run record was created."
        )


def prediction_interval_configuration(
    backend_name: str, model_names: list[str], h: int, levels: list[int]
) -> dict[str, Any] | None:
    """Manifest-ready description of how requested intervals were produced."""
    if not levels:
        return None
    if backend_name != "statsforecast":
        return {"provider": backend_name, "method": "backend_default"}
    conformal = [name for name in model_names if name in CONFORMAL_INTERVAL_MODELS]
    native = [name for name in model_names if name not in CONFORMAL_INTERVAL_MODELS]
    return {
        "provider": backend_name,
        "native_models": native,
        "conformal_models": conformal,
        "conformal": (
            {
                "method": CONFORMAL_METHOD,
                "n_windows": CONFORMAL_N_WINDOWS,
                "h": h,
            }
            if conformal
            else None
        ),
    }


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
