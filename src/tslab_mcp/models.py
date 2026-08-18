"""The model registry, split by what each model costs to install.

Design invariant I3: the environment is discovered, not assumed. The statistical
models come from statsforecast and are always present. Everything else needs the
``foundation`` extra, which pulls TimeCopilot and ~2 GB of torch — so nothing
here imports it until a tool actually asks for one of its models.

:func:`probe` reports what resolved on *this* machine, which is how an agent
learns in one turn that ``Chronos`` is unavailable rather than discovering it in
a traceback ten minutes into a run.
"""

from __future__ import annotations

import importlib
from typing import Any

#: Friendly name -> attribute in ``statsforecast.models``. These need no extra,
#: no weights, and no torch; they are the default backend.
STATS_MODELS: dict[str, str] = {
    "SeasonalNaive": "SeasonalNaive",
    "HistoricAverage": "HistoricAverage",
    "AutoARIMA": "AutoARIMA",
    "AutoETS": "AutoETS",
    "AutoCES": "AutoCES",
    "Theta": "Theta",
    "DynamicOptimizedTheta": "DynamicOptimizedTheta",
    "ADIDA": "ADIDA",
    "IMAPA": "IMAPA",
    "CrostonClassic": "CrostonClassic",
    "ZeroModel": "ZeroModel",
}

#: Friendly name -> (module, attribute) inside TimeCopilot. Verified against
#: timecopilot 0.0.30; ``probe`` exists so drift fails softly.
TIMECOPILOT_MODELS: dict[str, tuple[str, str]] = {
    "Prophet": ("timecopilot.models.prophet", "Prophet"),
    "Chronos": ("timecopilot.models.foundation.chronos", "Chronos"),
    "FlowState": ("timecopilot.models.foundation.flowstate", "FlowState"),
    "Moirai": ("timecopilot.models.foundation.moirai", "Moirai"),
    "PatchTSTFM": ("timecopilot.models.foundation.patchtst_fm", "PatchTSTFM"),
    "Sundial": ("timecopilot.models.foundation.sundial", "Sundial"),
    "T0": ("timecopilot.models.foundation.t0", "T0"),
    "TabPFN": ("timecopilot.models.foundation.tabpfn", "TabPFN"),
    "TiRex": ("timecopilot.models.foundation.tirex", "TiRex"),
    "TimesFM": ("timecopilot.models.foundation.timesfm", "TimesFM"),
    "TimeGPT": ("timecopilot.models.foundation.timegpt", "TimeGPT"),
    "Toto": ("timecopilot.models.foundation.toto", "Toto"),
}

#: Every name this server recognises.
MODEL_REGISTRY: dict[str, str] = {
    **{name: "statistical" for name in STATS_MODELS},
    **{name: "foundation" for name in TIMECOPILOT_MODELS},
}

STATISTICAL_MODELS = frozenset(STATS_MODELS)
FOUNDATION_MODELS = frozenset(TIMECOPILOT_MODELS)

#: Reaches an external API rather than running locally.
REMOTE_MODELS = frozenset({"TimeGPT"})

#: The extra that installs everything in :data:`TIMECOPILOT_MODELS`.
EXTRA = "foundation"


def family_of(name: str) -> str:
    """``'statistical'`` or ``'foundation'`` for a registered model name."""
    return MODEL_REGISTRY.get(name, "foundation")


def _cheap_alternatives() -> str:
    return ", ".join(sorted(STATS_MODELS))


def timecopilot_available() -> bool:
    """Whether the optional extra is installed, without importing the world."""
    return importlib.util.find_spec("timecopilot") is not None


def validate_names(names: list[str]) -> None:
    """Reject unknown model names before any expensive work starts."""
    for name in names:
        if name in MODEL_REGISTRY:
            continue
        close = [k for k in MODEL_REGISTRY if k.lower() == name.lower()]
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise ValueError(
            f"Unknown model '{name}'.{hint} Call tsf_list_models for the names "
            "available on this installation."
        )


def require_timecopilot(names: list[str]) -> None:
    """Raise an actionable error if these models need an extra that is absent."""
    validate_names(names)
    if timecopilot_available():
        return
    raise ValueError(
        f"{names} need the optional '{EXTRA}' extra, which is not installed. "
        f"Install it with `uv pip install 'tslab-mcp[{EXTRA}]'` (this pulls "
        "TimeCopilot and PyTorch, roughly 2 GB), or use a statistical model "
        f"that works right now: {_cheap_alternatives()}."
    )


def resolve_timecopilot(names: list[str]) -> list[Any]:
    """Instantiate model objects from TimeCopilot, preserving order.

    Statistical names are resolved through TimeCopilot's own wrappers here,
    because this path only runs when the request mixes families and the whole
    batch has to go through one forecaster.
    """
    from timecopilot.models import stats as tc_stats

    resolved: list[Any] = []
    for name in names:
        if name in STATS_MODELS:
            resolved.append(getattr(tc_stats, STATS_MODELS[name])())
            continue
        module_path, attribute = TIMECOPILOT_MODELS[name]
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ValueError(
                f"Model '{name}' could not be imported: {exc}. This usually "
                "means an unsupported Python version or a broken optional "
                f"dependency. Models that work right now: {_cheap_alternatives()}."
            ) from None
        try:
            resolved.append(getattr(module, attribute)())
        except AttributeError:
            raise ValueError(
                f"Model '{name}' is registered as {module_path}.{attribute}, "
                "but that attribute does not exist in the installed "
                "timecopilot. Call tsf_list_models to see what resolved."
            ) from None
    return resolved


def _probe_statistical() -> tuple[list[str], dict[str, str]]:
    available: list[str] = []
    unavailable: dict[str, str] = {}
    try:
        from statsforecast import models as sf_models
    except Exception as exc:  # noqa: BLE001 -- probing is the whole point
        return [], {name: f"{type(exc).__name__}: {exc}" for name in STATS_MODELS}
    for name, attribute in STATS_MODELS.items():
        if hasattr(sf_models, attribute):
            available.append(name)
        else:
            unavailable[name] = (
                f"AttributeError: statsforecast.models has no '{attribute}'"
            )
    return available, unavailable


def _probe_timecopilot() -> tuple[list[str], dict[str, str]]:
    if not timecopilot_available():
        reason = (
            f"not installed; add the '{EXTRA}' extra "
            f"(`uv pip install 'tslab-mcp[{EXTRA}]'`)"
        )
        return [], dict.fromkeys(TIMECOPILOT_MODELS, reason)

    available: list[str] = []
    unavailable: dict[str, str] = {}
    for name, (module_path, attribute) in TIMECOPILOT_MODELS.items():
        try:
            module = importlib.import_module(module_path)
            getattr(module, attribute)
        except Exception as exc:  # noqa: BLE001 -- report, never raise
            unavailable[name] = f"{type(exc).__name__}: {exc}"
        else:
            available.append(name)
    return available, unavailable


def probe(family: str = "all", include_unavailable: bool = True) -> dict[str, Any]:
    """Report which models actually resolve here, and why the rest do not.

    Probing the statistical family is instant. Probing the foundation family
    imports TimeCopilot when it is installed, which takes about 30 seconds the
    first time — pass ``family='statistical'`` to avoid that entirely.
    """
    statistical: list[str] = []
    foundation: list[str] = []
    unavailable: dict[str, str] = {}

    if family in ("all", "statistical"):
        statistical, missing = _probe_statistical()
        unavailable.update(missing)
    if family in ("all", "foundation"):
        foundation, missing = _probe_timecopilot()
        unavailable.update(missing)

    available = sorted(statistical + foundation)
    result: dict[str, Any] = {
        "available": available,
        "n_available": len(available),
        "statistical": sorted(statistical),
        "foundation": sorted(foundation),
        "backend": "statsforecast",
        "foundation_extra_installed": timecopilot_available(),
        "notes": [
            "Statistical models run in seconds, need no weights, and are always "
            "installed.",
            f"Foundation models need the '{EXTRA}' extra; they download "
            "hundreds of MB on first call and are slow on CPU.",
            "TimeGPT calls the Nixtla API and needs NIXTLA_API_KEY; every "
            "other model runs locally.",
        ],
    }
    if include_unavailable:
        result["unavailable"] = dict(sorted(unavailable.items()))
        result["n_unavailable"] = len(unavailable)
    return result
