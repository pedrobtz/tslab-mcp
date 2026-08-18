"""The model registry: friendly name -> import path, resolved lazily.

Design invariant I3: the environment is discovered, not assumed. Importing
``timecopilot`` costs tens of seconds and pulls in torch, so nothing here is
imported at module load. :func:`probe` reports what actually resolved on *this*
machine; :func:`resolve` turns a failure into a message naming the alternatives
that work right now.
"""

from __future__ import annotations

import importlib
from typing import Any

#: Friendly name -> (module path, attribute). Verified against timecopilot
#: 0.0.30; ``probe`` exists precisely so drift fails softly.
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    # statistical -- cheap, no weights, always available
    "SeasonalNaive": ("timecopilot.models.stats", "SeasonalNaive"),
    "HistoricAverage": ("timecopilot.models.stats", "HistoricAverage"),
    "AutoARIMA": ("timecopilot.models.stats", "AutoARIMA"),
    "AutoETS": ("timecopilot.models.stats", "AutoETS"),
    "AutoCES": ("timecopilot.models.stats", "AutoCES"),
    "Theta": ("timecopilot.models.stats", "Theta"),
    "DynamicOptimizedTheta": ("timecopilot.models.stats", "DynamicOptimizedTheta"),
    "ADIDA": ("timecopilot.models.stats", "ADIDA"),
    "IMAPA": ("timecopilot.models.stats", "IMAPA"),
    "CrostonClassic": ("timecopilot.models.stats", "CrostonClassic"),
    "ZeroModel": ("timecopilot.models.stats", "ZeroModel"),
    "Prophet": ("timecopilot.models.prophet", "Prophet"),
    # foundation -- pretrained, download weights on first call
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

#: Models that need no pretrained weights and run in seconds.
STATISTICAL_MODELS = frozenset(
    name
    for name, (module, _) in MODEL_REGISTRY.items()
    if "foundation" not in module
)

#: Models that download weights and may need a GPU to be practical.
FOUNDATION_MODELS = frozenset(MODEL_REGISTRY) - STATISTICAL_MODELS

#: Models that reach an external API rather than running locally.
REMOTE_MODELS = frozenset({"TimeGPT"})


def family_of(name: str) -> str:
    """``'statistical'`` or ``'foundation'`` for a registered model name."""
    return "statistical" if name in STATISTICAL_MODELS else "foundation"


def _cheap_alternatives() -> str:
    return ", ".join(sorted(STATISTICAL_MODELS))


def _import_model(name: str) -> type[Any]:
    """Import and return the model class, or raise an actionable ValueError."""
    if name not in MODEL_REGISTRY:
        close = [k for k in MODEL_REGISTRY if k.lower() == name.lower()]
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise ValueError(
            f"Unknown model '{name}'.{hint} Call tsf_list_models for the names "
            "available on this installation."
        )
    module_path, attribute = MODEL_REGISTRY[name]
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ValueError(
            f"Model '{name}' could not be imported: {exc}. TimeCopilot ships "
            "every model in the base install, so this usually means an "
            "unsupported Python version or a broken optional dependency. "
            f"Models that work right now: {_cheap_alternatives()}."
        ) from None
    try:
        return getattr(module, attribute)  # type: ignore[no-any-return]
    except AttributeError:
        raise ValueError(
            f"Model '{name}' is registered as {module_path}.{attribute}, but "
            "that attribute does not exist in the installed timecopilot. The "
            "registry is out of date; call tsf_list_models to see what "
            "resolved."
        ) from None


def resolve(names: list[str]) -> list[Any]:
    """Instantiate model objects from friendly names, preserving order."""
    return [_import_model(name)() for name in names]


def forecaster(names: list[str]) -> Any:
    """Build a ``TimeCopilotForecaster`` over the named models.

    Imported here rather than at module scope: the import alone costs ~30s and
    must never run during server startup.
    """
    from timecopilot import TimeCopilotForecaster

    return TimeCopilotForecaster(models=resolve(names))


def probe(family: str = "all", include_unavailable: bool = True) -> dict[str, Any]:
    """Attempt every registered model, partitioned into available/unavailable.

    The reason string is the exception *message*, not just its type -- some
    models fail with a genuinely actionable explanation (e.g. "requires Python
    < 3.13") that a bare type name would throw away.
    """
    selected = {
        name: spec
        for name, spec in MODEL_REGISTRY.items()
        if family == "all" or family_of(name) == family
    }

    available: list[str] = []
    unavailable: dict[str, str] = {}
    for name, (module_path, attribute) in selected.items():
        try:
            module = importlib.import_module(module_path)
            getattr(module, attribute)
        except Exception as exc:  # noqa: BLE001 -- probing is the whole point
            unavailable[name] = f"{type(exc).__name__}: {exc}"
        else:
            available.append(name)

    result: dict[str, Any] = {
        "available": sorted(available),
        "n_available": len(available),
        "statistical": sorted(n for n in available if n in STATISTICAL_MODELS),
        "foundation": sorted(n for n in available if n in FOUNDATION_MODELS),
        "notes": [
            "Statistical models run in seconds and need no weights.",
            "Foundation models download hundreds of MB on first call and are "
            "much slower on CPU.",
            "TimeGPT calls the Nixtla API and needs NIXTLA_API_KEY; every "
            "other model runs locally.",
        ],
    }
    if include_unavailable:
        result["unavailable"] = dict(sorted(unavailable.items()))
        result["n_unavailable"] = len(unavailable)
    return result
