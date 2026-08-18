"""Run records and the exported manifest -- the artifact of record.

Design invariant I4: the session's prose is commentary; this file is what
someone reruns in six months. Every mutation of a handle appends a record, and
:func:`build` pins those records to the input hash and the installed package
versions.
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any

from . import __version__
from .registry import SeriesHandle

MANIFEST_VERSION = 1

#: Packages whose versions change the numbers, so they are pinned in the
#: manifest. Absent packages are simply omitted.
PINNED_PACKAGES = (
    "tslab-mcp",
    "timecopilot",
    "statsforecast",
    "utilsforecast",
    "mlforecast",
    "neuralforecast",
    "pandas",
    "numpy",
    "torch",
)


def utc_now() -> str:
    """Current time as an ISO-8601 UTC string. Every timestamp uses this."""
    return datetime.now(timezone.utc).isoformat()


def environment() -> dict[str, str]:
    """Versions of everything that can move a forecast, plus the interpreter."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "tslab_mcp": __version__,
    }
    for package in PINNED_PACKAGES:
        try:
            versions[package.replace("-", "_")] = package_version(package)
        except PackageNotFoundError:
            continue
    return versions


def record(kind: str, **fields: Any) -> dict[str, Any]:
    """Build a run record. ``at`` and ``kind`` are always present."""
    entry: dict[str, Any] = {"kind": kind, "at": utc_now()}
    entry.update(fields)
    return entry


def build(series: SeriesHandle, note: str | None = None) -> dict[str, Any]:
    """Assemble the full manifest for a handle.

    Sufficient on its own to reconstruct the run: source path and hash,
    inferred frequency, every call with its arguments and artifact paths, and
    the pinned environment.
    """
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": utc_now(),
        "input": {
            "source": series.source,
            "sha256": series.sha256,
            "freq": series.freq,
            "loaded_at": series.loaded_at,
            "n_series": series.n_series,
            "n_obs": series.n_obs,
        },
        "runs": series.runs,
        "environment": environment(),
        "note": note,
    }
