"""Filesystem artifacts: output directory, parquet/JSON writing, hashing.

Bulk output never crosses the tool boundary (design invariant I1), so every
tool that produces a frame writes it here and returns the path instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

#: Environment variable overriding the artifact root.
HOME_ENV_VAR = "TSLAB_MCP_HOME"

#: Default artifact root when the environment variable is unset.
DEFAULT_HOME = Path.home() / ".tslab-mcp"

_HASH_CHUNK = 1 << 20


def output_dir() -> Path:
    """Return (and create) the directory run artifacts are written to.

    Resolved on every call rather than at import, so a client that sets
    ``TSLAB_MCP_HOME`` after the module loads -- and any test that
    monkeypatches it -- sees the change.
    """
    root = os.environ.get(HOME_ENV_VAR)
    base = Path(root).expanduser() if root else DEFAULT_HOME
    path = base / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _unique_path(prefix: str, handle: str, suffix: str) -> Path:
    """Build a collision-free path under the output directory."""
    directory = output_dir()
    safe_handle = "".join(c if c.isalnum() or c in "-_" else "_" for c in handle)
    while True:
        stem = f"{prefix}_{safe_handle}_{uuid.uuid4().hex[:8]}"
        path = directory / f"{stem}{suffix}"
        if not path.exists():
            return path


def write_parquet(df: pd.DataFrame, prefix: str, handle: str) -> Path:
    """Write ``df`` to a fresh parquet file and return its path."""
    path = _unique_path(prefix, handle, ".parquet")
    df.to_parquet(path, index=False)
    return path


def write_json(payload: Any, prefix: str, handle: str) -> Path:
    """Write ``payload`` to a fresh JSON file and return its path."""
    path = _unique_path(prefix, handle, ".json")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_text(content: str, prefix: str, handle: str, suffix: str) -> Path:
    """Write ``content`` to a fresh file with ``suffix`` and return its path."""
    path = _unique_path(prefix, handle, suffix)
    path.write_text(content, encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()
