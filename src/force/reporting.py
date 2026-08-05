"""Shared machine-readable reporting helpers for FORCE experiments."""

from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import numpy as np


VALID_STATUSES = frozenset(
    {
        "completed",
        "skipped_external",
        "not_reproducible",
        "failed",
        "timeout",
        "oom",
    }
)


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not-installed"


def environment_metadata() -> dict[str, Any]:
    """Return the version and numerical-thread context for a run."""
    import os

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "numba": _version("numba"),
        "scikit_learn": _version("scikit-learn"),
        "pandas": _version("pandas"),
        "threads": {
            name: os.environ.get(name)
            for name in (
                "NUMBA_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    }


def command_line() -> list[str]:
    """Return the exact command-line vector for provenance."""
    return [sys.executable, *sys.argv]


def estimator_parameters(estimator: Any) -> dict[str, Any]:
    """Extract stable public tuning parameters from a benchmark estimator."""
    names = (
        "lambda_scale",
        "exact_cutover",
        "use_ter",
        "ter_max",
        "epsilon",
        "lam",
        "eps",
        "limits",
        "random_state",
        "support_fraction",
    )
    result: dict[str, Any] = {"class": type(estimator).__name__}
    for name in names:
        if hasattr(estimator, name):
            result[name] = getattr(estimator, name)
    return json_safe(result)


def json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict JSON values."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write strict, stable JSON with no NaN or Infinity tokens."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            json_safe(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
