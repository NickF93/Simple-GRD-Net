"""Runtime version guards for Python and Torch."""

from __future__ import annotations

import re
import sys

import torch

from grdnet.core.exceptions import RuntimeCompatibilityError

MIN_PYTHON = (3, 11)
MIN_TORCH = (2, 7, 0)


def _parse_torch_version(raw: str) -> tuple[int, int, int]:
    """Extract semantic version triple from a Torch version string."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", raw)
    if match is None:
        raise RuntimeCompatibilityError(
            f"Unable to parse torch.__version__ value '{raw}'."
        )
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def enforce_runtime_versions() -> None:
    """Fail fast when runtime versions do not meet project minimums."""
    if sys.version_info < MIN_PYTHON:
        current = f"{sys.version_info.major}.{sys.version_info.minor}"
        required = f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
        raise RuntimeCompatibilityError(
            f"Unsupported Python runtime {current}. Minimum required is {required}."
        )

    torch_version = _parse_torch_version(torch.__version__)
    if torch_version < MIN_TORCH:
        current = ".".join(str(part) for part in torch_version)
        required = ".".join(str(part) for part in MIN_TORCH)
        raise RuntimeCompatibilityError(
            f"Unsupported Torch runtime {current}. Minimum required is {required}."
        )
