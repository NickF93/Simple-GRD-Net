"""Determinism runtime constants and helpers."""

from __future__ import annotations

CUBLAS_WORKSPACE_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_WORKSPACE_DEFAULT = ":4096:8"
CUBLAS_WORKSPACE_ALLOWED = frozenset({":4096:8", ":16:8"})


def is_valid_cublas_workspace(value: str | None) -> bool:
    """Return True when CuBLAS workspace config is one of NVIDIA-supported values."""
    return value in CUBLAS_WORKSPACE_ALLOWED

