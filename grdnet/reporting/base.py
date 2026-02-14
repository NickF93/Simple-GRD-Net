"""Reporting interface."""

from __future__ import annotations

from typing import Protocol


class Reporter(Protocol):
    """Reporter contract for metrics and predictions artifacts."""

    def log_epoch(self, *, epoch: int, split: str, metrics: dict[str, float]) -> None:
        """Record one epoch summary."""

    def log_evaluation(self, metrics: dict[str, float]) -> None:
        """Record one evaluation summary."""

    def write_predictions(self, rows: list[dict[str, str | float | int]]) -> None:
        """Persist prediction rows."""
