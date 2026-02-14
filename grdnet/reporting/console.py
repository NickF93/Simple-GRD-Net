"""Console reporter implementation."""

from __future__ import annotations

import json
import logging

from grdnet.reporting.base import Reporter

LOGGER = logging.getLogger(__name__)


class ConsoleReporter(Reporter):
    """Structured console reporting."""

    def log_epoch(self, *, epoch: int, split: str, metrics: dict[str, float]) -> None:
        """Emit one epoch summary as structured JSON."""
        payload = {"event": "epoch_summary", "epoch": epoch, "split": split, **metrics}
        LOGGER.info(json.dumps(payload, sort_keys=True))

    def log_evaluation(self, metrics: dict[str, float]) -> None:
        """Emit one evaluation summary as structured JSON."""
        payload = {"event": "evaluation_summary", **metrics}
        LOGGER.info(json.dumps(payload, sort_keys=True))

    def write_predictions(self, rows: list[dict[str, str | float | int]]) -> None:
        """Emit a compact prediction write summary."""
        payload = {
            "event": "prediction_summary",
            "rows": len(rows),
        }
        LOGGER.info(json.dumps(payload, sort_keys=True))
