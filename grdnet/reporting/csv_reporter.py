"""CSV reporter for training/evaluation artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

from grdnet.config.schema import ExperimentConfig
from grdnet.reporting.base import Reporter


class CsvReporter(Reporter):
    """CSV-backed reporter with deterministic columns."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        self._cfg = cfg
        self._output_dir = cfg.training.output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self._output_dir / cfg.reporting.csv_metrics_filename
        self._predictions_path = (
            self._output_dir / cfg.reporting.csv_predictions_filename
        )

        self._metrics_header_written = self._metrics_path.exists()
        # Prediction output is command-scoped: truncate once at startup, append later.
        if self._predictions_path.exists():
            self._predictions_path.unlink()
        self._predictions_header_written = False

    @staticmethod
    def _write_rows(
        path: Path,
        rows: list[dict[str, str | float | int]],
        *,
        header_written: bool,
    ) -> bool:
        if not rows:
            return header_written

        fieldnames = list(rows[0].keys())
        mode = "a" if header_written else "w"
        with path.open(mode, encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not header_written:
                writer.writeheader()
            writer.writerows(rows)
        return True

    def log_epoch(self, *, epoch: int, split: str, metrics: dict[str, float]) -> None:
        """Append one epoch metric row."""
        row = {"kind": "epoch", "epoch": epoch, "split": split, **metrics}
        self._metrics_header_written = self._write_rows(
            self._metrics_path,
            [row],
            header_written=self._metrics_header_written,
        )

    def log_evaluation(self, metrics: dict[str, float]) -> None:
        """Append one evaluation metric row."""
        row = {"kind": "evaluation", "epoch": -1, "split": "test", **metrics}
        self._metrics_header_written = self._write_rows(
            self._metrics_path,
            [row],
            header_written=self._metrics_header_written,
        )

    def write_predictions(self, rows: list[dict[str, str | float | int]]) -> None:
        """Append one chunk of prediction rows."""
        self._predictions_header_written = self._write_rows(
            self._predictions_path,
            rows,
            header_written=self._predictions_header_written,
        )
