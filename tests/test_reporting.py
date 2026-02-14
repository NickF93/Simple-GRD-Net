import json
from pathlib import Path

from grdnet.config.loader import load_experiment_config
from grdnet.reporting.console import ConsoleReporter
from grdnet.reporting.csv_reporter import CsvReporter


def test_csv_reporter_writes_metrics_and_predictions(tmp_path: Path) -> None:
    cfg = load_experiment_config("configs/profiles/deepindustrial_sn_2026.yaml")
    cfg.training.output_dir = tmp_path / "reports"
    cfg.reporting.csv_metrics_filename = "metrics.csv"
    cfg.reporting.csv_predictions_filename = "predictions.csv"

    reporter = CsvReporter(cfg)
    reporter.log_epoch(epoch=1, split="train", metrics={"loss": 1.0})
    reporter.log_evaluation(metrics={"accuracy": 0.5})
    reporter.write_predictions(
        [{"path": "a.png", "score": 0.1, "patch_prediction": 0, "image_prediction": 0}]
    )

    metrics_path = cfg.training.output_dir / cfg.reporting.csv_metrics_filename
    preds_path = cfg.training.output_dir / cfg.reporting.csv_predictions_filename
    assert metrics_path.exists()
    assert preds_path.exists()
    assert "kind,epoch,split" in metrics_path.read_text(encoding="utf-8")
    assert "path,score,patch_prediction,image_prediction" in preds_path.read_text(
        encoding="utf-8"
    )


def test_console_reporter_emits_json_payloads(monkeypatch) -> None:
    payloads: list[str] = []
    monkeypatch.setattr(
        "grdnet.reporting.console.LOGGER.info",
        lambda message: payloads.append(str(message)),
    )

    reporter = ConsoleReporter()
    reporter.log_epoch(epoch=1, split="train", metrics={"loss": 1.0})
    reporter.log_evaluation(metrics={"accuracy": 0.9})
    reporter.write_predictions(rows=[{"path": "x.png"}])

    decoded = [json.loads(raw) for raw in payloads]
    assert decoded[0]["event"] == "epoch_summary"
    assert decoded[1]["event"] == "evaluation_summary"
    assert decoded[2]["event"] == "prediction_summary"


def test_csv_reporter_truncates_existing_predictions_and_ignores_empty_rows(
    tmp_path: Path,
) -> None:
    cfg = load_experiment_config("configs/profiles/deepindustrial_sn_2026.yaml")
    cfg.training.output_dir = tmp_path / "reports"
    cfg.reporting.csv_metrics_filename = "metrics.csv"
    cfg.reporting.csv_predictions_filename = "predictions.csv"
    cfg.training.output_dir.mkdir(parents=True, exist_ok=True)
    existing_predictions = (
        cfg.training.output_dir / cfg.reporting.csv_predictions_filename
    )
    existing_predictions.write_text("stale,data\n", encoding="utf-8")

    reporter = CsvReporter(cfg)
    assert not existing_predictions.exists()

    reporter.write_predictions([])
    assert not existing_predictions.exists()
