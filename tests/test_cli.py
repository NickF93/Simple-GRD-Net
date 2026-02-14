import pytest

from grdnet.cli import main


@pytest.mark.parametrize(
    ("argv", "target", "expected"),
    [
        (["validate-config", "-c", "a.yaml"], "run_validate_config", 11),
        (["train", "-c", "a.yaml"], "run_train", 12),
        (["calibrate", "-c", "a.yaml", "--checkpoint", "c.pt"], "run_calibrate", 13),
        (["eval", "-c", "a.yaml", "--checkpoint", "c.pt"], "run_evaluate", 14),
        (["infer", "-c", "a.yaml", "--checkpoint", "c.pt"], "run_infer", 15),
    ],
)
def test_cli_dispatches_commands(monkeypatch, argv, target: str, expected: int) -> None:
    monkeypatch.setattr(
        f"grdnet.cli.{target}",
        lambda *args: expected,
    )
    assert main(argv) == expected
