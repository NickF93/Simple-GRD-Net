import os

import pytest

from grdnet.cli import main
from grdnet.core.determinism import CUBLAS_WORKSPACE_DEFAULT, CUBLAS_WORKSPACE_ENV
from grdnet.core.exceptions import ConfigurationError


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
    monkeypatch.setattr("grdnet.cli.enforce_runtime_versions", lambda: None)
    monkeypatch.setattr(
        f"grdnet.cli.{target}",
        lambda *args: expected,
    )
    assert main(argv) == expected


def test_cli_runs_runtime_guard(monkeypatch) -> None:
    calls = {"guard": 0}
    monkeypatch.setattr(
        "grdnet.cli.enforce_runtime_versions",
        lambda: calls.__setitem__("guard", calls["guard"] + 1),
    )
    monkeypatch.setattr("grdnet.cli.run_validate_config", lambda _cfg: 0)
    assert main(["validate-config", "-c", "a.yaml"]) == 0
    assert calls["guard"] == 1


def test_cli_sets_cublas_workspace_for_deterministic_cuda(
    monkeypatch,
    tmp_path,
) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        (
            "backend:\n"
            "  name: pytorch\n"
            "  device: auto\n"
            "system:\n"
            "  deterministic: true\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv(CUBLAS_WORKSPACE_ENV, raising=False)
    monkeypatch.setattr("grdnet.cli.enforce_runtime_versions", lambda: None)
    monkeypatch.setattr("grdnet.cli.run_train", lambda _cfg: 0)

    assert main(["train", "-c", str(cfg)]) == 0
    assert os.environ.get(CUBLAS_WORKSPACE_ENV) == CUBLAS_WORKSPACE_DEFAULT


def test_cli_rejects_invalid_cublas_workspace_for_deterministic_cuda(
    monkeypatch,
    tmp_path,
) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        (
            "backend:\n"
            "  name: pytorch\n"
            "  device: cuda\n"
            "system:\n"
            "  deterministic: true\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv(CUBLAS_WORKSPACE_ENV, "invalid")
    monkeypatch.setattr("grdnet.cli.enforce_runtime_versions", lambda: None)
    monkeypatch.setattr("grdnet.cli.run_train", lambda _cfg: 0)

    with pytest.raises(ConfigurationError, match=CUBLAS_WORKSPACE_ENV):
        _ = main(["train", "-c", str(cfg)])
