from grdnet.pipeline.runner import run_validate_config


def test_validate_config_does_not_use_print(monkeypatch) -> None:
    def _forbidden_print(*args, **kwargs):
        _ = args, kwargs
        raise AssertionError("print must not be called")

    monkeypatch.setattr("builtins.print", _forbidden_print)
    code = run_validate_config("configs/profiles/deepindustrial_sn_2026.yaml")
    assert code == 0
