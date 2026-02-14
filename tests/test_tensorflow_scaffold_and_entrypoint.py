import runpy

import pytest

from grdnet.backends.tensorflow_backend import TensorFlowScaffoldBackend
from grdnet.config.loader import load_experiment_config
from grdnet.models.tensorflow import (
    TensorFlowDiscriminatorScaffold,
    TensorFlowGeneratorScaffold,
    TensorFlowSegmentatorScaffold,
)


def test_tensorflow_model_scaffolds_raise() -> None:
    for cls in (
        TensorFlowGeneratorScaffold,
        TensorFlowDiscriminatorScaffold,
        TensorFlowSegmentatorScaffold,
    ):
        with pytest.raises(NotImplementedError):
            cls()()


def test_tensorflow_backend_methods_raise() -> None:
    cfg = load_experiment_config("configs/profiles/deepindustrial_sn_2026.yaml")
    backend = TensorFlowScaffoldBackend(cfg)

    with pytest.raises(NotImplementedError):
        backend.build_models()
    with pytest.raises(NotImplementedError):
        backend.build_optimizers(None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        backend.build_schedulers(None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        backend.train_step({})
    with pytest.raises(NotImplementedError):
        backend.eval_step({})
    with pytest.raises(NotImplementedError):
        backend.infer_step({})


def test_module_entrypoint_calls_cli_main(monkeypatch) -> None:
    monkeypatch.setattr("grdnet.cli.main", lambda: 0)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("grdnet.__main__", run_name="__main__")
    assert exc.value.code == 0
