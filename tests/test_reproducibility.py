import pytest
import torch

from grdnet.core.determinism import CUBLAS_WORKSPACE_DEFAULT, CUBLAS_WORKSPACE_ENV
from grdnet.core.exceptions import ConfigurationError
from grdnet.core.reproducibility import validate_determinism_runtime


def test_validate_determinism_runtime_ignores_cpu(monkeypatch) -> None:
    monkeypatch.delenv(CUBLAS_WORKSPACE_ENV, raising=False)
    validate_determinism_runtime(
        deterministic=True,
        backend_name="pytorch",
        backend_device=torch.device("cpu"),
    )


def test_validate_determinism_runtime_requires_workspace_for_cuda(monkeypatch) -> None:
    monkeypatch.delenv(CUBLAS_WORKSPACE_ENV, raising=False)
    with pytest.raises(ConfigurationError, match=CUBLAS_WORKSPACE_ENV):
        validate_determinism_runtime(
            deterministic=True,
            backend_name="pytorch",
            backend_device=torch.device("cuda"),
        )


def test_validate_determinism_runtime_accepts_allowed_workspace(monkeypatch) -> None:
    monkeypatch.setenv(CUBLAS_WORKSPACE_ENV, CUBLAS_WORKSPACE_DEFAULT)
    validate_determinism_runtime(
        deterministic=True,
        backend_name="pytorch",
        backend_device=torch.device("cuda"),
    )
