import pytest

from grdnet.core import runtime
from grdnet.core.exceptions import RuntimeCompatibilityError


def test_parse_torch_version_handles_build_suffix() -> None:
    assert runtime._parse_torch_version("2.7.1+cu124") == (2, 7, 1)


def test_parse_torch_version_rejects_invalid_string() -> None:
    with pytest.raises(RuntimeCompatibilityError, match="Unable to parse"):
        _ = runtime._parse_torch_version("invalid")


def test_enforce_runtime_versions_rejects_old_python(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "MIN_PYTHON", (99, 0))
    with pytest.raises(RuntimeCompatibilityError, match="Unsupported Python runtime"):
        runtime.enforce_runtime_versions()


def test_enforce_runtime_versions_rejects_old_torch(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "MIN_PYTHON", (3, 0))
    monkeypatch.setattr(runtime.torch, "__version__", "2.6.9")
    with pytest.raises(RuntimeCompatibilityError, match="Unsupported Torch runtime"):
        runtime.enforce_runtime_versions()
