"""TensorFlow/Keras placeholders for future backend parity."""

from __future__ import annotations


class _TensorFlowScaffold:
    """Base placeholder class with deterministic error messages."""

    _NAME = "TensorFlow scaffold"

    def __init__(self) -> None:
        self._message = (
            f"{self._NAME} is a placeholder in v1. "
            "Use backend.name='pytorch' for full implementation."
        )

    def __call__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(self._message)


class TensorFlowGeneratorScaffold(_TensorFlowScaffold):
    """Generator placeholder."""

    _NAME = "TensorFlow Generator"


class TensorFlowDiscriminatorScaffold(_TensorFlowScaffold):
    """Discriminator placeholder."""

    _NAME = "TensorFlow Discriminator"


class TensorFlowSegmentatorScaffold(_TensorFlowScaffold):
    """Segmentator placeholder."""

    _NAME = "TensorFlow Segmentator"
