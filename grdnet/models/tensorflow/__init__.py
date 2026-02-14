"""TensorFlow scaffold namespace (API placeholders)."""

from grdnet.models.tensorflow.scaffold import (
    TensorFlowDiscriminatorScaffold,
    TensorFlowGeneratorScaffold,
    TensorFlowSegmentatorScaffold,
)

__all__ = [
    "TensorFlowGeneratorScaffold",
    "TensorFlowDiscriminatorScaffold",
    "TensorFlowSegmentatorScaffold",
]
