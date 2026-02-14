"""Custom exception hierarchy for GRD-Net."""


class GrdNetError(Exception):
    """Base exception for all package-specific failures."""


class ConfigurationError(GrdNetError):
    """Raised when configuration validation fails."""


class BackendNotAvailableError(GrdNetError):
    """Raised when an unavailable backend is requested."""


class DatasetContractError(GrdNetError):
    """Raised when a dataset does not satisfy the expected contract."""


class CheckpointError(GrdNetError):
    """Raised when checkpoint save/load contract validation fails."""


class RuntimeCompatibilityError(GrdNetError):
    """Raised when Python/Torch runtime versions are below project minimums."""
