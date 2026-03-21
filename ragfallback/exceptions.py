"""Typed errors for reliability checks."""


class EmbeddingDimensionError(ValueError):
    """Raised when embedding width does not match the index / expected configuration."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
