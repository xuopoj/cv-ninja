"""Custom exceptions for CV Ninja."""


class CVNinjaError(Exception):
    """Base exception for all CV Ninja errors."""

    pass


class ConversionError(CVNinjaError):
    """Raised when annotation format conversion fails."""

    pass


class ValidationError(CVNinjaError):
    """Raised when input validation fails."""

    pass


class FormatNotSupportedError(CVNinjaError):
    """Raised when an annotation format is not supported."""

    pass
