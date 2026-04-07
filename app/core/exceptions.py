"""
Domain exception hierarchy.
Each exception maps to an HTTP status code and a machine-readable error_code.
Routers raise these; exception handlers in main.py convert them to HTTP responses.
"""
from typing import Any


class AppError(Exception):
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(self, message: str = "An unexpected error occurred", context: dict[str, Any] | None = None):
        self.message = message
        self.context = context or {}
        super().__init__(message)


class ImageDecodeError(AppError):
    status_code = 400
    error_code = "IMAGE_DECODE_ERROR"

    def __init__(self):
        super().__init__("Could not decode image. Ensure it is a valid JPEG, PNG, or WebP file.")


class ImageTooLargeError(AppError):
    status_code = 413
    error_code = "IMAGE_TOO_LARGE"

    def __init__(self, max_mb: int = 10, detail: str | None = None):
        super().__init__(
            detail or f"Image exceeds maximum size of {max_mb}MB.",
            context={"max_mb": max_mb},
        )


class InvalidMimeTypeError(AppError):
    status_code = 415
    error_code = "INVALID_MIME_TYPE"

    def __init__(self, received: str, allowed: set[str]):
        super().__init__(
            f"Unsupported file type '{received}'. Allowed: {', '.join(sorted(allowed))}.",
            context={"received": received, "allowed": sorted(allowed)},
        )


class UnknownChallengeError(AppError):
    status_code = 400
    error_code = "UNKNOWN_CHALLENGE"

    def __init__(self, challenge_type: str, valid: list[str]):
        super().__init__(
            f"Unknown challenge '{challenge_type}'.",
            context={"received": challenge_type, "valid": valid},
        )


class ModelNotReadyError(AppError):
    status_code = 503
    error_code = "MODEL_NOT_READY"

    def __init__(self, model_name: str):
        super().__init__(
            f"Model '{model_name}' is not loaded. Retry in a moment.",
            context={"model": model_name},
        )


class RateLimitError(AppError):
    status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"

    def __init__(self):
        super().__init__("Too many requests. Please slow down.")
