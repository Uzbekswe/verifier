"""
FastAPI dependency functions and shared async utilities.
"""
import asyncio
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, TypeVar

from fastapi import Depends, UploadFile, File
from PIL import Image

from app.core.config import settings
from app.core.exceptions import ImageDecodeError, ImageTooLargeError, InvalidMimeTypeError

logger = logging.getLogger(__name__)

# Dedicated thread pool for blocking CV operations (MediaPipe, OpenCV, DeepFace).
# These are CPU-bound and must not run on the event loop thread.
cv_executor = ThreadPoolExecutor(
    max_workers=settings.CV_THREAD_POOL_SIZE,
    thread_name_prefix="cv-worker",
)

T = TypeVar("T")

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


async def run_in_cv_executor(fn: Callable[..., T], *args) -> T:
    """Run a blocking CV function in the dedicated thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(cv_executor, partial(fn, *args))


async def validated_image_bytes(
    file: UploadFile = File(..., description="Selfie image (JPEG/PNG/WebP)"),
) -> bytes:
    """
    FastAPI dependency that validates MIME type, file size, and image dimensions.
    Returns raw bytes ready for CV processing.
    """
    content_type = file.content_type or ""
    if content_type not in ALLOWED_MIME_TYPES:
        raise InvalidMimeTypeError(received=content_type, allowed=ALLOWED_MIME_TYPES)

    file_bytes = await file.read()

    if len(file_bytes) > settings.MAX_IMAGE_BYTES:
        raise ImageTooLargeError(max_mb=settings.MAX_IMAGE_SIZE_MB)

    # Decompression bomb guard — check pixel dimensions without fully decoding
    try:
        img = Image.open(io.BytesIO(file_bytes))
        w, h = img.size
        if w > settings.MAX_IMAGE_DIMENSION or h > settings.MAX_IMAGE_DIMENSION:
            raise ImageTooLargeError(
                detail=f"Image dimensions {w}×{h} exceed maximum {settings.MAX_IMAGE_DIMENSION}px per side."
            )
    except ImageTooLargeError:
        raise
    except Exception:
        raise ImageDecodeError()

    return file_bytes
