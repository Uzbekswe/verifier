"""
Rate limiting via slowapi.
The limiter instance is imported by routers that need per-route limits.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.RATE_LIMIT_GLOBAL],
    storage_uri=settings.RATE_LIMIT_STORAGE_URI,
)
