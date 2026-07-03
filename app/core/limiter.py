"""
Rate limiting for EmoSense Backend API.

Provides a shared slowapi ``Limiter`` keyed by client IP, with a default limit
derived from settings (RATE_LIMIT_REQUESTS per RATE_LIMIT_WINDOW seconds).
Wired into the app in ``app.main``.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings

settings = get_settings()

# e.g. "100/60 seconds"
_default_limit = f"{settings.RATE_LIMIT_REQUESTS}/{settings.RATE_LIMIT_WINDOW} seconds"

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[_default_limit],
    # In-memory storage by default; point at Redis in production for multi-worker
    # correctness via RATE_LIMIT_STORAGE_URI.
    storage_uri=settings.RATE_LIMIT_STORAGE_URI or None,
)
