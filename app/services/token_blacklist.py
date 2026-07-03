"""
Token blacklist service for EmoSense Backend API.

Backs real logout / token revocation with Redis. Each token carries a unique
``jti`` claim; on logout the jti is stored in Redis with a TTL equal to the
token's remaining lifetime, so it is rejected until it would have expired anyway.

Redis is treated as best-effort: if it is unavailable, blacklisting is a no-op
and ``is_blacklisted`` returns ``False`` (fail-open) so authentication keeps
working in environments without Redis (e.g. local dev).
"""

import time
from typing import Optional

import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

_KEY_PREFIX = "revoked_jti:"
_redis = None
_redis_unavailable = False


def _disable(exc: Exception) -> None:
    """Disable the blacklist after a Redis failure, logging once."""
    global _redis, _redis_unavailable
    if not _redis_unavailable:
        logger.warning("Redis unavailable; token blacklist disabled", error=str(exc))
    _redis = None
    _redis_unavailable = True


async def _get_redis():
    """Lazily create a shared async Redis client, or None if unavailable."""
    global _redis, _redis_unavailable
    if _redis is not None:
        return _redis
    if _redis_unavailable:
        return None
    try:
        import redis.asyncio as redis

        _redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
        return _redis
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Redis unavailable; token blacklist disabled", error=str(exc))
        _redis_unavailable = True
        return None


async def blacklist_token(jti: Optional[str], exp: Optional[int]) -> None:
    """Revoke a token by its jti until its original expiry.

    Args:
        jti: The token's unique id claim.
        exp: The token's expiry as a POSIX timestamp.
    """
    if not jti:
        return

    client = await _get_redis()
    if client is None:
        return

    # TTL = remaining lifetime (at least 1s); expired tokens need no entry.
    ttl = int(exp - time.time()) if exp else settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    if ttl <= 0:
        return

    try:
        await client.set(f"{_KEY_PREFIX}{jti}", "1", ex=ttl)
    except Exception as exc:
        _disable(exc)


async def is_blacklisted(jti: Optional[str]) -> bool:
    """Return True if the given jti has been revoked (False if Redis is down)."""
    if not jti:
        return False

    client = await _get_redis()
    if client is None:
        return False

    try:
        return await client.exists(f"{_KEY_PREFIX}{jti}") == 1
    except Exception as exc:
        _disable(exc)
        return False
