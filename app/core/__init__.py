"""
EmoSense Backend API - Core Package

This package contains core functionality including security, authentication,
exceptions, and other shared utilities.
"""

from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    CustomHTTPException,
    DatabaseError,
    FileProcessingError,
    ModelProcessingError,
    ResourceNotFoundError,
    ValidationError,
)
from .security import (
    create_access_token,
    create_password_hash,
    create_refresh_token,
    create_user_tokens,
    verify_password,
    verify_token,
)

__all__ = [
    # Exceptions
    "CustomHTTPException",
    "AuthenticationError", 
    "AuthorizationError",
    "ResourceNotFoundError",
    "ValidationError",
    "FileProcessingError",
    "ModelProcessingError",
    "DatabaseError",
    # Security
    "create_password_hash",
    "verify_password",
    "create_access_token",
    "create_refresh_token",
    "verify_token",
    "create_user_tokens",
]
