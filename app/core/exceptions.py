"""
Custom Exception Classes for EmoSense Backend API

Defines custom HTTP exceptions and error handling utilities
for consistent error responses across the application.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import HTTPException, status


class CustomHTTPException(HTTPException):
    """
    Custom HTTP exception with enhanced error details.
    
    Extends FastAPI's HTTPException to include additional context
    and structured error information for better client handling.
    """
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize custom HTTP exception.
        
        Args:
            status_code: HTTP status code
            detail: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error context
            headers: Optional HTTP headers
        """
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class ValidationError(CustomHTTPException):
    """Exception for input validation errors."""
    
    def __init__(
        self,
        detail: str = "Validation failed",
        field_errors: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code="VALIDATION_ERROR",
            details={"field_errors": field_errors or {}},
        )


class AuthenticationError(CustomHTTPException):
    """Exception for authentication failures."""
    
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code="AUTHENTICATION_ERROR",
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthorizationError(CustomHTTPException):
    """Exception for authorization failures."""
    
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code="AUTHORIZATION_ERROR",
        )


class ResourceNotFoundError(CustomHTTPException):
    """Exception for resource not found errors."""
    
    def __init__(self, resource: str, identifier: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} not found",
            error_code="RESOURCE_NOT_FOUND",
            details={"resource": resource, "identifier": identifier},
        )


class ResourceConflictError(CustomHTTPException):
    """Exception for resource conflicts (e.g., duplicate creation)."""
    
    def __init__(self, detail: str, resource: str):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            error_code="RESOURCE_CONFLICT",
            details={"resource": resource},
        )


class FileProcessingError(CustomHTTPException):
    """Exception for file processing errors."""
    
    def __init__(self, detail: str, file_name: Optional[str] = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code="FILE_PROCESSING_ERROR",
            details={"file_name": file_name},
        )


class ModelProcessingError(CustomHTTPException):
    """Exception for machine learning model processing errors."""
    
    def __init__(self, detail: str, model_name: Optional[str] = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="MODEL_PROCESSING_ERROR",
            details={"model_name": model_name},
        )


class ExternalServiceError(CustomHTTPException):
    """Exception for external service errors."""
    
    def __init__(self, detail: str, service: str):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            error_code="EXTERNAL_SERVICE_ERROR",
            details={"service": service},
        )


class RateLimitError(CustomHTTPException):
    """Exception for rate limiting violations."""
    
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            error_code="RATE_LIMIT_EXCEEDED",
            headers={"Retry-After": "60"},
        )


class DatabaseError(CustomHTTPException):
    """Exception for database operation errors."""
    
    def __init__(self, detail: str = "Database operation failed"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="DATABASE_ERROR",
        )


class CacheError(CustomHTTPException):
    """Exception for cache operation errors."""
    
    def __init__(self, detail: str = "Cache operation failed"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="CACHE_ERROR",
        )


# Convenience functions for common error scenarios
def create_validation_error(field: str, message: str) -> ValidationError:
    """Create a validation error for a specific field."""
    return ValidationError(
        detail=f"Validation failed for field '{field}'",
        field_errors={field: message}
    )


def create_not_found_error(resource_type: str, resource_id: str) -> ResourceNotFoundError:
    """Create a not found error for a specific resource."""
    return ResourceNotFoundError(resource_type, resource_id)


def create_duplicate_error(resource_type: str, field: str) -> ResourceConflictError:
    """Create a conflict error for duplicate resources."""
    return ResourceConflictError(
        detail=f"{resource_type} with this {field} already exists",
        resource=resource_type
    )


def create_file_error(message: str, filename: Optional[str] = None) -> FileProcessingError:
    """Create a file processing error."""
    return FileProcessingError(detail=message, file_name=filename)


def create_model_error(message: str, model: Optional[str] = None) -> ModelProcessingError:
    """Create a model processing error."""
    return ModelProcessingError(detail=message, model_name=model)
