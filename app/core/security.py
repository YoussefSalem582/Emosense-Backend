"""
Authentication and Security Module for EmoSense Backend API

Handles JWT token creation, validation, password hashing,
and authentication dependencies for FastAPI endpoints.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import get_settings
from app.core.exceptions import AuthenticationError
from app.database import get_db_session


# Get application settings
settings = get_settings()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer security scheme (optional)
security = HTTPBearer(auto_error=False)


def create_password_hash(password: str) -> str:
    """
    Create a hashed password.
    
    Args:
        password: Plain text password
        
    Returns:
        str: Hashed password
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Stored hashed password
        
    Returns:
        bool: True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in the token
        expires_delta: Optional custom expiration time
        
    Returns:
        str: Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """
    Create a JWT refresh token.
    
    Args:
        data: Data to encode in the token
        
    Returns:
        str: Encoded JWT refresh token
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token to verify
        token_type: Expected token type ("access" or "refresh")
        
    Returns:
        Dict[str, Any]: Decoded token payload
        
    Raises:
        AuthenticationError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        
        # Check if token type matches expected type
        if payload.get("type") != token_type:
            raise AuthenticationError(f"Invalid token type. Expected {token_type}")
        
        # Check if token has expired
        exp = payload.get("exp")
        if exp is None:
            raise AuthenticationError("Token missing expiration")
        
        if datetime.utcnow() > datetime.fromtimestamp(exp):
            raise AuthenticationError("Token has expired")
        
        return payload
        
    except JWTError as e:
        raise AuthenticationError(f"Invalid token: {str(e)}")


def extract_token_data(token: str) -> Dict[str, Any]:
    """
    Extract data from a token without verification.
    
    Args:
        token: JWT token
        
    Returns:
        Dict[str, Any]: Token payload (unverified)
        
    Note:
        This function does not verify the token signature.
        Use only for extracting metadata from trusted tokens.
    """
    try:
        return jwt.get_unverified_claims(token)
    except JWTError:
        return {}


def is_token_expired(token: str) -> bool:
    """
    Check if a token is expired without full verification.
    
    Args:
        token: JWT token to check
        
    Returns:
        bool: True if token is expired, False otherwise
    """
    try:
        payload = jwt.get_unverified_claims(token)
        exp = payload.get("exp")
        
        if exp is None:
            return True
        
        return datetime.utcnow() > datetime.fromtimestamp(exp)
        
    except JWTError:
        return True


def create_user_tokens(user_id: str, email: str, roles: list = None) -> Dict[str, str]:
    """
    Create both access and refresh tokens for a user.
    
    Args:
        user_id: User identifier
        email: User email address
        roles: User roles/permissions
        
    Returns:
        Dict[str, str]: Dictionary containing access_token and refresh_token
    """
    # Use 'sub' for subject (user ID) to follow JWT standards
    access_data = {
        "sub": user_id,
        "email": email,
        "roles": roles or [],
    }
    
    refresh_data = {
        "sub": user_id,  # Use 'sub' for consistency
    }
    
    access_token = create_access_token(access_data)
    refresh_token = create_refresh_token(refresh_data)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


class PasswordValidator:
    """
    Password strength validator.
    
    Provides methods to validate password strength and security requirements.
    """
    
    MIN_LENGTH = 8
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_DIGITS = True
    REQUIRE_SPECIAL = True
    SPECIAL_CHARS = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    @classmethod
    def validate(cls, password: str) -> tuple[bool, list[str]]:
        """
        Validate password strength.
        
        Args:
            password: Password to validate
            
        Returns:
            tuple: (is_valid, list_of_errors)
        """
        errors = []
        
        if len(password) < cls.MIN_LENGTH:
            errors.append(f"Password must be at least {cls.MIN_LENGTH} characters long")
        
        if cls.REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if cls.REQUIRE_LOWERCASE and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if cls.REQUIRE_DIGITS and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        if cls.REQUIRE_SPECIAL and not any(c in cls.SPECIAL_CHARS for c in password):
            errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db_session)
) -> "User":
    """
    Get the current authenticated user from JWT token.
    
    Args:
        credentials: Bearer token credentials
        db: Database session
        
    Returns:
        User: The authenticated user
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    import uuid
    from app.models.user import User
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Check if credentials are provided
    if credentials is None:
        raise credentials_exception
    
    try:
        # Verify the access token
        payload = verify_token(credentials.credentials, token_type="access")
        if payload is None:
            raise credentials_exception
            
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
            
        # Convert string UUID to UUID object
        user_uuid = uuid.UUID(user_id)
            
    except (JWTError, ValueError):
        raise credentials_exception
    
    # Get user from database
    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
        
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled"
        )
    
    return user
