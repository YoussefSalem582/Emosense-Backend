"""
Authentication API Endpoints for EmoSense Backend API

Provides endpoints for user authentication including login, registration,
token refresh, and password management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AuthenticationError, ValidationError
from app.core.security import create_user_tokens, verify_password, create_password_hash
from app.database import get_db_session
from app.models.user import User
from app.schemas.user import (
    UserCreate,
    UserLogin,
    UserLoginResponse,
    TokenRefresh,
    TokenResponse,
    UserResponse,
)


# Create router for authentication endpoints
router = APIRouter()


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register New User",
    description="Create a new user account with email and password"
)
async def register_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db_session)
) -> UserResponse:
    """
    Register a new user account.
    
    Creates a new user with the provided email and password.
    Email must be unique and password must meet security requirements.
    
    Args:
        user_data: User registration data
        db: Database session
        
    Returns:
        UserResponse: Created user information
        
    Raises:
        ValidationError: If email already exists or data is invalid
    """
    try:
        # Check if user already exists
        from sqlalchemy import select
        result = await db.execute(select(User).where(User.email == user_data.email))
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            raise ValidationError(
                detail="User with this email already exists",
                field_errors={"email": "Email already registered"}
            )
        
        # Create new user
        hashed_password = create_password_hash(user_data.password)
        db_user = User(
            email=user_data.email,
            hashed_password=hashed_password,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            phone_number=user_data.phone_number,
            bio=user_data.bio,
            avatar_url=user_data.avatar_url,
        )
        
        db.add(db_user)
        await db.commit()
        await db.refresh(db_user)
        
        # Create user response data explicitly to avoid greenlet issues
        user_data = {
            "id": db_user.id,
            "email": db_user.email,
            "first_name": db_user.first_name,
            "last_name": db_user.last_name,
            "phone_number": db_user.phone_number,
            "bio": db_user.bio,
            "avatar_url": db_user.avatar_url,
            "is_active": db_user.is_active,
            "is_verified": db_user.is_verified,
            "created_at": db_user.created_at,
            "updated_at": db_user.updated_at,
            "last_login": db_user.last_login,
        }
        
        return UserResponse(**user_data)
        
    except ValidationError:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}"
        )


@router.post(
    "/login",
    response_model=UserLoginResponse,
    summary="User Login",
    description="Authenticate user and return access tokens"
)
async def login_user(
    login_data: UserLogin,
    db: AsyncSession = Depends(get_db_session)
) -> UserLoginResponse:
    """
    Authenticate user and return JWT tokens.
    
    Validates user credentials and returns access and refresh tokens
    for authenticated API access.
    
    Args:
        login_data: User login credentials
        db: Database session
        
    Returns:
        UserLoginResponse: Authentication tokens and user info
        
    Raises:
        AuthenticationError: If credentials are invalid
    """
    try:
        # Get user by email
        from sqlalchemy import select
        result = await db.execute(select(User).where(User.email == login_data.email))
        user = result.scalar_one_or_none()
        
        if not user:
            raise AuthenticationError("Invalid email or password")
        
        # Verify password
        if not verify_password(login_data.password, user.hashed_password):
            raise AuthenticationError("Invalid email or password")
        
        # Check if user is active
        if not user.is_active:
            raise AuthenticationError("User account is disabled")
        
        # Update last login
        user.update_last_login()
        await db.commit()
        await db.refresh(user)
        
        # Create tokens
        tokens = create_user_tokens(
            user_id=str(user.id),
            email=user.email,
            roles=["user"] + (["admin"] if user.is_superuser else [])
        )
        
        # Create user response data explicitly to avoid greenlet issues
        user_data = {
            "id": user.id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "phone_number": user.phone_number,
            "bio": user.bio,
            "avatar_url": user.avatar_url,
            "is_active": user.is_active,
            "is_verified": user.is_verified,
            "created_at": user.created_at,
            "updated_at": user.updated_at,
            "last_login": user.last_login,
        }
        
        return UserLoginResponse(
            **tokens,
            expires_in=1800,  # 30 minutes
            user=UserResponse(**user_data)
        )
        
    except AuthenticationError:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh Access Token",
    description="Generate new access token using refresh token"
)
async def refresh_access_token(
    token_data: TokenRefresh,
    db: AsyncSession = Depends(get_db_session)
) -> TokenResponse:
    """
    Refresh access token using a valid refresh token.
    
    Generates a new access token for continued API access
    without requiring user to login again.
    
    Args:
        token_data: Refresh token data
        db: Database session
        
    Returns:
        TokenResponse: New access token
        
    Raises:
        AuthenticationError: If refresh token is invalid
    """
    try:
        import uuid
        from app.core.security import verify_token, create_access_token
        
        # Verify refresh token
        payload = verify_token(token_data.refresh_token, token_type="refresh")
        user_id = payload.get("sub")  # Use 'sub' like in access tokens
        
        if not user_id:
            raise AuthenticationError("Invalid refresh token")
        
        # Convert string UUID to UUID object
        try:
            user_uuid = uuid.UUID(user_id)
        except ValueError:
            raise AuthenticationError("Invalid user ID in token")
        
        # Get user from database
        from sqlalchemy import select
        result = await db.execute(select(User).where(User.id == user_uuid))
        user = result.scalar_one_or_none()
        
        if not user or not user.is_active:
            raise AuthenticationError("User not found or inactive")
        
        # Create new access token
        access_token = create_access_token({
            "sub": str(user.id),  # Use 'sub' for consistency
            "email": user.email,
            "roles": ["user"] + (["admin"] if user.is_superuser else [])
        })
        
        return TokenResponse(
            access_token=access_token,
            expires_in=1800  # 30 minutes
        )
        
    except AuthenticationError:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token refresh failed: {str(e)}"
        )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="User Logout",
    description="Logout user (client should discard tokens)"
)
async def logout_user():
    """
    Logout user.
    
    Since we're using stateless JWT tokens, logout is handled on the client
    side by discarding the tokens. This endpoint exists for API completeness.
    
    In a production system, you might want to implement token blacklisting
    using Redis or database storage.
    """
    # In a stateless JWT system, logout is handled client-side
    # by discarding the tokens. This endpoint is for API completeness.
    # 
    # For enhanced security, you could implement:
    # - Token blacklisting in Redis
    # - Short-lived tokens with automatic refresh
    # - Token revocation lists
    
    return {"message": "Logout successful"}
