"""
User Management API Endpoints for EmoSense Backend API

Provides endpoints for user profile management, user listing,
and administrative user operations.
"""

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db_session
from app.models.user import User
from app.schemas.user import UserResponse
from app.core.security import get_current_user


# Create router for user management endpoints
router = APIRouter()


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get Current User",
    description="Get the current authenticated user's profile"
)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user)
) -> UserResponse:
    """
    Get current user profile.
    
    Returns the authenticated user's profile information.
    
    Args:
        current_user: The authenticated user
        
    Returns:
        UserResponse: Current user's profile data
    """
    # Create user response data explicitly to avoid async issues
    user_data = {
        "id": current_user.id,
        "email": current_user.email,
        "first_name": current_user.first_name,
        "last_name": current_user.last_name,
        "phone_number": current_user.phone_number,
        "bio": current_user.bio,
        "avatar_url": current_user.avatar_url,
        "is_active": current_user.is_active,
        "is_verified": current_user.is_verified,
        "created_at": current_user.created_at,
        "updated_at": current_user.updated_at,
        "last_login": current_user.last_login,
    }
    
    return UserResponse(**user_data)


# TODO: Implement additional user management endpoints
# - PUT /me - Update current user profile
# - GET / - List users (admin only)
# - GET /{user_id} - Get user by ID
# - PUT /{user_id} - Update user (admin only)
# - DELETE /{user_id} - Delete user (admin only)
