"""
User Pydantic Schemas for EmoSense Backend API

Defines request and response schemas for user-related API endpoints
using Pydantic for data validation and serialization.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, validator


class UserBase(BaseModel):
    """Base user schema with common fields."""
    
    email: EmailStr = Field(..., description="User's email address")
    first_name: Optional[str] = Field(None, max_length=100, description="User's first name")
    last_name: Optional[str] = Field(None, max_length=100, description="User's last name")
    phone_number: Optional[str] = Field(None, max_length=20, description="User's phone number")
    bio: Optional[str] = Field(None, max_length=1000, description="User's biography")
    avatar_url: Optional[str] = Field(None, max_length=500, description="URL to user's avatar")


class UserCreate(UserBase):
    """Schema for creating a new user."""
    
    password: str = Field(
        ..., 
        min_length=8, 
        max_length=100,
        description="User's password (minimum 8 characters)"
    )
    confirm_password: str = Field(
        ...,
        description="Password confirmation (must match password)"
    )
    
    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        """Validate that password and confirm_password match."""
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v
    
    @validator('phone_number')
    def validate_phone_number(cls, v):
        """Basic phone number validation."""
        if v is not None:
            # Remove all non-digit characters for basic validation
            digits_only = ''.join(filter(str.isdigit, v))
            if len(digits_only) < 10:
                raise ValueError('Phone number must contain at least 10 digits')
        return v


class UserUpdate(BaseModel):
    """Schema for updating user information."""
    
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    phone_number: Optional[str] = Field(None, max_length=20)
    bio: Optional[str] = Field(None, max_length=1000)
    avatar_url: Optional[str] = Field(None, max_length=500)
    
    @validator('phone_number')
    def validate_phone_number(cls, v):
        """Basic phone number validation."""
        if v is not None:
            digits_only = ''.join(filter(str.isdigit, v))
            if len(digits_only) < 10:
                raise ValueError('Phone number must contain at least 10 digits')
        return v


class UserResponse(UserBase):
    """Schema for user response data."""
    
    id: UUID = Field(..., description="User's unique identifier")
    full_name: str = Field(..., description="User's full name")
    is_active: bool = Field(..., description="Whether the user account is active")
    is_verified: bool = Field(..., description="Whether the user email is verified")
    created_at: datetime = Field(..., description="Account creation timestamp")
    updated_at: datetime = Field(..., description="Last account update timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class UserProfile(BaseModel):
    """Schema for detailed user profile information."""
    
    id: UUID
    email: EmailStr
    full_name: str
    first_name: Optional[str]
    last_name: Optional[str]
    phone_number: Optional[str]
    bio: Optional[str]
    avatar_url: Optional[str]
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime]
    
    # Statistics
    total_analyses: Optional[int] = Field(0, description="Total number of analyses performed")
    analyses_this_month: Optional[int] = Field(0, description="Analyses performed this month")
    
    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    """Schema for user login request."""
    
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., description="User's password")
    remember_me: bool = Field(default=False, description="Whether to extend session duration")


class UserLoginResponse(BaseModel):
    """Schema for user login response."""
    
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user: UserResponse = Field(..., description="User information")


class TokenRefresh(BaseModel):
    """Schema for token refresh request."""
    
    refresh_token: str = Field(..., description="Valid refresh token")


class TokenResponse(BaseModel):
    """Schema for token response."""
    
    access_token: str = Field(..., description="New JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")


class PasswordChangeRequest(BaseModel):
    """Schema for password change request."""
    
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(
        ..., 
        min_length=8, 
        max_length=100,
        description="New password (minimum 8 characters)"
    )
    confirm_new_password: str = Field(..., description="New password confirmation")
    
    @validator('confirm_new_password')
    def passwords_match(cls, v, values, **kwargs):
        """Validate that new passwords match."""
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('New passwords do not match')
        return v


class PasswordResetRequest(BaseModel):
    """Schema for password reset request."""
    
    email: EmailStr = Field(..., description="Email address for password reset")


class PasswordResetConfirm(BaseModel):
    """Schema for password reset confirmation."""
    
    token: str = Field(..., description="Password reset token")
    new_password: str = Field(
        ..., 
        min_length=8, 
        max_length=100,
        description="New password"
    )
    confirm_new_password: str = Field(..., description="New password confirmation")
    
    @validator('confirm_new_password')
    def passwords_match(cls, v, values, **kwargs):
        """Validate that passwords match."""
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v


class EmailVerificationRequest(BaseModel):
    """Schema for email verification request."""
    
    email: EmailStr = Field(..., description="Email address to verify")


class EmailVerificationConfirm(BaseModel):
    """Schema for email verification confirmation."""
    
    token: str = Field(..., description="Email verification token")


class UserListResponse(BaseModel):
    """Schema for paginated user list response."""
    
    users: list[UserResponse] = Field(..., description="List of users")
    total: int = Field(..., description="Total number of users")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Number of users per page")
    pages: int = Field(..., description="Total number of pages")
    
    class Config:
        from_attributes = True
