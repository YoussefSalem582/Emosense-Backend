"""
User Model for EmoSense Backend API

Defines the User SQLAlchemy model for storing user authentication
and profile information in the database.
"""

from datetime import datetime
from typing import List

from sqlalchemy import Boolean, Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.database import Base


class User(Base):
    """
    User model for storing user account information.
    
    This model handles user authentication data, profile information,
    and relationships with other entities in the system.
    """
    
    __tablename__ = "users"
    
    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
        doc="Unique user identifier"
    )
    
    # Authentication fields
    email = Column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
        doc="User email address (unique)"
    )
    
    hashed_password = Column(
        String(255),
        nullable=False,
        doc="BCrypt hashed password"
    )
    
    # Profile information
    first_name = Column(
        String(100),
        nullable=True,
        doc="User's first name"
    )
    
    last_name = Column(
        String(100),
        nullable=True,
        doc="User's last name"
    )
    
    phone_number = Column(
        String(20),
        nullable=True,
        doc="User's phone number"
    )
    
    # Account status
    is_active = Column(
        Boolean(),
        default=True,
        nullable=False,
        doc="Whether the user account is active"
    )
    
    is_verified = Column(
        Boolean(),
        default=False,
        nullable=False,
        doc="Whether the user email is verified"
    )
    
    is_superuser = Column(
        Boolean(),
        default=False,
        nullable=False,
        doc="Whether the user has superuser privileges"
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        doc="Account creation timestamp"
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        doc="Last account update timestamp"
    )
    
    last_login = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Last login timestamp"
    )
    
    # Additional profile fields
    avatar_url = Column(
        String(500),
        nullable=True,
        doc="URL to user's avatar image"
    )
    
    bio = Column(
        Text,
        nullable=True,
        doc="User's biography or description"
    )
    
    # Verification token for email verification
    verification_token = Column(
        String(255),
        nullable=True,
        doc="Email verification token"
    )
    
    # Password reset token
    reset_token = Column(
        String(255),
        nullable=True,
        doc="Password reset token"
    )
    
    reset_token_expires = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Password reset token expiration"
    )
    
    # Relationships
    emotion_analyses = relationship(
        "EmotionAnalysis",
        back_populates="user",
        cascade="all, delete-orphan",
        doc="User's emotion analysis records"
    )
    
    def __repr__(self) -> str:
        """String representation of the user."""
        return f"<User(id={self.id}, email={self.email})>"
    
    @property
    def full_name(self) -> str:
        """Get user's full name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        elif self.last_name:
            return self.last_name
        else:
            return self.email.split("@")[0]  # Use email prefix as fallback
    
    def update_last_login(self) -> None:
        """Update the last login timestamp."""
        self.last_login = datetime.utcnow()
    
    def is_password_reset_valid(self) -> bool:
        """Check if password reset token is valid and not expired."""
        if not self.reset_token or not self.reset_token_expires:
            return False
        return datetime.utcnow() < self.reset_token_expires
    
    def clear_reset_token(self) -> None:
        """Clear password reset token and expiration."""
        self.reset_token = None
        self.reset_token_expires = None
    
    def to_dict(self) -> dict:
        """Convert user to dictionary (excluding sensitive fields)."""
        return {
            "id": str(self.id),
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.full_name,
            "phone_number": self.phone_number,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "avatar_url": self.avatar_url,
            "bio": self.bio,
        }
