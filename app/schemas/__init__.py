"""
Pydantic Schemas for EmoSense Backend API

This package contains all Pydantic schemas for request/response validation
and serialization across the application.
"""

from .emotion import (
    AudioAnalysisRequest,
    AnalysisListResponse,
    AnalysisStatsResponse,
    AudioSegmentResult,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    DetailedAnalysisResponse,
    EmotionAnalysisResponse,
    EmotionScores,
    TextAnalysisRequest,
    TextSegmentResult,
    VideoAnalysisRequest,
    VideoFrameResult,
)
from .user import (
    EmailVerificationConfirm,
    EmailVerificationRequest,
    PasswordChangeRequest,
    PasswordResetConfirm,
    PasswordResetRequest,
    TokenRefresh,
    TokenResponse,
    UserCreate,
    UserListResponse,
    UserLogin,
    UserLoginResponse,
    UserProfile,
    UserResponse,
    UserUpdate,
)

__all__ = [
    # User schemas
    "UserCreate",
    "UserUpdate", 
    "UserResponse",
    "UserProfile",
    "UserLogin",
    "UserLoginResponse",
    "UserListResponse",
    "TokenRefresh",
    "TokenResponse",
    "PasswordChangeRequest",
    "PasswordResetRequest",
    "PasswordResetConfirm",
    "EmailVerificationRequest",
    "EmailVerificationConfirm",
    # Emotion analysis schemas
    "TextAnalysisRequest",
    "VideoAnalysisRequest",
    "AudioAnalysisRequest",
    "EmotionAnalysisResponse",
    "DetailedAnalysisResponse",
    "TextSegmentResult",
    "VideoFrameResult",
    "AudioSegmentResult",
    "BatchAnalysisRequest",
    "BatchAnalysisResponse",
    "AnalysisListResponse",
    "AnalysisStatsResponse",
    "EmotionScores",
]
