"""
Configuration Management for EmoSense Backend API

Handles environment variables, settings validation, and configuration
for different deployment environments (development, testing, production).
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Uses Pydantic BaseSettings for automatic environment variable loading
    and validation. Settings are cached for performance.
    """
    
    # Application metadata
    VERSION: str = "1.0.0"
    APP_NAME: str = "EmoSense Backend API"
    DESCRIPTION: str = "Emotion analysis API for text, video, and audio processing"
    
    # Environment configuration
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")
    
    # Server configuration
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # Security settings
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    ALGORITHM: str = Field(default="HS256", env="ALGORITHM")
    
    # Database configuration
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    DATABASE_POOL_SIZE: int = Field(default=20, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(default=10, env="DATABASE_MAX_OVERFLOW")
    
    # Redis configuration
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    CACHE_EXPIRE_SECONDS: int = Field(default=3600, env="CACHE_EXPIRE_SECONDS")
    
    # CORS configuration
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    ALLOWED_HOSTS: List[str] = Field(
        default=["localhost", "127.0.0.1"],
        env="ALLOWED_HOSTS"
    )
    
    # File upload settings
    MAX_FILE_SIZE: int = Field(default=100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    ALLOWED_VIDEO_FORMATS: List[str] = Field(
        default=["mp4", "avi", "mov", "mkv"],
        env="ALLOWED_VIDEO_FORMATS"
    )
    ALLOWED_AUDIO_FORMATS: List[str] = Field(
        default=["mp3", "wav", "m4a", "flac"],
        env="ALLOWED_AUDIO_FORMATS"
    )
    UPLOAD_DIRECTORY: str = Field(default="uploads", env="UPLOAD_DIRECTORY")
    
    # Machine Learning settings
    MODEL_CACHE_SIZE: int = Field(default=1000, env="MODEL_CACHE_SIZE")
    MODEL_BATCH_SIZE: int = Field(default=32, env="MODEL_BATCH_SIZE")
    TEXT_MODEL_NAME: str = Field(
        default="cardiffnlp/twitter-roberta-base-emotion",
        env="TEXT_MODEL_NAME"
    )
    
    # Task processing (Celery)
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/1", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/1", env="CELERY_RESULT_BACKEND")
    
    # Monitoring and logging
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    STRUCTURED_LOGGING: bool = Field(default=True, env="STRUCTURED_LOGGING")
    
    # API Rate limiting
    RATE_LIMIT_REQUESTS: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    
    # External API settings
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    GOOGLE_CLOUD_PROJECT: Optional[str] = Field(default=None, env="GOOGLE_CLOUD_PROJECT")
    
    # Email configuration (for notifications)
    SMTP_HOST: Optional[str] = Field(default=None, env="SMTP_HOST")
    SMTP_PORT: int = Field(default=587, env="SMTP_PORT")
    SMTP_USERNAME: Optional[str] = Field(default=None, env="SMTP_USERNAME")
    SMTP_PASSWORD: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, value):
        """Parse CORS origins from string or list."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",")]
        return value
    
    @validator("ALLOWED_HOSTS", pre=True)
    def parse_allowed_hosts(cls, value):
        """Parse allowed hosts from string or list."""
        if isinstance(value, str):
            return [host.strip() for host in value.split(",")]
        return value
    
    @validator("ALLOWED_VIDEO_FORMATS", pre=True)
    def parse_video_formats(cls, value):
        """Parse allowed video formats from string or list."""
        if isinstance(value, str):
            return [fmt.strip().lower() for fmt in value.split(",")]
        return [fmt.lower() for fmt in value]
    
    @validator("ALLOWED_AUDIO_FORMATS", pre=True)
    def parse_audio_formats(cls, value):
        """Parse allowed audio formats from string or list."""
        if isinstance(value, str):
            return [fmt.strip().lower() for fmt in value.split(",")]
        return [fmt.lower() for fmt in value]
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, value):
        """Validate environment value."""
        allowed_environments = ["development", "testing", "staging", "production"]
        if value.lower() not in allowed_environments:
            raise ValueError(f"Environment must be one of: {allowed_environments}")
        return value.lower()
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, value):
        """Validate log level value."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if value.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {allowed_levels}")
        return value.upper()
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Returns:
        Settings instance with all configuration values
    """
    return Settings()


# Convenience function to check if we're in development
def is_development() -> bool:
    """Check if running in development environment."""
    return get_settings().ENVIRONMENT == "development"


# Convenience function to check if we're in production
def is_production() -> bool:
    """Check if running in production environment."""
    return get_settings().ENVIRONMENT == "production"
