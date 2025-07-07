"""
EmoSense Backend API - Vercel Deployment Version

Simplified FastAPI application optimized for serverless deployment on Vercel.
Removes heavy dependencies and database connections that aren't suitable for serverless.
"""

import time
from typing import Dict, Any

import structlog
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configure structured logging
logger = structlog.get_logger(__name__)

# Simple in-memory settings for Vercel
class Settings:
    VERSION = "1.0.0"
    ENVIRONMENT = "production"
    CORS_ORIGINS = ["*"]  # Configure this properly for production

settings = Settings()

# Create FastAPI application instance
app = FastAPI(
    title="EmoSense Backend API",
    description="Comprehensive emotion analysis API for text, video, and audio processing",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class EmotionRequest(BaseModel):
    text: str

class EmotionResponse(BaseModel):
    text: str
    emotions: Dict[str, float]
    dominant_emotion: str
    confidence: float

class UserRegistration(BaseModel):
    email: str
    password: str
    confirm_password: str
    first_name: str
    last_name: str

class UserLogin(BaseModel):
    email: str
    password: str

# Exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": time.time(),
            }
        },
    )

# Middleware for logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None,
    )
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    logger.info(
        "Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=f"{process_time:.4f}s",
    )
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Root endpoints
@app.get("/", tags=["Root"])
async def root() -> dict:
    """Root endpoint providing basic API information."""
    return {
        "name": "EmoSense Backend API",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "status": "healthy",
        "docs_url": "/docs",
        "deployment": "vercel"
    }

@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "deployment": "vercel",
        "components": {
            "api": {"status": "healthy"},
            "serverless": {"status": "healthy"}
        }
    }

# Authentication endpoints (simplified for demo)
@app.post("/api/v1/auth/register", tags=["Authentication"])
async def register_user(user_data: UserRegistration):
    """Simplified user registration for demo purposes."""
    if user_data.password != user_data.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match"
        )
    
    # In a real implementation, you would save to database
    return {
        "message": "User registered successfully",
        "user": {
            "email": user_data.email,
            "first_name": user_data.first_name,
            "last_name": user_data.last_name
        }
    }

@app.post("/api/v1/auth/login", tags=["Authentication"])
async def login_user(login_data: UserLogin):
    """Simplified user login for demo purposes."""
    # In a real implementation, you would verify against database
    if login_data.email == "demo@example.com" and login_data.password == "demo123":
        return {
            "access_token": "demo_access_token",
            "refresh_token": "demo_refresh_token",
            "token_type": "bearer",
            "user": {
                "email": login_data.email,
                "first_name": "Demo",
                "last_name": "User"
            }
        }
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials"
    )

# Emotion analysis endpoints (simplified for demo)
@app.post("/api/v1/emotion/text", response_model=EmotionResponse, tags=["Emotion Analysis"])
async def analyze_text_emotion(request: EmotionRequest) -> EmotionResponse:
    """
    Analyze emotion in text using basic sentiment analysis.
    This is a simplified version for Vercel deployment.
    """
    text = request.text.lower()
    
    # Simple rule-based emotion detection for demo
    emotions = {
        "joy": 0.0,
        "sadness": 0.0,
        "anger": 0.0,
        "fear": 0.0,
        "surprise": 0.0,
        "disgust": 0.0,
        "neutral": 0.5
    }
    
    # Basic keyword matching
    if any(word in text for word in ["happy", "joy", "excited", "great", "wonderful", "amazing"]):
        emotions["joy"] = 0.8
        emotions["neutral"] = 0.2
    elif any(word in text for word in ["sad", "depressed", "terrible", "awful", "crying"]):
        emotions["sadness"] = 0.7
        emotions["neutral"] = 0.3
    elif any(word in text for word in ["angry", "mad", "furious", "hate", "annoyed"]):
        emotions["anger"] = 0.75
        emotions["neutral"] = 0.25
    elif any(word in text for word in ["scared", "afraid", "worried", "anxious", "fear"]):
        emotions["fear"] = 0.7
        emotions["neutral"] = 0.3
    
    # Find dominant emotion
    dominant_emotion = max(emotions, key=emotions.get)
    confidence = emotions[dominant_emotion]
    
    return EmotionResponse(
        text=request.text,
        emotions=emotions,
        dominant_emotion=dominant_emotion,
        confidence=confidence
    )

@app.get("/api/v1/emotion/models", tags=["Emotion Analysis"])
async def get_available_models():
    """Get list of available emotion analysis models."""
    return {
        "models": [
            {
                "id": "basic-text",
                "name": "Basic Text Analyzer",
                "type": "text",
                "description": "Simple rule-based text emotion analysis",
                "supported_emotions": ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
            }
        ]
    }

# User endpoints (simplified)
@app.get("/api/v1/users/me", tags=["Users"])
async def get_current_user():
    """Get current user information (demo version)."""
    return {
        "id": "demo-user-id",
        "email": "demo@example.com",
        "first_name": "Demo",
        "last_name": "User",
        "is_active": True,
        "created_at": "2024-01-01T00:00:00Z"
    }

# Analytics endpoints (simplified)
@app.get("/api/v1/analytics/summary", tags=["Analytics"])
async def get_analytics_summary():
    """Get analytics summary (demo version)."""
    return {
        "total_analyses": 1000,
        "today_analyses": 50,
        "most_common_emotion": "joy",
        "user_count": 100,
        "api_uptime": "99.9%"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
