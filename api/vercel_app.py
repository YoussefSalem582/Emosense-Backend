"""
Vercel-optimized FastAPI application for EmoSense Backend
Enhanced version with real model integration support
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List
import time
import uuid
import json
import base64

# Import emotion models
from emotion_models import model_manager, EmotionAnalysisRequest, EmotionResult

# Create FastAPI application
app = FastAPI(
    title="EmoSense Backend API",
    description="Advanced emotion analysis API with real model integration support",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage for demo purposes
users_db = {}
sessions_db = {}
analysis_history = []

# Models
class UserRegister(BaseModel):
    email: str
    password: str
    username: str

class UserLogin(BaseModel):
    email: str
    password: str

class TextEmotionRequest(BaseModel):
    text: str
    model_type: str = "text"

class MultimodalEmotionRequest(BaseModel):
    text: Optional[str] = None
    audio_base64: Optional[str] = None
    video_base64: Optional[str] = None
    model_type: str = "auto"

class EmotionResponse(BaseModel):
    emotions: Dict[str, float]
    dominant_emotion: str
    confidence: float
    processing_time: float
    model_version: str
    analysis_id: str
    metadata: Dict = {}

# Helper functions
def create_token(user_id: str) -> str:
    """Create a simple token for demo purposes"""
    token = str(uuid.uuid4())
    sessions_db[token] = {"user_id": user_id, "created_at": time.time()}
    return token

def verify_token(token: str) -> str:
    """Verify token and return user_id"""
    session = sessions_db.get(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid token")
    return session["user_id"]

def save_analysis_result(result: EmotionResult, user_id: str = None) -> str:
    """Save analysis result and return analysis ID"""
    analysis_id = str(uuid.uuid4())
    analysis_record = {
        "id": analysis_id,
        "user_id": user_id,
        "result": result.dict(),
        "timestamp": time.time()
    }
    analysis_history.append(analysis_record)
    return analysis_id

# Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "EmoSense Backend API",
        "version": "2.0.0",
        "status": "healthy",
        "deployment": "vercel",
        "features": ["text_analysis", "audio_analysis", "video_analysis", "multimodal_analysis"],
        "timestamp": time.time()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_info = model_manager.get_available_models()
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "deployment": "vercel",
        "users_count": len(users_db),
        "sessions_count": len(sessions_db),
        "total_analyses": len(analysis_history),
        "models": models_info
    }

@app.get("/api/v1/models")
async def get_available_models():
    """Get information about available emotion analysis models"""
    return {
        "models": model_manager.get_available_models(),
        "timestamp": time.time()
    }

@app.post("/api/v1/auth/register")
async def register(user_data: UserRegister):
    """Register a new user"""
    if user_data.email in users_db:
        raise HTTPException(status_code=400, detail="User already exists")
    
    user_id = str(uuid.uuid4())
    users_db[user_data.email] = {
        "id": user_id,
        "email": user_data.email,
        "username": user_data.username,
        "password": user_data.password,  # In real app, hash this!
        "created_at": time.time()
    }
    
    token = create_token(user_id)
    
    return {
        "message": "User registered successfully",
        "user": {
            "id": user_id,
            "email": user_data.email,
            "username": user_data.username
        },
        "access_token": token,
        "token_type": "bearer"
    }

@app.post("/api/v1/auth/login")
async def login(credentials: UserLogin):
    """Login user"""
    user = users_db.get(credentials.email)
    if not user or user["password"] != credentials.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token(user["id"])
    
    return {
        "message": "Login successful",
        "user": {
            "id": user["id"],
            "email": user["email"],
            "username": user["username"]
        },
        "access_token": token,
        "token_type": "bearer"
    }

@app.post("/api/v1/emotion/analyze/text", response_model=EmotionResponse)
async def analyze_text_emotion(request: TextEmotionRequest):
    """Analyze emotion from text using advanced models"""
    try:
        # Create emotion analysis request
        emotion_request = EmotionAnalysisRequest(
            text=request.text,
            analysis_type=request.model_type
        )
        
        # Analyze using model manager
        result = await model_manager.analyze_emotion(emotion_request)
        
        # Save analysis result
        analysis_id = save_analysis_result(result)
        
        return EmotionResponse(
            emotions=result.emotions,
            dominant_emotion=result.dominant_emotion,
            confidence=result.confidence,
            processing_time=result.processing_time,
            model_version=result.model_version,
            analysis_id=analysis_id,
            metadata=result.metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/v1/emotion/analyze/multimodal", response_model=EmotionResponse)
async def analyze_multimodal_emotion(request: MultimodalEmotionRequest):
    """Analyze emotion from multiple modalities (text, audio, video)"""
    try:
        # Convert base64 data to bytes if provided
        audio_data = None
        video_data = None
        
        if request.audio_base64:
            audio_data = base64.b64decode(request.audio_base64)
        
        if request.video_base64:
            video_data = base64.b64decode(request.video_base64)
        
        # Create emotion analysis request
        emotion_request = EmotionAnalysisRequest(
            text=request.text,
            audio_data=audio_data,
            video_data=video_data,
            analysis_type=request.model_type
        )
        
        # Analyze using model manager
        result = await model_manager.analyze_emotion(emotion_request)
        
        # Save analysis result
        analysis_id = save_analysis_result(result)
        
        return EmotionResponse(
            emotions=result.emotions,
            dominant_emotion=result.dominant_emotion,
            confidence=result.confidence,
            processing_time=result.processing_time,
            model_version=result.model_version,
            analysis_id=analysis_id,
            metadata=result.metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multimodal analysis failed: {str(e)}")

@app.post("/api/v1/emotion/analyze/file")
async def analyze_emotion_from_file(
    file: UploadFile = File(...),
    analysis_type: str = Form("auto"),
    text: Optional[str] = Form(None)
):
    """Analyze emotion from uploaded file (audio/video/image)"""
    try:
        # Read file data
        file_data = await file.read()
        
        # Determine analysis type based on file type if auto
        if analysis_type == "auto":
            content_type = file.content_type or ""
            if content_type.startswith("audio/"):
                analysis_type = "audio"
            elif content_type.startswith("video/"):
                analysis_type = "video"
            elif content_type.startswith("image/"):
                analysis_type = "video"  # Use video model for images
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Create emotion analysis request
        emotion_request = EmotionAnalysisRequest(
            text=text,
            audio_data=file_data if analysis_type == "audio" else None,
            video_data=file_data if analysis_type in ["video", "image"] else None,
            analysis_type=analysis_type
        )
        
        # Analyze using model manager
        result = await model_manager.analyze_emotion(emotion_request)
        
        # Save analysis result
        analysis_id = save_analysis_result(result)
        
        return EmotionResponse(
            emotions=result.emotions,
            dominant_emotion=result.dominant_emotion,
            confidence=result.confidence,
            processing_time=result.processing_time,
            model_version=result.model_version,
            analysis_id=analysis_id,
            metadata={
                **result.metadata,
                "file_name": file.filename,
                "file_type": file.content_type,
                "file_size": len(file_data)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File analysis failed: {str(e)}")

@app.get("/api/v1/emotion/history")
async def get_analysis_history(limit: int = 10):
    """Get recent emotion analysis history"""
    recent_analyses = sorted(analysis_history, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    return {
        "analyses": recent_analyses,
        "total_count": len(analysis_history),
        "timestamp": time.time()
    }

@app.get("/api/v1/emotion/analysis/{analysis_id}")
async def get_analysis_result(analysis_id: str):
    """Get specific analysis result by ID"""
    for analysis in analysis_history:
        if analysis["id"] == analysis_id:
            return analysis
    
    raise HTTPException(status_code=404, detail="Analysis not found")

@app.get("/api/v1/analytics/dashboard")
async def get_analytics():
    """Get analytics dashboard data"""
    # Calculate emotion distribution from history
    emotion_counts = {}
    for analysis in analysis_history:
        dominant_emotion = analysis["result"]["dominant_emotion"]
        emotion_counts[dominant_emotion] = emotion_counts.get(dominant_emotion, 0) + 1
    
    return {
        "total_users": len(users_db),
        "total_sessions": len(sessions_db),
        "analysis_stats": {
            "total_analyses": len(analysis_history),
            "emotion_distribution": emotion_counts,
            "models_available": len(model_manager.get_available_models()),
            "recent_analyses": len([a for a in analysis_history if time.time() - a["timestamp"] < 3600])
        },
        "timestamp": time.time()
    }

# Legacy endpoint for backward compatibility
@app.post("/api/v1/emotion/analyze")
async def analyze_emotion_legacy(request: TextEmotionRequest) -> EmotionResponse:
    """Legacy emotion analysis endpoint for backward compatibility"""
    return await analyze_text_emotion(request)
