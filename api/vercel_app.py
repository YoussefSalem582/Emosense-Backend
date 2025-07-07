"""
Vercel-optimized FastAPI application for EmoSense Backend
Minimal version for serverless deployment
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import time
import uuid
import json

# Create FastAPI application
app = FastAPI(
    title="EmoSense Backend API",
    description="Emotion analysis API optimized for Vercel deployment",
    version="1.0.0"
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

# Models
class UserRegister(BaseModel):
    email: str
    password: str
    username: str

class UserLogin(BaseModel):
    email: str
    password: str

class EmotionRequest(BaseModel):
    text: str

class EmotionResponse(BaseModel):
    emotions: Dict[str, float]
    dominant_emotion: str
    confidence: float

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

# Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "EmoSense Backend API",
        "version": "1.0.0",
        "status": "healthy",
        "deployment": "vercel",
        "timestamp": time.time()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "deployment": "vercel",
        "users_count": len(users_db),
        "sessions_count": len(sessions_db)
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

@app.post("/api/v1/emotion/analyze")
async def analyze_emotion(request: EmotionRequest) -> EmotionResponse:
    """Analyze emotion in text (demo version)"""
    # Simple demo emotion analysis
    text = request.text.lower()
    emotions = {
        "happy": 0.1,
        "sad": 0.1,
        "angry": 0.1,
        "fear": 0.1,
        "surprise": 0.1,
        "neutral": 0.5
    }
    
    # Simple keyword-based demo analysis
    if any(word in text for word in ["happy", "joy", "great", "awesome", "wonderful"]):
        emotions["happy"] = 0.8
        emotions["neutral"] = 0.2
    elif any(word in text for word in ["sad", "cry", "terrible", "awful", "horrible"]):
        emotions["sad"] = 0.8
        emotions["neutral"] = 0.2
    elif any(word in text for word in ["angry", "mad", "hate", "furious", "rage"]):
        emotions["angry"] = 0.8
        emotions["neutral"] = 0.2
    elif any(word in text for word in ["scared", "afraid", "terrified", "nervous"]):
        emotions["fear"] = 0.8
        emotions["neutral"] = 0.2
    elif any(word in text for word in ["surprised", "shocked", "amazed", "wow"]):
        emotions["surprise"] = 0.8
        emotions["neutral"] = 0.2
    
    dominant_emotion = max(emotions, key=emotions.get)
    confidence = emotions[dominant_emotion]
    
    return EmotionResponse(
        emotions=emotions,
        dominant_emotion=dominant_emotion,
        confidence=confidence
    )

@app.get("/api/v1/analytics/dashboard")
async def get_analytics():
    """Get analytics dashboard data"""
    return {
        "total_users": len(users_db),
        "total_sessions": len(sessions_db),
        "analysis_stats": {
            "total_analyses": 0,
            "emotion_distribution": {
                "happy": 25,
                "sad": 15,
                "angry": 10,
                "fear": 5,
                "surprise": 10,
                "neutral": 35
            }
        },
        "timestamp": time.time()
    }
