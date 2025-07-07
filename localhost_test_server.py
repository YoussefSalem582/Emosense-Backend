"""
Simple localhost FastAPI server for testing Flutter integration
Run this to test your Flutter app with a local backend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio
import logging
from typing import Dict, Any
import random
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="EmoSense Local Test API",
    description="Simple localhost API for testing Flutter integration",
    version="1.0.0"
)

# Add CORS middleware to allow Flutter web/mobile to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class TextAnalysisRequest(BaseModel):
    text: str

class EmotionResult(BaseModel):
    emotion: str
    confidence: float
    emotions: Dict[str, float]
    processing_time: float
    model_used: str
    success: bool = True

# Mock emotion analysis function
def analyze_text_emotion(text: str) -> EmotionResult:
    """Simple mock emotion analysis for testing"""
    start_time = time.time()
    
    # Simple keyword-based emotion detection for demo
    text_lower = text.lower()
    
    if any(word in text_lower for word in ["happy", "joy", "great", "awesome", "love", "excellent"]):
        primary_emotion = "joy"
        emotions = {"joy": 0.8, "neutral": 0.15, "surprise": 0.05}
    elif any(word in text_lower for word in ["sad", "depressed", "unhappy", "terrible", "awful"]):
        primary_emotion = "sadness"
        emotions = {"sadness": 0.75, "neutral": 0.2, "anger": 0.05}
    elif any(word in text_lower for word in ["angry", "mad", "furious", "hate", "annoyed"]):
        primary_emotion = "anger"
        emotions = {"anger": 0.7, "sadness": 0.2, "neutral": 0.1}
    elif any(word in text_lower for word in ["scared", "afraid", "worried", "nervous", "anxious"]):
        primary_emotion = "fear"
        emotions = {"fear": 0.65, "neutral": 0.25, "sadness": 0.1}
    elif any(word in text_lower for word in ["wow", "amazing", "shocked", "surprised", "unexpected"]):
        primary_emotion = "surprise"
        emotions = {"surprise": 0.7, "joy": 0.2, "neutral": 0.1}
    elif any(word in text_lower for word in ["disgusting", "gross", "yuck", "revolting"]):
        primary_emotion = "disgust"
        emotions = {"disgust": 0.8, "anger": 0.15, "neutral": 0.05}
    else:
        primary_emotion = "neutral"
        emotions = {"neutral": 0.6, "joy": 0.2, "sadness": 0.1, "surprise": 0.1}
    
    processing_time = time.time() - start_time
    
    return EmotionResult(
        emotion=primary_emotion,
        confidence=emotions[primary_emotion],
        emotions=emotions,
        processing_time=processing_time,
        model_used="localhost_mock_model",
        success=True
    )

# API Endpoints
@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "EmoSense Local Test API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "text_analysis": "/analyze/text",
            "model_status": "/models/status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "message": "Localhost backend is running"
    }

@app.post("/analyze/text")
async def analyze_text(request: TextAnalysisRequest):
    """Analyze text emotion"""
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        result = analyze_text_emotion(request.text.strip())
        logger.info(f"Analyzed text: '{request.text[:50]}...' -> {result.emotion}")
        
        return result.dict()
        
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/audio")
async def analyze_audio():
    """Mock audio analysis"""
    await asyncio.sleep(0.1)  # Simulate processing
    return {
        "emotion": "joy",
        "confidence": 0.75,
        "emotions": {"joy": 0.75, "neutral": 0.15, "surprise": 0.1},
        "processing_time": 0.1,
        "model_used": "localhost_mock_audio",
        "success": True
    }

@app.post("/analyze/video")
async def analyze_video():
    """Mock video analysis"""
    await asyncio.sleep(0.2)  # Simulate processing
    return {
        "emotion": "surprise",
        "confidence": 0.8,
        "emotions": {"surprise": 0.8, "joy": 0.15, "neutral": 0.05},
        "processing_time": 0.2,
        "model_used": "localhost_mock_video",
        "success": True
    }

@app.get("/models/status")
async def models_status():
    """Get model status"""
    return {
        "models": {
            "text_model": {
                "name": "localhost_mock_model",
                "status": "loaded",
                "type": "keyword_based"
            },
            "audio_model": {
                "name": "localhost_mock_audio",
                "status": "loaded",
                "type": "mock"
            },
            "video_model": {
                "name": "localhost_mock_video", 
                "status": "loaded",
                "type": "mock"
            }
        },
        "backend_status": "running",
        "memory_usage": "low"
    }

if __name__ == "__main__":
    print("🚀 Starting EmoSense Localhost Test Server...")
    print("📱 Flutter Web: http://localhost:8000")
    print("📱 Android Emulator: http://10.0.2.2:8000")
    print("📱 iOS Simulator: http://localhost:8000")
    print("📱 Desktop Apps: http://localhost:8000")
    print("📱 Physical Devices: Use your computer's IP address")
    print("🔍 API Docs: http://localhost:8000/docs")
    print("\n✅ Ready for Flutter integration testing!")
    
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Listen on all interfaces
        port=8000,
        log_level="info",
        reload=False
    )
