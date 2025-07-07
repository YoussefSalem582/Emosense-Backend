"""
Railway-optimized FastAPI server for EmoSense Backend
Optimized for Railway's free tier with efficient resource usage
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
import logging
import sys
import os
import time
import psutil

# Configure logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Railway environment variables
PORT = int(os.getenv("PORT", 8000))
RAILWAY_ENVIRONMENT = os.getenv("RAILWAY_ENVIRONMENT", "production")
RAILWAY_PUBLIC_DOMAIN = os.getenv("RAILWAY_PUBLIC_DOMAIN", "localhost")

# Add the api directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

# Import emotion models with error handling
try:
    from emotion_models_railway import (
        EmotionAnalysisRequest,
        EmotionResult,
        RailwayTextEmotionModel,
        MockAudioEmotionModel,
        MockVideoEmotionModel,
    )
    MODELS_AVAILABLE = True
    logger.info("✅ Railway emotion models imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import emotion models: {e}")
    MODELS_AVAILABLE = False

# Create FastAPI app optimized for Railway
app = FastAPI(
    title="EmoSense Emotion Analysis API",
    description="Railway-deployed emotion analysis for text, audio, and video",
    version="1.5.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class TextAnalysisRequest(BaseModel):
    text: str
    analysis_type: str = "text"

class AnalysisResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    platform: str
    models_available: bool
    uptime: float
    memory_usage: Dict[str, Any]
    railway_info: Dict[str, Any]

# Global variables
text_model = None
audio_model = None
video_model = None
startup_time = time.time()

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global text_model, audio_model, video_model
    
    logger.info("🚂 Starting EmoSense API on Railway...")
    logger.info(f"🌍 Environment: {RAILWAY_ENVIRONMENT}")
    logger.info(f"🚪 Port: {PORT}")
    logger.info(f"🔗 Domain: {RAILWAY_PUBLIC_DOMAIN}")
    
    if MODELS_AVAILABLE:
        try:
            # Initialize models
            text_model = RailwayTextEmotionModel()
            audio_model = MockAudioEmotionModel()
            video_model = MockVideoEmotionModel()
            
            # Load models asynchronously
            await asyncio.gather(
                text_model.load_model(),
                audio_model.load_model(),
                video_model.load_model(),
            )
            
            logger.info("✅ All models loaded successfully")
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
    else:
        logger.warning("⚠️ Models not available, API will return errors")

def get_memory_info():
    """Get current memory usage"""
    process = psutil.Process()
    memory = process.memory_info()
    return {
        "rss_mb": round(memory.rss / 1024 / 1024, 2),
        "vms_mb": round(memory.vms / 1024 / 1024, 2),
        "percent": round(process.memory_percent(), 2)
    }

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health information"""
    return HealthResponse(
        status="healthy",
        version="1.5.0",
        environment=RAILWAY_ENVIRONMENT,
        platform="railway",
        models_available=MODELS_AVAILABLE and text_model is not None,
        uptime=time.time() - startup_time,
        memory_usage=get_memory_info(),
        railway_info={
            "domain": RAILWAY_PUBLIC_DOMAIN,
            "port": PORT,
            "optimized": True
        }
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint"""
    return await root()

@app.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """Analyze emotion in text"""
    if not MODELS_AVAILABLE or text_model is None:
        raise HTTPException(
            status_code=503,
            detail="Text analysis model not available"
        )
    
    try:
        # Validate input
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty"
            )
        
        if len(request.text) > 5000:  # Railway memory limit
            raise HTTPException(
                status_code=400,
                detail="Text too long (max 5000 characters for Railway)"
            )
        
        # Analyze emotion
        result = await text_model.analyze_emotion(request.text)
        
        return AnalysisResponse(
            success=result.success,
            data={
                "emotion": result.emotion,
                "confidence": result.confidence,
                "emotions": result.emotions,
                "processing_time": result.processing_time,
                "model_used": result.model_used,
                "platform": "railway",
                "memory_usage": get_memory_info()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Text analysis failed: {str(e)}"
        )

@app.post("/analyze/audio", response_model=AnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)):
    """Analyze emotion in audio (mock on Railway free tier)"""
    if not MODELS_AVAILABLE or audio_model is None:
        raise HTTPException(
            status_code=503,
            detail="Audio analysis model not available"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Mock analysis (real analysis would require more resources)
        result = await audio_model.analyze_emotion(content)
        
        return AnalysisResponse(
            success=result.success,
            data={
                "emotion": result.emotion,
                "confidence": result.confidence,
                "emotions": result.emotions,
                "processing_time": result.processing_time,
                "model_used": result.model_used,
                "platform": "railway",
                "note": "Audio analysis is mocked on Railway free tier"
            },
            error=result.error
        )
        
    except Exception as e:
        logger.error(f"Audio analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Audio analysis failed: {str(e)}"
        )

@app.post("/analyze/video", response_model=AnalysisResponse)
async def analyze_video(file: UploadFile = File(...)):
    """Analyze emotion in video (mock on Railway free tier)"""
    if not MODELS_AVAILABLE or video_model is None:
        raise HTTPException(
            status_code=503,
            detail="Video analysis model not available"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Mock analysis (real analysis would require more resources)
        result = await video_model.analyze_emotion(content)
        
        return AnalysisResponse(
            success=result.success,
            data={
                "emotion": result.emotion,
                "confidence": result.confidence,
                "emotions": result.emotions,
                "processing_time": result.processing_time,
                "model_used": result.model_used,
                "platform": "railway",
                "note": "Video analysis is mocked on Railway free tier"
            },
            error=result.error
        )
        
    except Exception as e:
        logger.error(f"Video analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed: {str(e)}"
        )

@app.get("/metrics")
async def get_metrics():
    """Get system metrics for monitoring"""
    return {
        "platform": "railway",
        "uptime": time.time() - startup_time,
        "memory": get_memory_info(),
        "models_loaded": MODELS_AVAILABLE and text_model is not None,
        "environment": RAILWAY_ENVIRONMENT,
        "port": PORT
    }

# Railway production server configuration
if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"🚂 Starting EmoSense API on Railway")
    logger.info(f"🌍 Port: {PORT}")
    logger.info(f"🔗 Domain: {RAILWAY_PUBLIC_DOMAIN}")
    logger.info(f"📊 Environment: {RAILWAY_ENVIRONMENT}")
    
    # Run with uvicorn optimized for Railway
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        access_log=True,
        reload=False,
        workers=1,  # Single worker for Railway free tier
    )
