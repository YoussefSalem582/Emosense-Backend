"""
Production-ready FastAPI server for global deployment
Optimized for cloud platforms like Railway, Render, Google Cloud Run
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

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the api directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

# Production environment variables
PORT = int(os.getenv("PORT", 8000))
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
MODEL_CACHE_DIR = os.getenv("EMOTION_MODEL_CACHE_DIR", "/tmp/model_cache")

# Create cache directory
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Import emotion models with error handling
try:
    from emotion_models import (
        EmotionAnalysisRequest,
        EmotionResult,
        TextEmotionModel,
        AudioEmotionModel,
        VideoEmotionModel,
    )
    MODELS_AVAILABLE = True
    logger.info("✅ Emotion models imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import emotion models: {e}")
    MODELS_AVAILABLE = False

# Create FastAPI app with production settings
app = FastAPI(
    title="EmoSense Emotion Analysis API",
    description="Production-ready emotion analysis for text, audio, and video - Deployed Globally",
    version="2.0.0",
    docs_url="/docs" if ENVIRONMENT != "production" else None,  # Disable docs in production
    redoc_url="/redoc" if ENVIRONMENT != "production" else None,
)

# Add CORS middleware for global access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for global access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class TextAnalysisRequest(BaseModel):
    text: str
    analysis_type: str = "text"

class TextAnalysisResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    models_available: bool
    uptime: float
    deployment: str = "global"

# Global variables
text_model = None
audio_model = None
video_model = None
startup_time = time.time()

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global text_model, audio_model, video_model
    
    logger.info("🚀 Starting EmoSense API server for global deployment...")
    logger.info(f"📊 Environment: {ENVIRONMENT}")
    logger.info(f"🌍 Port: {PORT}")
    logger.info(f"💾 Cache Dir: {MODEL_CACHE_DIR}")
    
    if not MODELS_AVAILABLE:
        logger.warning("⚠️ Emotion models not available, running in demo mode")
        return
    
    try:
        # Initialize text model (primary)
        logger.info("🔄 Loading text emotion model...")
        text_model = TextEmotionModel()
        await text_model.load_model()
        logger.info("✅ Text emotion model loaded")
        
        # Initialize audio model
        logger.info("🔄 Loading audio emotion model...")
        audio_model = AudioEmotionModel()
        await audio_model.load_model()
        logger.info("✅ Audio emotion model loaded")
        
        # Initialize video model
        logger.info("🔄 Loading video emotion model...")
        video_model = VideoEmotionModel()
        await video_model.load_model()
        logger.info("✅ Video emotion model loaded")
        
        logger.info("🎉 All emotion models ready for global access!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        # Continue anyway with fallback models

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with global deployment info"""
    return {
        "message": "EmoSense Emotion Analysis API - Global Deployment",
        "version": "2.0.0",
        "status": "running",
        "deployment": "global",
        "docs": "/docs" if ENVIRONMENT != "production" else "disabled",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        uptime = time.time() - startup_time
        
        models_status = False
        if MODELS_AVAILABLE:
            models_status = all([
                text_model is not None and text_model.is_loaded,
                audio_model is not None and audio_model.is_loaded,
                video_model is not None and video_model.is_loaded
            ])
        
        return HealthResponse(
            status="healthy",
            version="2.0.0",
            environment=ENVIRONMENT,
            models_available=models_status,
            uptime=round(uptime, 2),
            deployment="global"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/api/v1/analyze/text", response_model=TextAnalysisResponse)
async def analyze_text_emotion(request: TextAnalysisRequest):
    """
    Analyze emotion in text - Global endpoint
    """
    try:
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if not MODELS_AVAILABLE or not text_model:
            # Fallback demo response for when models aren't available
            emotions = {
                "joy": 0.3, "sadness": 0.1, "anger": 0.1,
                "fear": 0.1, "surprise": 0.1, "disgust": 0.1, "neutral": 0.2
            }
            dominant_emotion = max(emotions, key=emotions.get)
            
            response_data = {
                "dominant_emotion": dominant_emotion,
                "confidence": emotions[dominant_emotion],
                "emotions": emotions,
                "processing_time": 0.001,
                "model_version": "2.0.0_demo_fallback",
                "text_length": len(request.text),
                "timestamp": time.time(),
                "deployment": "global"
            }
            
            return TextAnalysisResponse(success=True, data=response_data)
        
        # Real model analysis
        analysis_request = EmotionAnalysisRequest(
            text=request.text,
            analysis_type="text"
        )
        
        result = await text_model.analyze(analysis_request)
        
        response_data = {
            "dominant_emotion": result.dominant_emotion,
            "confidence": round(result.confidence, 3),
            "emotions": {k: round(v, 3) for k, v in result.emotions.items()},
            "processing_time": round(result.processing_time, 3),
            "model_version": result.model_version,
            "text_length": len(request.text),
            "timestamp": time.time(),
            "deployment": "global"
        }
        
        logger.info(f"Global analysis: '{request.text[:50]}...' -> {result.dominant_emotion} ({result.confidence:.1%})")
        
        return TextAnalysisResponse(success=True, data=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text analysis failed: {e}")
        return TextAnalysisResponse(
            success=False,
            error=f"Analysis failed: {str(e)}"
        )

@app.get("/api/v1/models")
async def get_available_models():
    """Get information about available models - Global endpoint"""
    try:
        if not MODELS_AVAILABLE:
            return JSONResponse(content={
                "success": True,
                "data": {
                    "models": {
                        "text": {"available": False, "loaded": False, "status": "demo_mode"},
                        "audio": {"available": False, "loaded": False, "status": "demo_mode"},
                        "video": {"available": False, "loaded": False, "status": "demo_mode"}
                    },
                    "api_version": "2.0.0",
                    "deployment": "global",
                    "status": "demo_mode"
                }
            })
        
        models_info = {
            "text": {
                "available": text_model is not None,
                "loaded": text_model.is_loaded if text_model else False,
                "name": text_model.model_name if text_model else "text_emotion_model",
                "version": text_model.version if text_model else "unknown",
                "supported_emotions": text_model.get_supported_emotions() if text_model else []
            },
            "audio": {
                "available": audio_model is not None,
                "loaded": audio_model.is_loaded if audio_model else False,
                "name": audio_model.model_name if audio_model else "audio_emotion_model",
                "version": audio_model.version if audio_model else "unknown",
                "supported_emotions": audio_model.get_supported_emotions() if audio_model else []
            },
            "video": {
                "available": video_model is not None,
                "loaded": video_model.is_loaded if video_model else False,
                "name": video_model.model_name if video_model else "video_emotion_model",
                "version": video_model.version if video_model else "unknown",
                "supported_emotions": video_model.get_supported_emotions() if video_model else []
            }
        }
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "models": models_info,
                "api_version": "2.0.0",
                "deployment": "global",
                "total_models": len([m for m in models_info.values() if m["available"]])
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get models info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get models information")

@app.post("/api/v1/analyze/batch")
async def analyze_batch_text(texts: list[str]):
    """Batch analyze multiple texts - Global endpoint"""
    try:
        if not texts or len(texts) == 0:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        if len(texts) > 50:  # Reduced limit for global deployment
            raise HTTPException(status_code=400, detail="Too many texts (max 50 for global deployment)")
        
        results = []
        total_time = 0
        
        for i, text in enumerate(texts):
            if not text or len(text.strip()) == 0:
                results.append({
                    "index": i,
                    "success": False,
                    "error": "Empty text"
                })
                continue
            
            try:
                if not MODELS_AVAILABLE or not text_model:
                    # Demo fallback
                    emotions = {"joy": 0.3, "neutral": 0.7}
                    dominant = "neutral"
                    confidence = 0.7
                    processing_time = 0.001
                    model_version = "demo_fallback"
                else:
                    # Real analysis
                    analysis_request = EmotionAnalysisRequest(text=text, analysis_type="text")
                    result = await text_model.analyze(analysis_request)
                    emotions = result.emotions
                    dominant = result.dominant_emotion
                    confidence = result.confidence
                    processing_time = result.processing_time
                    model_version = result.model_version
                
                total_time += processing_time
                
                results.append({
                    "index": i,
                    "success": True,
                    "text": text[:100],
                    "dominant_emotion": dominant,
                    "confidence": round(confidence, 3),
                    "emotions": {k: round(v, 3) for k, v in emotions.items()},
                    "processing_time": round(processing_time, 3)
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e)
                })
        
        successful_analyses = len([r for r in results if r["success"]])
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "results": results,
                "summary": {
                    "total_texts": len(texts),
                    "successful": successful_analyses,
                    "failed": len(texts) - successful_analyses,
                    "total_processing_time": round(total_time, 3),
                    "average_time": round(total_time / successful_analyses, 3) if successful_analyses > 0 else 0
                },
                "deployment": "global"
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        return JSONResponse(content={
            "success": False,
            "error": f"Batch analysis failed: {str(e)}"
        })

# Production server configuration
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "production_server:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )
