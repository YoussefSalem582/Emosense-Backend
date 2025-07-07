"""
FastAPI server for EmoSense emotion analysis
Connects the emotion models to the Flutter app
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

# Add the api directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

from emotion_models import (
    EmotionAnalysisRequest,
    EmotionResult,
    TextEmotionModel,
    AudioEmotionModel,
    VideoEmotionModel,
    model_manager
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="EmoSense Emotion Analysis API",
    description="Production-ready emotion analysis for text, audio, and video",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models for API
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
    models_available: Dict[str, bool]

# Global model instances
text_model = None
audio_model = None
video_model = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global text_model, audio_model, video_model
    
    logger.info("🚀 Starting EmoSense API server...")
    
    try:
        # Initialize text model (primary)
        text_model = TextEmotionModel()
        await text_model.load_model()
        logger.info("✅ Text emotion model loaded")
        
        # Initialize audio model
        audio_model = AudioEmotionModel()
        await audio_model.load_model()
        logger.info("✅ Audio emotion model loaded")
        
        # Initialize video model
        video_model = VideoEmotionModel()
        await video_model.load_model()
        logger.info("✅ Video emotion model loaded")
        
        logger.info("🎉 All emotion models ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        # Continue anyway with fallback models

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "EmoSense Emotion Analysis API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for app monitoring"""
    try:
        models_status = {
            "text_model": text_model is not None and text_model.is_loaded,
            "audio_model": audio_model is not None and audio_model.is_loaded,
            "video_model": video_model is not None and video_model.is_loaded
        }
        
        return HealthResponse(
            status="healthy",
            version="2.0.0",
            models_available=models_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/api/v1/analyze/text", response_model=TextAnalysisResponse)
async def analyze_text_emotion(request: TextAnalysisRequest):
    """
    Analyze emotion in text
    Main endpoint for Flutter app text analysis
    """
    try:
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if not text_model:
            raise HTTPException(status_code=503, detail="Text model not available")
        
        # Create emotion analysis request
        analysis_request = EmotionAnalysisRequest(
            text=request.text,
            analysis_type="text"
        )
        
        # Analyze emotion
        result = await text_model.analyze(analysis_request)
        
        # Format response for Flutter app
        response_data = {
            "dominant_emotion": result.dominant_emotion,
            "confidence": round(result.confidence, 3),
            "emotions": {k: round(v, 3) for k, v in result.emotions.items()},
            "processing_time": round(result.processing_time, 3),
            "model_version": result.model_version,
            "text_length": len(request.text),
            "timestamp": asyncio.get_event_loop().time()
        }
        
        logger.info(f"Text analysis: '{request.text[:50]}...' -> {result.dominant_emotion} ({result.confidence:.1%})")
        
        return TextAnalysisResponse(
            success=True,
            data=response_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text analysis failed: {e}")
        return TextAnalysisResponse(
            success=False,
            error=f"Analysis failed: {str(e)}"
        )

@app.post("/api/v1/analyze/audio")
async def analyze_audio_emotion(
    audio_file: UploadFile = File(...),
    analysis_type: str = Form(default="audio")
):
    """
    Analyze emotion in audio file
    For Flutter app audio analysis
    """
    try:
        if not audio_model:
            raise HTTPException(status_code=503, detail="Audio model not available")
        
        # Read audio file
        audio_data = await audio_file.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty")
        
        # Create emotion analysis request
        analysis_request = EmotionAnalysisRequest(
            audio_data=audio_data,
            analysis_type="audio"
        )
        
        # Analyze emotion
        result = await audio_model.analyze(analysis_request)
        
        # Format response
        response_data = {
            "dominant_emotion": result.dominant_emotion,
            "confidence": round(result.confidence, 3),
            "emotions": {k: round(v, 3) for k, v in result.emotions.items()},
            "processing_time": round(result.processing_time, 3),
            "model_version": result.model_version,
            "file_size": len(audio_data),
            "filename": audio_file.filename,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        logger.info(f"Audio analysis: {audio_file.filename} -> {result.dominant_emotion} ({result.confidence:.1%})")
        
        return JSONResponse(content={
            "success": True,
            "data": response_data
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio analysis failed: {e}")
        return JSONResponse(content={
            "success": False,
            "error": f"Audio analysis failed: {str(e)}"
        })

@app.post("/api/v1/analyze/video")
async def analyze_video_emotion(
    video_file: UploadFile = File(...),
    analysis_type: str = Form(default="video")
):
    """
    Analyze emotion in video file
    For Flutter app video analysis
    """
    try:
        if not video_model:
            raise HTTPException(status_code=503, detail="Video model not available")
        
        # Read video file
        video_data = await video_file.read()
        
        if len(video_data) == 0:
            raise HTTPException(status_code=400, detail="Video file is empty")
        
        # Create emotion analysis request
        analysis_request = EmotionAnalysisRequest(
            video_data=video_data,
            analysis_type="video"
        )
        
        # Analyze emotion
        result = await video_model.analyze(analysis_request)
        
        # Format response
        response_data = {
            "dominant_emotion": result.dominant_emotion,
            "confidence": round(result.confidence, 3),
            "emotions": {k: round(v, 3) for k, v in result.emotions.items()},
            "processing_time": round(result.processing_time, 3),
            "model_version": result.model_version,
            "file_size": len(video_data),
            "filename": video_file.filename,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        logger.info(f"Video analysis: {video_file.filename} -> {result.dominant_emotion} ({result.confidence:.1%})")
        
        return JSONResponse(content={
            "success": True,
            "data": response_data
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video analysis failed: {e}")
        return JSONResponse(content={
            "success": False,
            "error": f"Video analysis failed: {str(e)}"
        })

@app.get("/api/v1/models")
async def get_available_models():
    """
    Get information about available models
    For Flutter app to check capabilities
    """
    try:
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
                "total_models": len([m for m in models_info.values() if m["available"]])
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get models info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get models information")

@app.post("/api/v1/analyze/batch")
async def analyze_batch_text(texts: list[str]):
    """
    Batch analyze multiple texts
    For Flutter app batch processing
    """
    try:
        if not text_model:
            raise HTTPException(status_code=503, detail="Text model not available")
        
        if not texts or len(texts) == 0:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        if len(texts) > 100:  # Limit batch size
            raise HTTPException(status_code=400, detail="Too many texts (max 100)")
        
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
                analysis_request = EmotionAnalysisRequest(
                    text=text,
                    analysis_type="text"
                )
                
                result = await text_model.analyze(analysis_request)
                total_time += result.processing_time
                
                results.append({
                    "index": i,
                    "success": True,
                    "text": text[:100],  # First 100 chars
                    "dominant_emotion": result.dominant_emotion,
                    "confidence": round(result.confidence, 3),
                    "emotions": {k: round(v, 3) for k, v in result.emotions.items()},
                    "processing_time": round(result.processing_time, 3)
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
                }
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app_server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
