"""
Emotion Analysis API Endpoints for EmoSense Backend API

Provides endpoints for analyzing emotions in text, video, and audio content.
Supports both real-time and batch processing with detailed results.
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import FileProcessingError, ModelProcessingError, ValidationError
from app.database import get_db_session
from app.models.emotion import AnalysisType
from app.schemas.emotion import (
    AudioAnalysisRequest,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    DetailedAnalysisResponse,
    EmotionAnalysisResponse,
    TextAnalysisRequest,
    VideoAnalysisRequest,
)
from app.services.emotion.text_analyzer import TextEmotionAnalyzer
from app.services.emotion.video_analyzer import VideoEmotionAnalyzer
from app.services.emotion.audio_analyzer import AudioEmotionAnalyzer
from app.dependencies import get_current_user


# Create router for emotion analysis endpoints
router = APIRouter()


@router.post(
    "/text",
    response_model=EmotionAnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Analyze Text Emotions",
    description="Analyze emotions in text content with optional sentence-level breakdown"
)
async def analyze_text_emotion(
    request: TextAnalysisRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user = Depends(get_current_user)
) -> EmotionAnalysisResponse:
    """
    Analyze emotions in text content.
    
    This endpoint processes text input and returns emotion analysis results
    including dominant emotion, confidence scores, and sentiment analysis.
    
    Args:
        request: Text analysis request with content and options
        db: Database session
        current_user: Authenticated user
        
    Returns:
        EmotionAnalysisResponse: Analysis results with emotion scores
        
    Raises:
        ValidationError: If text content is invalid
        ModelProcessingError: If emotion analysis fails
    """
    try:
        # Initialize text emotion analyzer
        analyzer = TextEmotionAnalyzer(db=db, user_id=current_user.id)
        
        # Perform emotion analysis
        analysis_result = await analyzer.analyze_text(
            text=request.text,
            language=request.language,
            confidence_threshold=request.confidence_threshold,
            segment_analysis=request.segment_analysis
        )
        
        return analysis_result
        
    except ValueError as e:
        raise ValidationError(detail=str(e))
    except Exception as e:
        raise ModelProcessingError(
            detail=f"Text emotion analysis failed: {str(e)}",
            model_name="text-emotion-analyzer"
        )


@router.post(
    "/video",
    response_model=EmotionAnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Analyze Video Emotions",
    description="Analyze emotions in video content with frame-by-frame processing"
)
async def analyze_video_emotion(
    video_file: UploadFile = File(..., description="Video file to analyze"),
    frame_interval: float = Form(default=1.0, description="Frame analysis interval in seconds"),
    confidence_threshold: float = Form(default=0.5, description="Confidence threshold"),
    detect_faces: bool = Form(default=True, description="Enable face detection"),
    max_faces: int = Form(default=5, description="Maximum faces per frame"),
    db: AsyncSession = Depends(get_db_session),
    current_user = Depends(get_current_user)
) -> EmotionAnalysisResponse:
    """
    Analyze emotions in video content.
    
    This endpoint processes uploaded video files and analyzes emotions
    in detected faces across video frames.
    
    Args:
        video_file: Uploaded video file
        frame_interval: Time interval between analyzed frames
        confidence_threshold: Minimum confidence for emotion detection
        detect_faces: Whether to detect faces in video
        max_faces: Maximum number of faces to analyze per frame
        db: Database session
        current_user: Authenticated user
        
    Returns:
        EmotionAnalysisResponse: Analysis results with frame-level emotions
        
    Raises:
        FileProcessingError: If video file processing fails
        ModelProcessingError: If emotion analysis fails
    """
    try:
        # Validate video file
        if not video_file.content_type.startswith('video/'):
            raise FileProcessingError(
                detail="Invalid file type. Only video files are supported.",
                file_name=video_file.filename
            )
        
        # Create video analysis request
        request = VideoAnalysisRequest(
            frame_interval=frame_interval,
            confidence_threshold=confidence_threshold,
            detect_faces=detect_faces,
            max_faces=max_faces
        )
        
        # Initialize video emotion analyzer
        analyzer = VideoEmotionAnalyzer(db=db, user_id=current_user.id)
        
        # Perform emotion analysis
        analysis_result = await analyzer.analyze_video(
            video_file=video_file,
            request=request
        )
        
        return analysis_result
        
    except FileProcessingError:
        raise
    except Exception as e:
        raise ModelProcessingError(
            detail=f"Video emotion analysis failed: {str(e)}",
            model_name="video-emotion-analyzer"
        )


@router.post(
    "/audio",
    response_model=EmotionAnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Analyze Audio Emotions",
    description="Analyze emotions in audio content with segment-level processing"
)
async def analyze_audio_emotion(
    audio_file: UploadFile = File(..., description="Audio file to analyze"),
    segment_duration: float = Form(default=3.0, description="Segment duration in seconds"),
    confidence_threshold: float = Form(default=0.5, description="Confidence threshold"),
    transcribe_speech: bool = Form(default=False, description="Enable speech transcription"),
    language: str = Form(default="auto", description="Speech language"),
    db: AsyncSession = Depends(get_db_session),
    current_user = Depends(get_current_user)
) -> EmotionAnalysisResponse:
    """
    Analyze emotions in audio content.
    
    This endpoint processes uploaded audio files and analyzes emotions
    in audio segments with optional speech transcription.
    
    Args:
        audio_file: Uploaded audio file
        segment_duration: Duration of audio segments to analyze
        confidence_threshold: Minimum confidence for emotion detection
        transcribe_speech: Whether to transcribe speech for text analysis
        language: Language for speech transcription
        db: Database session
        current_user: Authenticated user
        
    Returns:
        EmotionAnalysisResponse: Analysis results with segment-level emotions
        
    Raises:
        FileProcessingError: If audio file processing fails
        ModelProcessingError: If emotion analysis fails
    """
    try:
        # Validate audio file
        if not audio_file.content_type.startswith('audio/'):
            raise FileProcessingError(
                detail="Invalid file type. Only audio files are supported.",
                file_name=audio_file.filename
            )
        
        # Create audio analysis request
        request = AudioAnalysisRequest(
            segment_duration=segment_duration,
            confidence_threshold=confidence_threshold,
            transcribe_speech=transcribe_speech,
            language=language
        )
        
        # Initialize audio emotion analyzer
        analyzer = AudioEmotionAnalyzer(db=db, user_id=current_user.id)
        
        # Perform emotion analysis
        analysis_result = await analyzer.analyze_audio(
            audio_file=audio_file,
            request=request
        )
        
        return analysis_result
        
    except FileProcessingError:
        raise
    except Exception as e:
        raise ModelProcessingError(
            detail=f"Audio emotion analysis failed: {str(e)}",
            model_name="audio-emotion-analyzer"
        )


@router.post(
    "/batch",
    response_model=BatchAnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Batch Emotion Analysis",
    description="Process multiple text inputs for emotion analysis in batch"
)
async def batch_analyze_emotions(
    request: BatchAnalysisRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user = Depends(get_current_user)
) -> BatchAnalysisResponse:
    """
    Process multiple inputs for emotion analysis.
    
    This endpoint accepts multiple text inputs and processes them
    in batch for efficient emotion analysis.
    
    Args:
        request: Batch analysis request with multiple inputs
        db: Database session
        current_user: Authenticated user
        
    Returns:
        BatchAnalysisResponse: Batch processing results
        
    Raises:
        ValidationError: If batch request is invalid
        ModelProcessingError: If batch processing fails
    """
    try:
        # Validate batch request
        if len(request.inputs) > 100:
            raise ValidationError(detail="Maximum 100 inputs allowed per batch")
        
        # Initialize appropriate analyzer based on analysis type
        if request.analysis_type == AnalysisType.TEXT:
            analyzer = TextEmotionAnalyzer(db=db, user_id=current_user.id)
            batch_result = await analyzer.batch_analyze_text(
                texts=request.inputs,
                confidence_threshold=request.confidence_threshold
            )
        else:
            raise ValidationError(
                detail=f"Batch analysis not supported for {request.analysis_type}"
            )
        
        return batch_result
        
    except ValidationError:
        raise
    except Exception as e:
        raise ModelProcessingError(
            detail=f"Batch emotion analysis failed: {str(e)}",
            model_name="batch-emotion-analyzer"
        )


@router.get(
    "/{analysis_id}",
    response_model=DetailedAnalysisResponse,
    summary="Get Analysis Results",
    description="Retrieve detailed emotion analysis results by ID"
)
async def get_analysis_results(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user = Depends(get_current_user)
) -> DetailedAnalysisResponse:
    """
    Get detailed emotion analysis results.
    
    Retrieves comprehensive analysis results including segment-level
    or frame-level details depending on the analysis type.
    
    Args:
        analysis_id: Unique analysis identifier
        db: Database session
        current_user: Authenticated user
        
    Returns:
        DetailedAnalysisResponse: Detailed analysis results
        
    Raises:
        HTTPException: If analysis not found or access denied
    """
    try:
        from app.services.emotion.analysis_service import EmotionAnalysisService
        
        analysis_service = EmotionAnalysisService(db=db)
        analysis = await analysis_service.get_analysis_by_id(
            analysis_id=analysis_id,
            user_id=current_user.id
        )
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve analysis: {str(e)}"
        )


@router.get(
    "/",
    response_model=List[EmotionAnalysisResponse],
    summary="List User Analyses",
    description="Get a list of emotion analyses for the current user"
)
async def list_user_analyses(
    skip: int = 0,
    limit: int = 20,
    analysis_type: AnalysisType = None,
    db: AsyncSession = Depends(get_db_session),
    current_user = Depends(get_current_user)
) -> List[EmotionAnalysisResponse]:
    """
    List emotion analyses for the current user.
    
    Returns a paginated list of the user's emotion analysis results
    with optional filtering by analysis type.
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        analysis_type: Optional filter by analysis type
        db: Database session
        current_user: Authenticated user
        
    Returns:
        List[EmotionAnalysisResponse]: List of user's analyses
    """
    try:
        from app.services.emotion.analysis_service import EmotionAnalysisService
        
        analysis_service = EmotionAnalysisService(db=db)
        analyses = await analysis_service.get_user_analyses(
            user_id=current_user.id,
            skip=skip,
            limit=min(limit, 100),  # Cap at 100
            analysis_type=analysis_type
        )
        
        return analyses
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve analyses: {str(e)}"
        )
