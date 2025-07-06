"""
Emotion Analysis Pydantic Schemas for EmoSense Backend API

Defines request and response schemas for emotion analysis endpoints
using Pydantic for data validation and serialization.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator

from app.models.emotion import AnalysisStatus, AnalysisType, EmotionLabel


class EmotionScores(BaseModel):
    """Schema for emotion confidence scores."""
    
    joy: float = Field(ge=0.0, le=1.0, description="Joy confidence score")
    sadness: float = Field(ge=0.0, le=1.0, description="Sadness confidence score")
    anger: float = Field(ge=0.0, le=1.0, description="Anger confidence score")
    fear: float = Field(ge=0.0, le=1.0, description="Fear confidence score")
    surprise: float = Field(ge=0.0, le=1.0, description="Surprise confidence score")
    disgust: float = Field(ge=0.0, le=1.0, description="Disgust confidence score")
    love: float = Field(ge=0.0, le=1.0, description="Love confidence score")
    excitement: float = Field(ge=0.0, le=1.0, description="Excitement confidence score")
    neutral: float = Field(ge=0.0, le=1.0, description="Neutral confidence score")


class TextAnalysisRequest(BaseModel):
    """Schema for text emotion analysis request."""
    
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=10000,
        description="Text content to analyze for emotions"
    )
    
    language: Optional[str] = Field(
        default="auto",
        description="Language code (ISO 639-1) or 'auto' for detection"
    )
    
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for emotion detection"
    )
    
    segment_analysis: bool = Field(
        default=False,
        description="Whether to perform sentence-level analysis"
    )
    
    @validator('text')
    def validate_text_content(cls, v):
        """Validate text content."""
        if not v.strip():
            raise ValueError('Text cannot be empty or contain only whitespace')
        return v.strip()


class VideoAnalysisRequest(BaseModel):
    """Schema for video emotion analysis request."""
    
    frame_interval: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Interval between frames to analyze (in seconds)"
    )
    
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for emotion detection"
    )
    
    detect_faces: bool = Field(
        default=True,
        description="Whether to detect and analyze faces in video"
    )
    
    max_faces: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of faces to analyze per frame"
    )


class AudioAnalysisRequest(BaseModel):
    """Schema for audio emotion analysis request."""
    
    segment_duration: float = Field(
        default=3.0,
        ge=1.0,
        le=30.0,
        description="Duration of audio segments to analyze (in seconds)"
    )
    
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for emotion detection"
    )
    
    transcribe_speech: bool = Field(
        default=False,
        description="Whether to transcribe speech for text analysis"
    )
    
    language: Optional[str] = Field(
        default="auto",
        description="Language code for speech transcription"
    )


class EmotionAnalysisResponse(BaseModel):
    """Schema for emotion analysis response."""
    
    id: UUID = Field(..., description="Unique analysis identifier")
    analysis_type: AnalysisType = Field(..., description="Type of analysis performed")
    status: AnalysisStatus = Field(..., description="Current processing status")
    
    # Input information
    input_text: Optional[str] = Field(None, description="Original text input")
    input_file_name: Optional[str] = Field(None, description="Original filename")
    input_file_size: Optional[int] = Field(None, description="File size in bytes")
    
    # Results
    dominant_emotion: Optional[EmotionLabel] = Field(None, description="Primary detected emotion")
    dominant_emotion_confidence: Optional[float] = Field(None, description="Confidence score")
    emotion_scores: Optional[Dict[str, float]] = Field(None, description="All emotion scores")
    emotion_percentages: Optional[Dict[str, float]] = Field(None, description="Emotion scores as percentages")
    
    # Additional metrics
    sentiment_score: Optional[float] = Field(None, description="Overall sentiment (-1 to 1)")
    arousal_score: Optional[float] = Field(None, description="Arousal/intensity (0 to 1)")
    valence_score: Optional[float] = Field(None, description="Valence/pleasantness (-1 to 1)")
    
    # Processing information
    processing_duration: Optional[float] = Field(None, description="Processing time in seconds")
    model_version: Optional[str] = Field(None, description="ML model version used")
    
    # Timestamps
    created_at: datetime = Field(..., description="Analysis request timestamp")
    completed_at: Optional[datetime] = Field(None, description="Processing completion timestamp")
    
    # Error information (if applicable)
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class TextSegmentResult(BaseModel):
    """Schema for individual text segment analysis result."""
    
    segment_index: int = Field(..., description="Index of this segment")
    segment_text: str = Field(..., description="Text content of this segment")
    start_position: Optional[int] = Field(None, description="Start character position")
    end_position: Optional[int] = Field(None, description="End character position")
    dominant_emotion: Optional[EmotionLabel] = Field(None, description="Primary emotion")
    emotion_scores: Optional[Dict[str, float]] = Field(None, description="Emotion scores")
    confidence: Optional[float] = Field(None, description="Overall confidence")


class VideoFrameResult(BaseModel):
    """Schema for individual video frame analysis result."""
    
    frame_number: int = Field(..., description="Frame number in video")
    timestamp: float = Field(..., description="Timestamp in seconds")
    faces_detected: int = Field(..., description="Number of faces detected")
    face_locations: Optional[List[Dict]] = Field(None, description="Face bounding boxes")
    emotions_per_face: Optional[List[Dict]] = Field(None, description="Emotions for each face")
    dominant_emotion: Optional[EmotionLabel] = Field(None, description="Overall frame emotion")


class AudioSegmentResult(BaseModel):
    """Schema for individual audio segment analysis result."""
    
    segment_index: int = Field(..., description="Index of this segment")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    audio_features: Optional[Dict] = Field(None, description="Extracted audio features")
    dominant_emotion: Optional[EmotionLabel] = Field(None, description="Primary emotion")
    emotion_scores: Optional[Dict[str, float]] = Field(None, description="Emotion scores")
    transcribed_text: Optional[str] = Field(None, description="Transcribed speech text")


class DetailedAnalysisResponse(EmotionAnalysisResponse):
    """Schema for detailed analysis response with segment-level results."""
    
    text_segments: Optional[List[TextSegmentResult]] = Field(None, description="Text segment results")
    video_frames: Optional[List[VideoFrameResult]] = Field(None, description="Video frame results")
    audio_segments: Optional[List[AudioSegmentResult]] = Field(None, description="Audio segment results")


class BatchAnalysisRequest(BaseModel):
    """Schema for batch analysis request."""
    
    analysis_type: AnalysisType = Field(..., description="Type of analysis to perform")
    inputs: List[str] = Field(
        ..., 
        min_items=1, 
        max_items=100,
        description="List of text inputs or file paths"
    )
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    
    @validator('inputs')
    def validate_inputs(cls, v):
        """Validate batch inputs."""
        if not v:
            raise ValueError('At least one input is required')
        return v


class BatchAnalysisResponse(BaseModel):
    """Schema for batch analysis response."""
    
    batch_id: UUID = Field(..., description="Unique batch identifier")
    total_items: int = Field(..., description="Total number of items to process")
    completed_items: int = Field(..., description="Number of completed items")
    failed_items: int = Field(..., description="Number of failed items")
    status: str = Field(..., description="Overall batch status")
    results: List[EmotionAnalysisResponse] = Field(..., description="Individual analysis results")
    created_at: datetime = Field(..., description="Batch creation timestamp")
    
    class Config:
        from_attributes = True


class AnalysisListResponse(BaseModel):
    """Schema for paginated analysis list response."""
    
    analyses: List[EmotionAnalysisResponse] = Field(..., description="List of analyses")
    total: int = Field(..., description="Total number of analyses")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Number of analyses per page")
    pages: int = Field(..., description="Total number of pages")
    
    class Config:
        from_attributes = True


class AnalysisStatsResponse(BaseModel):
    """Schema for analysis statistics response."""
    
    total_analyses: int = Field(..., description="Total number of analyses")
    analyses_by_type: Dict[str, int] = Field(..., description="Count by analysis type")
    analyses_by_status: Dict[str, int] = Field(..., description="Count by status")
    analyses_by_emotion: Dict[str, int] = Field(..., description="Count by dominant emotion")
    average_processing_time: float = Field(..., description="Average processing time")
    analyses_this_week: int = Field(..., description="Analyses performed this week")
    analyses_this_month: int = Field(..., description="Analyses performed this month")
    
    class Config:
        from_attributes = True
