"""
Emotion Analysis Models for EmoSense Backend API

Defines SQLAlchemy models for storing emotion analysis results,
including text, video, and audio analysis data.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Enum as SQLEnum, Float, 
    ForeignKey, Integer, JSON, String, Text
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.database import Base


class AnalysisType(str, Enum):
    """Enumeration of supported analysis types."""
    TEXT = "text"
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"


class AnalysisStatus(str, Enum):
    """Enumeration of analysis processing statuses."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EmotionLabel(str, Enum):
    """Standard emotion labels used across the system."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    LOVE = "love"
    EXCITEMENT = "excitement"
    NEUTRAL = "neutral"


class EmotionAnalysis(Base):
    """
    Main emotion analysis model for storing analysis results.
    
    This model stores the core information about emotion analysis
    requests and their results across different input types.
    """
    
    __tablename__ = "emotion_analyses"
    
    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
        doc="Unique analysis identifier"
    )
    
    # User relationship
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="User who requested the analysis"
    )
    
    # Analysis metadata
    analysis_type = Column(
        SQLEnum(AnalysisType),
        nullable=False,
        index=True,
        doc="Type of analysis performed"
    )
    
    status = Column(
        SQLEnum(AnalysisStatus),
        default=AnalysisStatus.PENDING,
        nullable=False,
        index=True,
        doc="Current processing status"
    )
    
    # Input data
    input_text = Column(
        Text,
        nullable=True,
        doc="Original text input for text analysis"
    )
    
    input_file_path = Column(
        String(500),
        nullable=True,
        doc="Path to uploaded file (video/audio)"
    )
    
    input_file_name = Column(
        String(255),
        nullable=True,
        doc="Original filename of uploaded file"
    )
    
    input_file_size = Column(
        Integer,
        nullable=True,
        doc="File size in bytes"
    )
    
    # Processing information
    processing_duration = Column(
        Float,
        nullable=True,
        doc="Processing time in seconds"
    )
    
    model_version = Column(
        String(50),
        nullable=True,
        doc="Version of the ML model used"
    )
    
    confidence_threshold = Column(
        Float,
        default=0.5,
        nullable=False,
        doc="Confidence threshold used for predictions"
    )
    
    # Results
    dominant_emotion = Column(
        SQLEnum(EmotionLabel),
        nullable=True,
        doc="Primary detected emotion"
    )
    
    dominant_emotion_confidence = Column(
        Float,
        nullable=True,
        doc="Confidence score for dominant emotion"
    )
    
    emotion_scores = Column(
        JSON,
        nullable=True,
        doc="Dictionary of all emotion scores"
    )
    
    # Additional analysis results
    sentiment_score = Column(
        Float,
        nullable=True,
        doc="Overall sentiment score (-1 to 1)"
    )
    
    arousal_score = Column(
        Float,
        nullable=True,
        doc="Arousal/intensity score (0 to 1)"
    )
    
    valence_score = Column(
        Float,
        nullable=True,
        doc="Valence/pleasantness score (-1 to 1)"
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        doc="Analysis request timestamp"
    )
    
    started_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Processing start timestamp"
    )
    
    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Processing completion timestamp"
    )
    
    # Error information
    error_message = Column(
        Text,
        nullable=True,
        doc="Error message if processing failed"
    )
    
    error_details = Column(
        JSON,
        nullable=True,
        doc="Detailed error information"
    )
    
    # Relationships
    user = relationship(
        "User",
        back_populates="emotion_analyses",
        doc="User who requested the analysis"
    )
    
    text_segments = relationship(
        "TextSegmentAnalysis",
        back_populates="analysis",
        cascade="all, delete-orphan",
        doc="Text segment analysis results"
    )
    
    video_frames = relationship(
        "VideoFrameAnalysis", 
        back_populates="analysis",
        cascade="all, delete-orphan",
        doc="Video frame analysis results"
    )
    
    audio_segments = relationship(
        "AudioSegmentAnalysis",
        back_populates="analysis", 
        cascade="all, delete-orphan",
        doc="Audio segment analysis results"
    )
    
    def __repr__(self) -> str:
        """String representation of the analysis."""
        return f"<EmotionAnalysis(id={self.id}, type={self.analysis_type}, status={self.status})>"
    
    def update_status(self, status: AnalysisStatus, error_message: str = None) -> None:
        """Update analysis status with optional error message."""
        self.status = status
        if status == AnalysisStatus.PROCESSING and not self.started_at:
            self.started_at = datetime.utcnow()
        elif status in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED]:
            self.completed_at = datetime.utcnow()
            if self.started_at:
                self.processing_duration = (
                    self.completed_at - self.started_at
                ).total_seconds()
        
        if error_message:
            self.error_message = error_message
    
    def get_emotion_percentages(self) -> Dict[str, float]:
        """Get emotion scores as percentages."""
        if not self.emotion_scores:
            return {}
        
        return {
            emotion: round(score * 100, 2)
            for emotion, score in self.emotion_scores.items()
        }
    
    def to_dict(self) -> dict:
        """Convert analysis to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "analysis_type": self.analysis_type.value if self.analysis_type else None,
            "status": self.status.value if self.status else None,
            "input_text": self.input_text,
            "input_file_name": self.input_file_name,
            "input_file_size": self.input_file_size,
            "processing_duration": self.processing_duration,
            "model_version": self.model_version,
            "dominant_emotion": self.dominant_emotion.value if self.dominant_emotion else None,
            "dominant_emotion_confidence": self.dominant_emotion_confidence,
            "emotion_scores": self.emotion_scores,
            "emotion_percentages": self.get_emotion_percentages(),
            "sentiment_score": self.sentiment_score,
            "arousal_score": self.arousal_score,
            "valence_score": self.valence_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
        }


class TextSegmentAnalysis(Base):
    """
    Model for storing emotion analysis of individual text segments.
    
    Used when analyzing longer texts that are split into sentences
    or paragraphs for more granular emotion detection.
    """
    
    __tablename__ = "text_segment_analyses"
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Unique segment analysis identifier"
    )
    
    analysis_id = Column(
        UUID(as_uuid=True),
        ForeignKey("emotion_analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Parent analysis ID"
    )
    
    segment_index = Column(
        Integer,
        nullable=False,
        doc="Index of this segment in the original text"
    )
    
    segment_text = Column(
        Text,
        nullable=False,
        doc="Text content of this segment"
    )
    
    start_position = Column(
        Integer,
        nullable=True,
        doc="Start character position in original text"
    )
    
    end_position = Column(
        Integer,
        nullable=True,
        doc="End character position in original text"
    )
    
    dominant_emotion = Column(
        SQLEnum(EmotionLabel),
        nullable=True,
        doc="Primary detected emotion for this segment"
    )
    
    emotion_scores = Column(
        JSON,
        nullable=True,
        doc="Emotion scores for this segment"
    )
    
    confidence = Column(
        Float,
        nullable=True,
        doc="Overall confidence for this segment"
    )
    
    analysis = relationship(
        "EmotionAnalysis",
        back_populates="text_segments"
    )


class VideoFrameAnalysis(Base):
    """
    Model for storing emotion analysis of individual video frames.
    """
    
    __tablename__ = "video_frame_analyses"
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    analysis_id = Column(
        UUID(as_uuid=True),
        ForeignKey("emotion_analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False, doc="Timestamp in seconds")
    
    # Face detection results
    faces_detected = Column(Integer, default=0)
    face_locations = Column(JSON, nullable=True)
    
    # Emotion results per face
    emotions_per_face = Column(JSON, nullable=True)
    dominant_emotion = Column(SQLEnum(EmotionLabel), nullable=True)
    
    analysis = relationship("EmotionAnalysis", back_populates="video_frames")


class AudioSegmentAnalysis(Base):
    """
    Model for storing emotion analysis of audio segments.
    """
    
    __tablename__ = "audio_segment_analyses"
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    analysis_id = Column(
        UUID(as_uuid=True),
        ForeignKey("emotion_analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    segment_index = Column(Integer, nullable=False)
    start_time = Column(Float, nullable=False, doc="Start time in seconds")
    end_time = Column(Float, nullable=False, doc="End time in seconds")
    
    # Audio features
    audio_features = Column(JSON, nullable=True)
    
    # Emotion results
    dominant_emotion = Column(SQLEnum(EmotionLabel), nullable=True)
    emotion_scores = Column(JSON, nullable=True)
    
    # Speech-to-text results (if applicable)
    transcribed_text = Column(Text, nullable=True)
    
    analysis = relationship("EmotionAnalysis", back_populates="audio_segments")
