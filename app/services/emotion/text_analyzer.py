"""
Text Emotion Analyzer Service for EmoSense Backend API

Implements text emotion analysis using transformer models (RoBERTa-based)
for detecting emotions in text content with high accuracy.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.emotion import (
    AnalysisStatus, 
    AnalysisType, 
    EmotionAnalysis, 
    EmotionLabel,
    TextSegmentAnalysis
)
from app.schemas.emotion import EmotionAnalysisResponse, TextSegmentResult
from app.config import get_settings
from app.core.exceptions import ModelProcessingError


settings = get_settings()


class TextEmotionAnalyzer:
    """
    Text emotion analysis service using transformer models.
    
    This service processes text input and returns emotion predictions
    with confidence scores for multiple emotion categories.
    """
    
    def __init__(self, db: AsyncSession, user_id: UUID):
        """
        Initialize text emotion analyzer.
        
        Args:
            db: Database session
            user_id: User requesting the analysis
        """
        self.db = db
        self.user_id = user_id
        self.model_name = settings.TEXT_MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._emotion_pipeline = None
        self._tokenizer = None
        self._model = None
    
    @property
    async def emotion_pipeline(self):
        """Lazy load emotion analysis pipeline."""
        if self._emotion_pipeline is None:
            try:
                self._emotion_pipeline = pipeline(
                    "text-classification",
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1,
                    return_all_scores=True
                )
            except Exception as e:
                # Fallback to a basic model if the specified one fails
                self._emotion_pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=0 if self.device == "cuda" else -1,
                    return_all_scores=True
                )
        return self._emotion_pipeline
    
    def _map_emotion_labels(self, model_output: List[Dict]) -> Dict[str, float]:
        """
        Map model output to standardized emotion labels.
        
        Args:
            model_output: Raw model output with labels and scores
            
        Returns:
            Dict with standardized emotion labels and scores
        """
        # Mapping from common model labels to our standardized labels
        label_mapping = {
            'joy': EmotionLabel.JOY,
            'happiness': EmotionLabel.JOY,
            'sadness': EmotionLabel.SADNESS,
            'anger': EmotionLabel.ANGER,
            'fear': EmotionLabel.FEAR,
            'surprise': EmotionLabel.SURPRISE,
            'disgust': EmotionLabel.DISGUST,
            'love': EmotionLabel.LOVE,
            'excitement': EmotionLabel.EXCITEMENT,
            'neutral': EmotionLabel.NEUTRAL,
            'optimism': EmotionLabel.JOY,
            'pessimism': EmotionLabel.SADNESS,
        }
        
        # Initialize all emotions with 0 score
        emotion_scores = {label.value: 0.0 for label in EmotionLabel}
        
        # Map model outputs to our labels
        for prediction in model_output:
            label = prediction['label'].lower()
            score = prediction['score']
            
            # Map to our standardized label
            if label in label_mapping:
                standard_label = label_mapping[label].value
                emotion_scores[standard_label] = max(emotion_scores[standard_label], score)
        
        return emotion_scores
    
    def _calculate_sentiment_metrics(self, emotion_scores: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Calculate sentiment, arousal, and valence scores from emotions.
        
        Args:
            emotion_scores: Dictionary of emotion scores
            
        Returns:
            Tuple of (sentiment, arousal, valence) scores
        """
        # Sentiment: positive emotions vs negative emotions
        positive_emotions = [
            EmotionLabel.JOY.value,
            EmotionLabel.LOVE.value,
            EmotionLabel.EXCITEMENT.value
        ]
        negative_emotions = [
            EmotionLabel.SADNESS.value,
            EmotionLabel.ANGER.value,
            EmotionLabel.FEAR.value,
            EmotionLabel.DISGUST.value
        ]
        
        positive_score = sum(emotion_scores.get(emotion, 0) for emotion in positive_emotions)
        negative_score = sum(emotion_scores.get(emotion, 0) for emotion in negative_emotions)
        
        sentiment = positive_score - negative_score  # Range: -1 to 1
        
        # Arousal: high arousal emotions vs low arousal emotions
        high_arousal = [
            EmotionLabel.ANGER.value,
            EmotionLabel.FEAR.value,
            EmotionLabel.EXCITEMENT.value,
            EmotionLabel.SURPRISE.value
        ]
        arousal = sum(emotion_scores.get(emotion, 0) for emotion in high_arousal)
        
        # Valence: similar to sentiment but normalized
        valence = sentiment
        
        return sentiment, min(arousal, 1.0), valence
    
    def _get_dominant_emotion(self, emotion_scores: Dict[str, float]) -> Tuple[EmotionLabel, float]:
        """
        Get the dominant emotion and its confidence score.
        
        Args:
            emotion_scores: Dictionary of emotion scores
            
        Returns:
            Tuple of (dominant_emotion, confidence_score)
        """
        if not emotion_scores:
            return EmotionLabel.NEUTRAL, 0.0
        
        dominant_emotion_str = max(emotion_scores.keys(), key=lambda k: emotion_scores[k])
        confidence = emotion_scores[dominant_emotion_str]
        
        # Convert string back to enum
        try:
            dominant_emotion = EmotionLabel(dominant_emotion_str)
        except ValueError:
            dominant_emotion = EmotionLabel.NEUTRAL
            confidence = 0.0
        
        return dominant_emotion, confidence
    
    def _split_text_into_segments(self, text: str, max_length: int = 512) -> List[str]:
        """
        Split text into segments for processing.
        
        Args:
            text: Input text to split
            max_length: Maximum length per segment
            
        Returns:
            List of text segments
        """
        # Simple sentence-based splitting
        import re
        
        # Split by sentences
        sentences = re.split(r'[.!?]+', text)
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed max_length, start new segment
            if len(current_segment) + len(sentence) > max_length and current_segment:
                segments.append(current_segment.strip())
                current_segment = sentence
            else:
                current_segment += " " + sentence if current_segment else sentence
        
        # Add the last segment
        if current_segment.strip():
            segments.append(current_segment.strip())
        
        return segments if segments else [text]
    
    async def analyze_text(
        self,
        text: str,
        language: str = "auto",
        confidence_threshold: float = 0.5,
        segment_analysis: bool = False
    ) -> EmotionAnalysisResponse:
        """
        Analyze emotions in text content.
        
        Args:
            text: Text content to analyze
            language: Language code (currently ignored, auto-detected)
            confidence_threshold: Minimum confidence threshold
            segment_analysis: Whether to perform segment-level analysis
            
        Returns:
            EmotionAnalysisResponse with analysis results
        """
        start_time = time.time()
        
        # Create analysis record
        analysis = EmotionAnalysis(
            user_id=self.user_id,
            analysis_type=AnalysisType.TEXT,
            input_text=text,
            confidence_threshold=confidence_threshold,
            model_version=self.model_name,
            status=AnalysisStatus.PROCESSING
        )
        
        self.db.add(analysis)
        await self.db.commit()
        await self.db.refresh(analysis)
        
        try:
            # Update status to processing
            analysis.update_status(AnalysisStatus.PROCESSING)
            await self.db.commit()
            
            # Get emotion pipeline
            pipeline = await self.emotion_pipeline
            
            # Analyze overall text
            model_results = pipeline(text)
            emotion_scores = self._map_emotion_labels(model_results[0])
            
            # Calculate metrics
            dominant_emotion, confidence = self._get_dominant_emotion(emotion_scores)
            sentiment, arousal, valence = self._calculate_sentiment_metrics(emotion_scores)
            
            # Update analysis with results
            analysis.emotion_scores = emotion_scores
            analysis.dominant_emotion = dominant_emotion
            analysis.dominant_emotion_confidence = confidence
            analysis.sentiment_score = sentiment
            analysis.arousal_score = arousal
            analysis.valence_score = valence
            
            # Perform segment analysis if requested
            if segment_analysis and len(text) > 100:  # Only for longer texts
                segments = self._split_text_into_segments(text)
                
                for i, segment_text in enumerate(segments):
                    segment_results = pipeline(segment_text)
                    segment_scores = self._map_emotion_labels(segment_results[0])
                    segment_emotion, segment_confidence = self._get_dominant_emotion(segment_scores)
                    
                    segment_analysis_obj = TextSegmentAnalysis(
                        analysis_id=analysis.id,
                        segment_index=i,
                        segment_text=segment_text,
                        start_position=text.find(segment_text),
                        end_position=text.find(segment_text) + len(segment_text),
                        dominant_emotion=segment_emotion,
                        emotion_scores=segment_scores,
                        confidence=segment_confidence
                    )
                    
                    self.db.add(segment_analysis_obj)
            
            # Complete analysis
            analysis.update_status(AnalysisStatus.COMPLETED)
            analysis.processing_duration = time.time() - start_time
            
            await self.db.commit()
            await self.db.refresh(analysis)
            
            return EmotionAnalysisResponse.from_orm(analysis)
            
        except Exception as e:
            # Handle errors
            error_message = f"Text emotion analysis failed: {str(e)}"
            analysis.update_status(AnalysisStatus.FAILED, error_message)
            await self.db.commit()
            
            raise ModelProcessingError(
                detail=error_message,
                model_name=self.model_name
            )
    
    async def batch_analyze_text(
        self,
        texts: List[str],
        confidence_threshold: float = 0.5
    ) -> List[EmotionAnalysisResponse]:
        """
        Analyze multiple texts in batch.
        
        Args:
            texts: List of text strings to analyze
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of EmotionAnalysisResponse objects
        """
        results = []
        
        # Process texts concurrently (but limit concurrency to prevent memory issues)
        semaphore = asyncio.Semaphore(5)  # Process max 5 texts concurrently
        
        async def analyze_single_text(text: str) -> EmotionAnalysisResponse:
            async with semaphore:
                return await self.analyze_text(
                    text=text,
                    confidence_threshold=confidence_threshold,
                    segment_analysis=False  # Disable for batch processing
                )
        
        # Create tasks for all texts
        tasks = [analyze_single_text(text) for text in texts]
        
        # Execute tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return successful results
        successful_results = []
        for result in results:
            if isinstance(result, EmotionAnalysisResponse):
                successful_results.append(result)
        
        return successful_results
