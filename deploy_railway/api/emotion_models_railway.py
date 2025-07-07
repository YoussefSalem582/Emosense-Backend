"""
Railway-optimized Emotion Analysis Models
Ultra-lightweight version optimized for Railway's free tier constraints
"""

import asyncio
import logging
import re
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Basic imports that should be available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Configure logging for Railway
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmotionResult:
    """Result of emotion analysis"""
    emotion: str
    confidence: float
    emotions: Dict[str, float]
    processing_time: float
    model_used: str
    success: bool = True
    error: Optional[str] = None

@dataclass
class EmotionAnalysisRequest:
    """Request for emotion analysis"""
    text: str
    analysis_type: str = "text"

class RailwayTextEmotionModel:
    """Ultra-lightweight text emotion analysis optimized for Railway free tier"""
    
    def __init__(self):
        self.sentiment_analyzer = None
        self.model_loaded = False
        self.emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
        
        # Emotion keywords for rule-based analysis
        self.emotion_keywords = {
            "joy": ["happy", "joy", "excited", "pleased", "delighted", "cheerful", "glad", "love", "amazing", "wonderful", "great", "awesome", "fantastic", "excellent", "brilliant"],
            "sadness": ["sad", "depressed", "unhappy", "sorrow", "grief", "melancholy", "blue", "down", "gloomy", "disappointed", "heartbroken", "miserable"],
            "anger": ["angry", "mad", "furious", "rage", "irritated", "annoyed", "upset", "frustrated", "outraged", "livid", "hate", "disgusted"],
            "fear": ["afraid", "scared", "frightened", "terrified", "anxious", "worried", "nervous", "panic", "fearful", "concerned", "uneasy"],
            "surprise": ["surprised", "shocked", "amazed", "astonished", "stunned", "bewildered", "confused", "unexpected", "wow"],
            "disgust": ["disgusting", "revolting", "gross", "sick", "nauseated", "repulsive", "awful", "terrible", "horrible", "yuck"],
            "neutral": ["okay", "fine", "normal", "regular", "standard", "usual", "typical", "average"]
        }
        
    async def load_model(self):
        """Load the emotion analysis model with maximum fallbacks"""
        start_time = time.time()
        
        try:
            # Try NLTK first (lightweight)
            if NLTK_AVAILABLE:
                logger.info("📚 Loading NLTK sentiment analyzer for Railway...")
                try:
                    # Download required NLTK data if not present
                    import ssl
                    try:
                        _create_unverified_https_context = ssl._create_unverified_context
                    except AttributeError:
                        pass
                    else:
                        ssl._create_default_https_context = _create_unverified_https_context
                    
                    try:
                        nltk.data.find('vader_lexicon')
                    except LookupError:
                        logger.info("Downloading NLTK vader_lexicon...")
                        nltk.download('vader_lexicon', quiet=True)
                    
                    self.sentiment_analyzer = SentimentIntensityAnalyzer()
                    self.model_loaded = True
                    logger.info(f"✅ NLTK model loaded in {time.time() - start_time:.2f}s")
                    return True
                except Exception as e:
                    logger.warning(f"NLTK failed: {e}")
            
            # Always have rule-based fallback
            logger.info("🔧 Using rule-based emotion analysis (ultra-lightweight)")
            self.model_loaded = True
            logger.info(f"✅ Rule-based model ready in {time.time() - start_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load any model: {e}")
            return False
    
    def _rule_based_emotion_analysis(self, text: str) -> Dict[str, float]:
        """Simple rule-based emotion analysis using keyword matching"""
        text_lower = text.lower()
        emotion_scores = {emotion: 0.0 for emotion in self.emotions}
        
        total_matches = 0
        for emotion, keywords in self.emotion_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = matches
            total_matches += matches
        
        # Normalize scores
        if total_matches > 0:
            emotion_scores = {k: v / total_matches for k, v in emotion_scores.items()}
        else:
            # Default to neutral if no keywords found
            emotion_scores["neutral"] = 1.0
            
        return emotion_scores
    
    def _nltk_to_emotions(self, sentiment_scores: Dict[str, float]) -> Dict[str, float]:
        """Convert NLTK sentiment to emotion categories"""
        compound = sentiment_scores.get('compound', 0)
        pos = sentiment_scores.get('pos', 0)
        neu = sentiment_scores.get('neu', 0)
        neg = sentiment_scores.get('neg', 0)
        
        emotions = {emotion: 0.0 for emotion in self.emotions}
        
        if compound >= 0.5:
            emotions["joy"] = pos * 0.8 + compound * 0.2
        elif compound <= -0.5:
            if neg > 0.6:
                emotions["anger"] = neg * 0.6
                emotions["sadness"] = neg * 0.4
            else:
                emotions["sadness"] = neg * 0.7
                emotions["fear"] = neg * 0.3
        else:
            emotions["neutral"] = neu
            if pos > 0:
                emotions["joy"] = pos * 0.5
            if neg > 0:
                emotions["sadness"] = neg * 0.3
                emotions["anger"] = neg * 0.2
        
        # Normalize
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v / total for k, v in emotions.items()}
        else:
            emotions["neutral"] = 1.0
            
        return emotions
    
    async def analyze_emotion(self, text: str) -> EmotionResult:
        """Analyze emotion in text using available models"""
        start_time = time.time()
        
        if not self.model_loaded:
            await self.load_model()
        
        try:
            # Clean and validate text
            if not text or not text.strip():
                return EmotionResult(
                    emotion="neutral",
                    confidence=1.0,
                    emotions={"neutral": 1.0},
                    processing_time=time.time() - start_time,
                    model_used="railway_fallback",
                    success=False,
                    error="Empty text provided"
                )
            
            text = text.strip()
            
            # Try NLTK first if available
            if self.sentiment_analyzer:
                try:
                    sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
                    emotions = self._nltk_to_emotions(sentiment_scores)
                    
                    # Get primary emotion
                    primary_emotion = max(emotions, key=emotions.get)
                    confidence = emotions[primary_emotion]
                    
                    return EmotionResult(
                        emotion=primary_emotion,
                        confidence=confidence,
                        emotions=emotions,
                        processing_time=time.time() - start_time,
                        model_used="railway_nltk",
                        success=True
                    )
                except Exception as e:
                    logger.warning(f"NLTK analysis failed: {e}")
            
            # Fallback to rule-based analysis
            emotions = self._rule_based_emotion_analysis(text)
            primary_emotion = max(emotions, key=emotions.get)
            confidence = emotions[primary_emotion]
            
            # Boost confidence if it's too low
            if confidence < 0.1:
                confidence = 0.6
                emotions[primary_emotion] = confidence
                # Normalize again
                total = sum(emotions.values())
                emotions = {k: v / total for k, v in emotions.items()}
            
            return EmotionResult(
                emotion=primary_emotion,
                confidence=confidence,
                emotions=emotions,
                processing_time=time.time() - start_time,
                model_used="railway_rule_based",
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error in emotion analysis: {e}")
            return EmotionResult(
                emotion="neutral",
                confidence=0.5,
                emotions={"neutral": 1.0},
                processing_time=time.time() - start_time,
                model_used="railway_error_fallback",
                success=False,
                error=str(e)
            )

class MockAudioEmotionModel:
    """Mock audio emotion analysis for Railway deployment"""
    
    def __init__(self):
        self.model_loaded = True
        
    async def load_model(self):
        """Mock model loading"""
        logger.info("🎵 Mock audio emotion model loaded")
        return True
    
    async def analyze_emotion(self, audio_data: bytes) -> EmotionResult:
        """Mock audio emotion analysis"""
        start_time = time.time()
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Mock emotion based on audio file size
        audio_size = len(audio_data)
        emotions = {
            "joy": 0.3,
            "neutral": 0.25,
            "surprise": 0.2,
            "sadness": 0.15,
            "anger": 0.05,
            "fear": 0.03,
            "disgust": 0.02
        }
        
        # Vary based on size
        if audio_size > 1000000:  # Large file
            emotions["anger"] = 0.4
            emotions["joy"] = 0.3
        elif audio_size < 100000:  # Small file
            emotions["neutral"] = 0.5
            emotions["sadness"] = 0.3
        
        primary_emotion = max(emotions, key=emotions.get)
        
        return EmotionResult(
            emotion=primary_emotion,
            confidence=emotions[primary_emotion],
            emotions=emotions,
            processing_time=time.time() - start_time,
            model_used="railway_mock_audio",
            success=True
        )

class MockVideoEmotionModel:
    """Mock video emotion analysis for Railway deployment"""
    
    def __init__(self):
        self.model_loaded = True
        
    async def load_model(self):
        """Mock model loading"""
        logger.info("🎬 Mock video emotion model loaded")
        return True
    
    async def analyze_emotion(self, video_data: bytes) -> EmotionResult:
        """Mock video emotion analysis"""
        start_time = time.time()
        
        # Simulate processing time
        await asyncio.sleep(0.2)
        
        # Mock emotion based on video file size
        video_size = len(video_data)
        emotions = {
            "surprise": 0.35,
            "joy": 0.3,
            "neutral": 0.15,
            "fear": 0.1,
            "sadness": 0.05,
            "anger": 0.03,
            "disgust": 0.02
        }
        
        # Vary based on size
        if video_size > 5000000:  # Large video
            emotions["surprise"] = 0.5
            emotions["joy"] = 0.25
        elif video_size < 500000:  # Small video
            emotions["neutral"] = 0.4
            emotions["sadness"] = 0.3
        
        primary_emotion = max(emotions, key=emotions.get)
        
        return EmotionResult(
            emotion=primary_emotion,
            confidence=emotions[primary_emotion],
            emotions=emotions,
            processing_time=time.time() - start_time,
            model_used="railway_mock_video",
            success=True
        )

# Global model instances
railway_text_model = RailwayTextEmotionModel()
mock_audio_model = MockAudioEmotionModel()
mock_video_model = MockVideoEmotionModel()

async def initialize_models():
    """Initialize all emotion models for Railway"""
    logger.info("🚂 Initializing Railway emotion models...")
    
    try:
        # Load text model
        await railway_text_model.load_model()
        logger.info("✅ Text emotion model ready")
        
        # Load mock models
        await mock_audio_model.load_model()
        await mock_video_model.load_model()
        logger.info("✅ Mock audio/video models ready")
        
        logger.info("🎯 All Railway models initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        return False

# Export main functions
__all__ = [
    'EmotionResult',
    'EmotionAnalysisRequest', 
    'RailwayTextEmotionModel',
    'MockAudioEmotionModel',
    'MockVideoEmotionModel',
    'railway_text_model',
    'mock_audio_model', 
    'mock_video_model',
    'initialize_models'
]
