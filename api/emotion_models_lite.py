"""
Lightweight Emotion Analysis Models for Global Deployment
CPU-only version with minimal dependencies
"""

import asyncio
import logging
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time

# Basic imports that should be available
try:
    import torch
    import transformers
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

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

# Configure logging
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

class LightweightTextEmotionModel:
    """Lightweight text emotion analysis using CPU-only models"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.sentiment_analyzer = None
        self.model_name = "j-hartmann/emotion-english-distilroberta-base"
        self.model_loaded = False
        self.fallback_emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
        
    async def load_model(self):
        """Load the emotion analysis model with fallbacks"""
        start_time = time.time()
        
        try:
            if TRANSFORMERS_AVAILABLE:
                logger.info("🤖 Loading transformers emotion model (CPU)...")
                # Force CPU usage
                device = "cpu"
                
                # Try to load the emotion model
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                    self.model.to(device)
                    self.model.eval()
                    self.model_loaded = True
                    logger.info(f"✅ Loaded emotion model in {time.time() - start_time:.2f}s")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load main model: {e}")
                    
            # Fallback to NLTK if available
            if NLTK_AVAILABLE:
                logger.info("📚 Loading NLTK sentiment analyzer...")
                try:
                    # Download required NLTK data if not present
                    import ssl
                    try:
                        _create_unverified_https_context = ssl._create_unverified_context
                    except AttributeError:
                        pass
                    else:
                        ssl._create_default_https_context = _create_unverified_https_context
                    
                    nltk.download('vader_lexicon', quiet=True)
                    self.sentiment_analyzer = SentimentIntensityAnalyzer()
                    logger.info(f"✅ Loaded NLTK analyzer in {time.time() - start_time:.2f}s")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load NLTK: {e}")
            
            # Final fallback to keyword-based analysis
            logger.info("🔤 Using keyword-based emotion analysis")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load any model: {e}")
            return False
    
    def _keyword_emotion_analysis(self, text: str) -> Dict[str, float]:
        """Keyword-based emotion analysis as final fallback"""
        emotion_keywords = {
            "joy": ["happy", "joy", "glad", "pleased", "delighted", "excited", "cheerful", "wonderful", "amazing", "great", "fantastic", "love", "excellent"],
            "sadness": ["sad", "depressed", "unhappy", "sorrowful", "miserable", "heartbroken", "disappointed", "down", "blue", "melancholy"],
            "anger": ["angry", "mad", "furious", "irritated", "annoyed", "rage", "frustrated", "outraged", "livid", "hate", "disgusted"],
            "fear": ["afraid", "scared", "terrified", "anxious", "worried", "nervous", "frightened", "panic", "dread", "concern"],
            "surprise": ["surprised", "shocked", "amazed", "astonished", "unexpected", "sudden", "wow", "incredible", "unbelievable"],
            "disgust": ["disgusting", "gross", "revolting", "sick", "nasty", "awful", "terrible", "horrible", "repulsive"],
            "neutral": ["okay", "fine", "normal", "regular", "typical", "standard", "average", "usual"]
        }
        
        text_lower = text.lower()
        emotion_scores = {emotion: 0.0 for emotion in emotion_keywords.keys()}
        total_matches = 0
        
        for emotion, keywords in emotion_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = matches
            total_matches += matches
        
        # Normalize scores
        if total_matches > 0:
            emotion_scores = {k: v/total_matches for k, v in emotion_scores.items()}
        else:
            # Default to neutral if no keywords found
            emotion_scores["neutral"] = 1.0
        
        return emotion_scores
    
    async def analyze_emotion(self, text: str) -> EmotionResult:
        """Analyze emotion in text"""
        start_time = time.time()
        
        try:
            # Clean and validate input
            if not text or not text.strip():
                return EmotionResult(
                    emotion="neutral",
                    confidence=0.0,
                    emotions={"neutral": 1.0},
                    processing_time=time.time() - start_time,
                    model_used="none",
                    success=False,
                    error="Empty text provided"
                )
            
            text = text.strip()[:512]  # Limit text length
            
            # Try transformers model first
            if self.model_loaded and self.model is not None and self.tokenizer is not None:
                try:
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        predicted_class_id = predictions.argmax().item()
                        confidence = predictions.max().item()
                    
                    # Map to emotion labels
                    emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
                    emotion = emotion_labels[predicted_class_id] if predicted_class_id < len(emotion_labels) else "neutral"
                    
                    # Create emotion distribution
                    emotions = {}
                    for i, label in enumerate(emotion_labels):
                        emotions[label] = predictions[0][i].item() if i < len(predictions[0]) else 0.0
                    
                    return EmotionResult(
                        emotion=emotion,
                        confidence=float(confidence),
                        emotions=emotions,
                        processing_time=time.time() - start_time,
                        model_used="transformers-cpu",
                        success=True
                    )
                except Exception as e:
                    logger.warning(f"Transformers model failed: {e}")
            
            # Try NLTK sentiment analyzer
            if self.sentiment_analyzer is not None:
                try:
                    scores = self.sentiment_analyzer.polarity_scores(text)
                    
                    # Map sentiment to emotions
                    compound = scores['compound']
                    if compound >= 0.05:
                        primary_emotion = "joy"
                        confidence = compound
                    elif compound <= -0.05:
                        primary_emotion = "sadness"
                        confidence = abs(compound)
                    else:
                        primary_emotion = "neutral"
                        confidence = 1.0 - abs(compound)
                    
                    emotions = {
                        "joy": max(0, scores['pos']),
                        "sadness": max(0, scores['neg']),
                        "neutral": max(0, scores['neu']),
                        "anger": max(0, scores['neg'] * 0.5),
                        "fear": max(0, scores['neg'] * 0.3),
                        "surprise": 0.1,
                        "disgust": max(0, scores['neg'] * 0.2)
                    }
                    
                    return EmotionResult(
                        emotion=primary_emotion,
                        confidence=float(confidence),
                        emotions=emotions,
                        processing_time=time.time() - start_time,
                        model_used="nltk-vader",
                        success=True
                    )
                except Exception as e:
                    logger.warning(f"NLTK analyzer failed: {e}")
            
            # Final fallback to keyword analysis
            emotions = self._keyword_emotion_analysis(text)
            primary_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            confidence = emotions[primary_emotion]
            
            return EmotionResult(
                emotion=primary_emotion,
                confidence=float(confidence),
                emotions=emotions,
                processing_time=time.time() - start_time,
                model_used="keyword-based",
                success=True
            )
            
        except Exception as e:
            logger.error(f"❌ Emotion analysis failed: {e}")
            return EmotionResult(
                emotion="neutral",
                confidence=0.0,
                emotions={"neutral": 1.0, "error": 1.0},
                processing_time=time.time() - start_time,
                model_used="error-fallback",
                success=False,
                error=str(e)
            )

class MockAudioEmotionModel:
    """Mock audio emotion model for lightweight deployment"""
    
    def __init__(self):
        self.model_loaded = False
    
    async def load_model(self):
        """Mock load model"""
        logger.info("📱 Audio analysis not available in lightweight mode")
        return True
    
    async def analyze_emotion(self, audio_data: bytes) -> EmotionResult:
        """Mock audio analysis"""
        return EmotionResult(
            emotion="neutral",
            confidence=0.5,
            emotions={"neutral": 1.0},
            processing_time=0.1,
            model_used="mock-audio",
            success=False,
            error="Audio analysis not available in lightweight mode"
        )

class MockVideoEmotionModel:
    """Mock video emotion model for lightweight deployment"""
    
    def __init__(self):
        self.model_loaded = False
    
    async def load_model(self):
        """Mock load model"""
        logger.info("🎥 Video analysis not available in lightweight mode")
        return True
    
    async def analyze_emotion(self, video_data: bytes) -> EmotionResult:
        """Mock video analysis"""
        return EmotionResult(
            emotion="neutral",
            confidence=0.5,
            emotions={"neutral": 1.0},
            processing_time=0.1,
            model_used="mock-video",
            success=False,
            error="Video analysis not available in lightweight mode"
        )

# Export aliases for compatibility
TextEmotionModel = LightweightTextEmotionModel
AudioEmotionModel = MockAudioEmotionModel
VideoEmotionModel = MockVideoEmotionModel
