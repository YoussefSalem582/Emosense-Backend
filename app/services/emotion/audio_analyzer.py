"""
Audio Emotion Analysis Service for EmoSense Backend API

Provides audio-based emotion analysis using speech processing and
machine learning techniques for emotional speech recognition.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Dict, List, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import FileProcessingError, ModelProcessingError
from app.models.emotion import (
    AnalysisStatus,
    AnalysisType,
    EmotionAnalysis,
    EmotionLabel,
)
from app.schemas.emotion import EmotionAnalysisResponse, AudioAnalysisRequest
from app.services.emotion.labels import get_dominant_emotion, standardize_emotions


class AudioEmotionAnalyzer:
    """
    Audio emotion analyzer using speech signal processing and emotion recognition.
    
    Processes audio files to extract emotional features and classify emotions
    in speech segments. Supports multiple audio formats and transcription.
    """
    
    def __init__(self, db: AsyncSession, user_id: int):
        """
        Initialize audio emotion analyzer.
        
        Args:
            db: Database session
            user_id: User ID for tracking analyses
        """
        self.db = db
        self.user_id = user_id
        self.emotion_model = None
        self.transcription_model = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    async def initialize_models(self) -> None:
        """Initialize audio processing and emotion recognition models."""
        try:
            # TODO: Load pre-trained audio emotion recognition model
            # For now, we'll use placeholder models
            self.emotion_model = None
            self.transcription_model = None
            
        except Exception as e:
            raise ModelProcessingError(
                detail=f"Failed to initialize audio analysis models",
                model_name="audio-emotion-analyzer"
            )
    
    async def analyze_audio(
        self,
        audio_file,
        request: AudioAnalysisRequest,
    ) -> EmotionAnalysisResponse:
        """
        Analyze emotions in an uploaded audio file.

        Args:
            audio_file: Uploaded audio (Starlette ``UploadFile``)
            request: Audio analysis options

        Returns:
            EmotionAnalysisResponse with aggregated audio analysis results

        Raises:
            FileProcessingError: If audio processing fails
            ModelProcessingError: If emotion analysis fails
        """
        start_time = time.time()

        # Persist the upload to a temporary file for librosa to read.
        contents = await audio_file.read()
        suffix = os.path.splitext(audio_file.filename or "")[1] or ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            tmp.write(contents)
            tmp.flush()
            tmp.close()

            await self.initialize_models()

            # Process audio in a background thread (librosa is blocking).
            results = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._process_audio_sync,
                tmp.name,
                request.segment_duration,
                request.confidence_threshold,
                request.transcribe_speech,
                request.language,
            )
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        try:
            emotion_scores = standardize_emotions(results.get("emotions", {}))
            dominant_emotion, confidence = get_dominant_emotion(emotion_scores)

            analysis = EmotionAnalysis(
                user_id=self.user_id,
                analysis_type=AnalysisType.AUDIO,
                status=AnalysisStatus.COMPLETED,
                input_file_name=audio_file.filename,
                input_file_size=len(contents),
                confidence_threshold=request.confidence_threshold,
                model_version="audio-emotion-analyzer-placeholder",
                dominant_emotion=dominant_emotion,
                dominant_emotion_confidence=confidence,
                emotion_scores=emotion_scores,
                processing_duration=time.time() - start_time,
            )

            self.db.add(analysis)
            await self.db.commit()
            await self.db.refresh(analysis)

            return EmotionAnalysisResponse.model_validate(analysis)

        except Exception as e:
            if isinstance(e, (FileProcessingError, ModelProcessingError)):
                raise
            raise FileProcessingError(
                detail=f"Audio emotion analysis failed",
                file_name=audio_file.filename,
            )
    
    def _process_audio_sync(
        self,
        audio_path: str,
        segment_duration: float,
        confidence_threshold: float,
        transcribe_speech: bool,
        language: str
    ) -> Dict:
        """
        Synchronous audio processing function.
        
        Args:
            audio_path: Path to audio file
            segment_duration: Segment duration for analysis
            confidence_threshold: Minimum confidence threshold
            transcribe_speech: Whether to transcribe speech
            language: Language for transcription
            
        Returns:
            Dictionary with analysis results
        """
        try:
            import librosa
            import numpy as np

            # Load audio file
            audio_data, sample_rate = librosa.load(audio_path, sr=None)
            total_duration = len(audio_data) / sample_rate
            
            # Calculate segment parameters
            segment_samples = int(segment_duration * sample_rate)
            n_segments = int(np.ceil(len(audio_data) / segment_samples))
            
            segment_results = []
            emotions_aggregate = {}
            transcriptions = []
            analyzed_segments = 0
            
            for i in range(n_segments):
                start_sample = i * segment_samples
                end_sample = min((i + 1) * segment_samples, len(audio_data))
                
                segment = audio_data[start_sample:end_sample]
                start_time = start_sample / sample_rate
                end_time = end_sample / sample_rate
                
                # Skip very short segments
                if len(segment) < sample_rate * 0.5:  # Less than 0.5 seconds
                    continue
                
                segment_analysis = self._analyze_audio_segment(
                    segment, sample_rate, start_time, end_time, 
                    confidence_threshold, transcribe_speech, language
                )
                
                if segment_analysis:
                    segment_results.append(segment_analysis)
                    analyzed_segments += 1
                    
                    # Aggregate emotions
                    for emotion, score in segment_analysis.get("emotions", {}).items():
                        emotions_aggregate[emotion] = emotions_aggregate.get(emotion, []) + [score]
                    
                    # Collect transcriptions
                    if transcribe_speech and segment_analysis.get("transcription"):
                        transcriptions.append({
                            "start_time": start_time,
                            "end_time": end_time,
                            "text": segment_analysis["transcription"]
                        })
            
            # Calculate average emotions
            avg_emotions = {}
            for emotion, scores in emotions_aggregate.items():
                avg_emotions[emotion] = np.mean(scores) if scores else 0.0
            
            # Find dominant emotion
            dominant_emotion = max(avg_emotions.items(), key=lambda x: x[1]) if avg_emotions else ("neutral", 0.0)
            
            # Extract audio features for metadata
            audio_features = self._extract_audio_features(audio_data, sample_rate)
            
            return {
                "dominant_emotion": dominant_emotion[0],
                "emotions": avg_emotions,
                "average_confidence": dominant_emotion[1],
                "segment_results": segment_results,
                "total_duration": total_duration,
                "analyzed_segments": analyzed_segments,
                "sample_rate": sample_rate,
                "transcriptions": transcriptions if transcribe_speech else [],
                "audio_features": audio_features
            }
            
        except FileProcessingError:
            raise
        except Exception as e:
            raise FileProcessingError(
                detail=f"Audio processing failed",
                file_name=os.path.basename(audio_path),
            )

    def _analyze_audio_segment(
        self,
        segment: np.ndarray,
        sample_rate: int,
        start_time: float,
        end_time: float,
        confidence_threshold: float,
        transcribe_speech: bool,
        language: str
    ) -> Optional[Dict]:
        """
        Analyze emotions in a single audio segment.
        
        Args:
            segment: Audio segment as numpy array
            sample_rate: Audio sample rate
            start_time: Segment start time
            end_time: Segment end time
            confidence_threshold: Minimum confidence threshold
            transcribe_speech: Whether to transcribe speech
            language: Language for transcription
            
        Returns:
            Segment analysis results or None if processing fails
        """
        try:
            # Extract audio features
            features = self._extract_segment_features(segment, sample_rate)
            
            # Predict emotions
            emotions = self._predict_emotions_from_audio(features)
            
            # Filter by confidence threshold
            filtered_emotions = {
                emotion: score for emotion, score in emotions.items()
                if score >= confidence_threshold
            }
            
            if not filtered_emotions:
                return None
            
            result = {
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "emotions": filtered_emotions,
                "dominant_emotion": max(filtered_emotions.items(), key=lambda x: x[1]),
                "features": features
            }
            
            # Add transcription if requested
            if transcribe_speech:
                transcription = self._transcribe_segment(segment, sample_rate, language)
                if transcription:
                    result["transcription"] = transcription
            
            return result
            
        except Exception as e:
            # Log error but don't fail entire audio processing
            print(f"Segment analysis error at {start_time}-{end_time}s")
            return None
    
    def _extract_segment_features(self, segment: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """
        Extract audio features from a segment for emotion analysis.
        
        Args:
            segment: Audio segment
            sample_rate: Sample rate
            
        Returns:
            Dictionary of extracted features
        """
        try:
            import librosa
            import numpy as np

            features = {}

            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=segment, sr=sample_rate)[0]
            features["spectral_centroid_mean"] = np.mean(spectral_centroids)
            features["spectral_centroid_std"] = np.std(spectral_centroids)
            
            # MFCCs (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=13)
            for i in range(mfccs.shape[0]):
                features[f"mfcc_{i}_mean"] = np.mean(mfccs[i])
                features[f"mfcc_{i}_std"] = np.std(mfccs[i])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=segment, sr=sample_rate)
            features["chroma_mean"] = np.mean(chroma)
            features["chroma_std"] = np.std(chroma)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(segment)[0]
            features["zcr_mean"] = np.mean(zcr)
            features["zcr_std"] = np.std(zcr)
            
            # RMS energy
            rms = librosa.feature.rms(y=segment)[0]
            features["rms_mean"] = np.mean(rms)
            features["rms_std"] = np.std(rms)
            
            return features
            
        except Exception as e:
            print(f"Feature extraction error")
            return {}
    
    def _extract_audio_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """
        Extract global audio features for metadata.
        
        Args:
            audio_data: Full audio data
            sample_rate: Sample rate
            
        Returns:
            Dictionary of audio features
        """
        try:
            import numpy as np

            features = {}

            # Basic audio properties
            features["duration"] = len(audio_data) / sample_rate
            features["sample_rate"] = sample_rate
            features["n_samples"] = len(audio_data)
            
            # Signal statistics
            features["rms_energy"] = np.sqrt(np.mean(audio_data**2))
            features["max_amplitude"] = np.max(np.abs(audio_data))
            features["dynamic_range"] = np.max(audio_data) - np.min(audio_data)
            
            # Spectral statistics
            fft = np.fft.fft(audio_data)
            magnitude = np.abs(fft)
            features["spectral_peak"] = np.max(magnitude)
            features["spectral_mean"] = np.mean(magnitude)
            
            return features
            
        except Exception as e:
            print(f"Global feature extraction error")
            return {}
    
    def _predict_emotions_from_audio(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Predict emotions from extracted audio features.
        
        In a real implementation, this would use a trained audio emotion model.
        
        Args:
            features: Extracted audio features
            
        Returns:
            Dictionary of emotion predictions
        """
        # TODO: Replace with actual audio emotion recognition model (see Phase 4).
        # For now, return random emotions as a placeholder.
        import numpy as np

        emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

        # Generate random scores that sum to 1.0
        scores = np.random.random(len(emotions))
        scores = scores / scores.sum()

        return {emotion: float(score) for emotion, score in zip(emotions, scores)}
    
    def _transcribe_segment(self, segment: np.ndarray, sample_rate: int, language: str) -> Optional[str]:
        """
        Transcribe speech in audio segment.
        
        Args:
            segment: Audio segment
            sample_rate: Sample rate
            language: Target language
            
        Returns:
            Transcribed text or None if transcription fails
        """
        # TODO: Implement speech-to-text transcription
        # This could use Whisper, Google Speech-to-Text, or similar services
        
        # Placeholder implementation
        if len(segment) > sample_rate * 1.0:  # At least 1 second of audio
            return f"[Transcription placeholder for {len(segment)/sample_rate:.1f}s segment]"
        
        return None
    
    async def analyze_audio_batch(
        self,
        audio_paths: List[str],
        **kwargs
    ) -> List[EmotionAnalysisResponse]:
        """
        Analyze multiple audio files in batch.
        
        Args:
            audio_paths: List of audio file paths
            **kwargs: Additional analysis parameters
            
        Returns:
            List of analysis results
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = await self.analyze_audio(audio_path, **kwargs)
                results.append(result)
            except Exception as e:
                # Log error and continue with next audio file
                print(f"Failed to analyze audio {audio_path}")
                continue
        
        return results
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
