"""
Video Emotion Analysis Service for EmoSense Backend API

Provides video-based emotion analysis using computer vision and 
facial emotion recognition techniques.
"""

import os
import tempfile
from typing import Dict, List, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import FileProcessingError, ModelProcessingError
from app.models.emotion import EmotionAnalysis, AnalysisType
from app.schemas.emotion import EmotionAnalysisResponse, VideoAnalysisRequest


class VideoEmotionAnalyzer:
    """
    Video emotion analyzer using OpenCV and facial emotion recognition.
    
    Processes video files frame by frame to detect faces and analyze emotions.
    Supports multiple video formats and provides frame-level and aggregated results.
    """
    
    def __init__(self, db: AsyncSession, user_id: int):
        """
        Initialize video emotion analyzer.
        
        Args:
            db: Database session
            user_id: User ID for tracking analyses
        """
        self.db = db
        self.user_id = user_id
        self.face_cascade = None
        self.emotion_model = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize_models(self) -> None:
        """Initialize face detection and emotion recognition models."""
        try:
            # Load face cascade classifier
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # TODO: Load pre-trained emotion recognition model
            # For now, we'll use a placeholder
            self.emotion_model = None
            
        except Exception as e:
            raise ModelProcessingError(
                detail=f"Failed to initialize video analysis models: {str(e)}",
                model_name="video-emotion-analyzer"
            )
    
    async def analyze_video(
        self,
        video_file_path: str,
        frame_interval: float = 1.0,
        confidence_threshold: float = 0.5,
        detect_faces: bool = True,
        max_faces: int = 5
    ) -> EmotionAnalysisResponse:
        """
        Analyze emotions in video content.
        
        Args:
            video_file_path: Path to video file
            frame_interval: Time interval between analyzed frames (seconds)
            confidence_threshold: Minimum confidence for emotion predictions
            detect_faces: Whether to detect and analyze faces
            max_faces: Maximum number of faces to analyze per frame
            
        Returns:
            EmotionAnalysisResponse with video analysis results
            
        Raises:
            FileProcessingError: If video processing fails
            ModelProcessingError: If emotion analysis fails
        """
        try:
            await self.initialize_models()
            
            # Process video in background thread
            results = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._process_video_sync,
                video_file_path,
                frame_interval,
                confidence_threshold,
                detect_faces,
                max_faces
            )
            
            # Save analysis results to database
            analysis = EmotionAnalysis(
                user_id=self.user_id,
                analysis_type=AnalysisType.VIDEO,
                input_data={"video_file": os.path.basename(video_file_path)},
                results=results,
                confidence_score=results.get("average_confidence", 0.0),
                metadata={
                    "frame_interval": frame_interval,
                    "total_frames": results.get("total_frames", 0),
                    "analyzed_frames": results.get("analyzed_frames", 0),
                    "faces_detected": results.get("total_faces", 0)
                }
            )
            
            self.db.add(analysis)
            await self.db.commit()
            await self.db.refresh(analysis)
            
            return EmotionAnalysisResponse(
                id=analysis.id,
                analysis_type=analysis.analysis_type,
                results=analysis.results,
                confidence_score=analysis.confidence_score,
                metadata=analysis.metadata,
                created_at=analysis.created_at
            )
            
        except Exception as e:
            if isinstance(e, (FileProcessingError, ModelProcessingError)):
                raise
            raise FileProcessingError(
                detail=f"Video emotion analysis failed: {str(e)}",
                file_type="video"
            )
    
    def _process_video_sync(
        self,
        video_path: str,
        frame_interval: float,
        confidence_threshold: float,
        detect_faces: bool,
        max_faces: int
    ) -> Dict:
        """
        Synchronous video processing function.
        
        Args:
            video_path: Path to video file
            frame_interval: Frame analysis interval
            confidence_threshold: Minimum confidence threshold
            detect_faces: Whether to detect faces
            max_faces: Maximum faces per frame
            
        Returns:
            Dictionary with analysis results
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise FileProcessingError(
                    detail="Failed to open video file",
                    file_type="video"
                )
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Calculate frame skip interval
            frame_skip = int(fps * frame_interval) if fps > 0 else 1
            
            frame_results = []
            emotions_aggregate = {}
            total_faces = 0
            analyzed_frames = 0
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on interval
                if frame_idx % frame_skip == 0:
                    timestamp = frame_idx / fps if fps > 0 else frame_idx
                    
                    frame_analysis = self._analyze_frame(
                        frame, timestamp, detect_faces, max_faces, confidence_threshold
                    )
                    
                    if frame_analysis:
                        frame_results.append(frame_analysis)
                        analyzed_frames += 1
                        total_faces += len(frame_analysis.get("faces", []))
                        
                        # Aggregate emotions
                        for emotion, score in frame_analysis.get("emotions", {}).items():
                            emotions_aggregate[emotion] = emotions_aggregate.get(emotion, []) + [score]
                
                frame_idx += 1
            
            cap.release()
            
            # Calculate average emotions
            avg_emotions = {}
            for emotion, scores in emotions_aggregate.items():
                avg_emotions[emotion] = np.mean(scores) if scores else 0.0
            
            # Find dominant emotion
            dominant_emotion = max(avg_emotions.items(), key=lambda x: x[1]) if avg_emotions else ("neutral", 0.0)
            
            return {
                "dominant_emotion": dominant_emotion[0],
                "emotions": avg_emotions,
                "average_confidence": dominant_emotion[1],
                "frame_results": frame_results,
                "total_frames": total_frames,
                "analyzed_frames": analyzed_frames,
                "total_faces": total_faces,
                "duration_seconds": duration,
                "fps": fps
            }
            
        except Exception as e:
            raise FileProcessingError(
                detail=f"Video processing failed: {str(e)}",
                file_type="video"
            )
    
    def _analyze_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        detect_faces: bool,
        max_faces: int,
        confidence_threshold: float
    ) -> Optional[Dict]:
        """
        Analyze emotions in a single video frame.
        
        Args:
            frame: Video frame as numpy array
            timestamp: Frame timestamp in seconds
            detect_faces: Whether to detect faces
            max_faces: Maximum faces to analyze
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Frame analysis results or None if no faces detected
        """
        try:
            if not detect_faces:
                # Analyze entire frame (simplified approach)
                emotions = self._predict_emotions_placeholder(frame)
                return {
                    "timestamp": timestamp,
                    "emotions": emotions,
                    "faces": [],
                    "face_count": 0
                }
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return None
            
            # Limit number of faces
            faces = faces[:max_faces]
            
            face_results = []
            frame_emotions = {}
            
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_roi = frame[y:y+h, x:x+w]
                
                # Analyze emotions in face
                face_emotions = self._predict_emotions_placeholder(face_roi)
                
                # Filter by confidence threshold
                filtered_emotions = {
                    emotion: score for emotion, score in face_emotions.items()
                    if score >= confidence_threshold
                }
                
                if filtered_emotions:
                    face_results.append({
                        "bbox": [x, y, w, h],
                        "emotions": filtered_emotions,
                        "dominant_emotion": max(filtered_emotions.items(), key=lambda x: x[1])
                    })
                    
                    # Aggregate emotions for the frame
                    for emotion, score in filtered_emotions.items():
                        if emotion in frame_emotions:
                            frame_emotions[emotion] = max(frame_emotions[emotion], score)
                        else:
                            frame_emotions[emotion] = score
            
            if not face_results:
                return None
            
            return {
                "timestamp": timestamp,
                "emotions": frame_emotions,
                "faces": face_results,
                "face_count": len(face_results)
            }
            
        except Exception as e:
            # Log error but don't fail entire video processing
            print(f"Frame analysis error at {timestamp}s: {str(e)}")
            return None
    
    def _predict_emotions_placeholder(self, image: np.ndarray) -> Dict[str, float]:
        """
        Placeholder emotion prediction function.
        
        In a real implementation, this would use a trained emotion recognition model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary of emotion predictions
        """
        # TODO: Replace with actual emotion recognition model
        # For now, return random emotions as placeholder
        emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        
        # Generate random scores that sum to 1.0
        scores = np.random.random(len(emotions))
        scores = scores / scores.sum()
        
        return {emotion: float(score) for emotion, score in zip(emotions, scores)}
    
    async def analyze_video_batch(
        self,
        video_paths: List[str],
        **kwargs
    ) -> List[EmotionAnalysisResponse]:
        """
        Analyze multiple videos in batch.
        
        Args:
            video_paths: List of video file paths
            **kwargs: Additional analysis parameters
            
        Returns:
            List of analysis results
        """
        results = []
        for video_path in video_paths:
            try:
                result = await self.analyze_video(video_path, **kwargs)
                results.append(result)
            except Exception as e:
                # Log error and continue with next video
                print(f"Failed to analyze video {video_path}: {str(e)}")
                continue
        
        return results
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
