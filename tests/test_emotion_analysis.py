"""
Test cases for emotion analysis endpoints.

Tests text, video, and audio emotion analysis functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import status
from httpx import AsyncClient


class TestTextEmotionAnalysis:
    """Test text emotion analysis endpoints."""
    
    @pytest.mark.asyncio
    async def test_analyze_text_success(self, async_client: AsyncClient, auth_headers, sample_text_data):
        """Test successful text emotion analysis."""
        with patch('app.services.emotion.text_analyzer.TextEmotionAnalyzer.analyze_text') as mock_analyze:
            mock_analyze.return_value = {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "analysis_type": "text",
                "results": {
                    "dominant_emotion": "happy",
                    "emotions": {"happy": 0.85, "neutral": 0.10, "sad": 0.05},
                    "segments": []
                },
                "confidence_score": 0.85,
                "metadata": {"processing_time": 1.23},
                "created_at": "2024-01-01T00:00:00Z"
            }
            
            response = await async_client.post(
                "/api/v1/emotion/text",
                json=sample_text_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            assert data["results"]["dominant_emotion"] == "happy"
            assert "emotions" in data["results"]
            assert data["confidence_score"] == 0.85
    
    @pytest.mark.asyncio
    async def test_analyze_text_unauthorized(self, async_client: AsyncClient, sample_text_data):
        """Test text analysis without authentication."""
        response = await async_client.post("/api/v1/emotion/text", json=sample_text_data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    @pytest.mark.asyncio
    async def test_analyze_text_invalid_data(self, async_client: AsyncClient, auth_headers):
        """Test text analysis with invalid data."""
        invalid_data = {
            "text": "",  # Empty text
            "confidence_threshold": 1.5  # Invalid threshold
        }
        
        response = await async_client.post(
            "/api/v1/emotion/text",
            json=invalid_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.asyncio
    async def test_analyze_text_processing_error(self, async_client: AsyncClient, auth_headers, sample_text_data):
        """Test text analysis with processing error."""
        with patch('app.services.emotion.text_analyzer.TextEmotionAnalyzer.analyze_text') as mock_analyze:
            mock_analyze.side_effect = Exception("Model processing failed")
            
            response = await async_client.post(
                "/api/v1/emotion/text",
                json=sample_text_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "failed" in response.json()["detail"].lower()


class TestVideoEmotionAnalysis:
    """Test video emotion analysis endpoints."""
    
    @pytest.mark.asyncio
    async def test_analyze_video_success(self, async_client: AsyncClient, auth_headers):
        """Test successful video emotion analysis."""
        # Mock file upload
        files = {"video_file": ("test_video.mp4", b"fake video content", "video/mp4")}
        data = {
            "frame_interval": 1.0,
            "confidence_threshold": 0.5,
            "detect_faces": True,
            "max_faces": 5
        }
        
        with patch('app.services.emotion.video_analyzer.VideoEmotionAnalyzer.analyze_video') as mock_analyze:
            mock_analyze.return_value = {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "analysis_type": "video",
                "results": {
                    "dominant_emotion": "happy",
                    "emotions": {"happy": 0.75, "neutral": 0.20, "sad": 0.05},
                    "frame_results": [],
                    "total_frames": 30,
                    "analyzed_frames": 10
                },
                "confidence_score": 0.75,
                "metadata": {"duration": 10.0, "fps": 30},
                "created_at": "2024-01-01T00:00:00Z"
            }
            
            response = await async_client.post(
                "/api/v1/emotion/video",
                files=files,
                data=data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_201_CREATED
            response_data = response.json()
            assert response_data["results"]["dominant_emotion"] == "happy"
            assert response_data["results"]["total_frames"] == 30
    
    @pytest.mark.asyncio
    async def test_analyze_video_unauthorized(self, async_client: AsyncClient):
        """Test video analysis without authentication."""
        files = {"video_file": ("test_video.mp4", b"fake video content", "video/mp4")}
        
        response = await async_client.post("/api/v1/emotion/video", files=files)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    @pytest.mark.asyncio
    async def test_analyze_video_invalid_file_type(self, async_client: AsyncClient, auth_headers):
        """Test video analysis with invalid file type."""
        files = {"video_file": ("test.txt", b"text content", "text/plain")}
        
        response = await async_client.post(
            "/api/v1/emotion/video",
            files=files,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestAudioEmotionAnalysis:
    """Test audio emotion analysis endpoints."""
    
    @pytest.mark.asyncio
    async def test_analyze_audio_success(self, async_client: AsyncClient, auth_headers):
        """Test successful audio emotion analysis."""
        files = {"audio_file": ("test_audio.wav", b"fake audio content", "audio/wav")}
        data = {
            "segment_duration": 3.0,
            "confidence_threshold": 0.5,
            "transcribe_speech": False,
            "language": "en"
        }
        
        with patch('app.services.emotion.audio_analyzer.AudioEmotionAnalyzer.analyze_audio') as mock_analyze:
            mock_analyze.return_value = {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "analysis_type": "audio",
                "results": {
                    "dominant_emotion": "neutral",
                    "emotions": {"neutral": 0.60, "happy": 0.25, "sad": 0.15},
                    "segment_results": [],
                    "total_duration": 15.0,
                    "analyzed_segments": 5
                },
                "confidence_score": 0.60,
                "metadata": {"sample_rate": 44100},
                "created_at": "2024-01-01T00:00:00Z"
            }
            
            response = await async_client.post(
                "/api/v1/emotion/audio",
                files=files,
                data=data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_201_CREATED
            response_data = response.json()
            assert response_data["results"]["dominant_emotion"] == "neutral"
            assert response_data["results"]["total_duration"] == 15.0


class TestBatchEmotionAnalysis:
    """Test batch emotion analysis endpoints."""
    
    @pytest.mark.asyncio
    async def test_batch_analyze_text(self, async_client: AsyncClient, auth_headers):
        """Test batch text emotion analysis."""
        batch_data = {
            "analysis_type": "text",
            "inputs": [
                "I am very happy today!",
                "This is sad news.",
                "I feel neutral about this."
            ],
            "confidence_threshold": 0.5
        }
        
        with patch('app.services.emotion.text_analyzer.TextEmotionAnalyzer.batch_analyze_text') as mock_batch:
            mock_batch.return_value = {
                "batch_id": "batch-123",
                "total_inputs": 3,
                "processed_inputs": 3,
                "results": [
                    {"input_index": 0, "dominant_emotion": "happy", "confidence": 0.85},
                    {"input_index": 1, "dominant_emotion": "sad", "confidence": 0.75},
                    {"input_index": 2, "dominant_emotion": "neutral", "confidence": 0.60}
                ],
                "summary": {
                    "emotions_distribution": {"happy": 1, "sad": 1, "neutral": 1},
                    "average_confidence": 0.73
                }
            }
            
            response = await async_client.post(
                "/api/v1/emotion/batch",
                json=batch_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_202_ACCEPTED
            response_data = response.json()
            assert response_data["total_inputs"] == 3
            assert response_data["processed_inputs"] == 3
            assert len(response_data["results"]) == 3
    
    @pytest.mark.asyncio
    async def test_batch_analyze_too_many_inputs(self, async_client: AsyncClient, auth_headers):
        """Test batch analysis with too many inputs."""
        batch_data = {
            "analysis_type": "text",
            "inputs": ["text"] * 101,  # Exceeds maximum of 100
            "confidence_threshold": 0.5
        }
        
        response = await async_client.post(
            "/api/v1/emotion/batch",
            json=batch_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "maximum" in response.json()["detail"].lower()


class TestEmotionAnalysisRetrieval:
    """Test emotion analysis retrieval endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_analysis_results(self, async_client: AsyncClient, auth_headers):
        """Test retrieving specific analysis results."""
        analysis_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('app.services.emotion.EmotionAnalysisService.get_analysis_by_id') as mock_get:
            mock_get.return_value = {
                "id": analysis_id,
                "analysis_type": "text",
                "results": {"dominant_emotion": "happy", "emotions": {"happy": 0.85}},
                "confidence_score": 0.85,
                "metadata": {},
                "created_at": "2024-01-01T00:00:00Z"
            }
            
            response = await async_client.get(
                f"/api/v1/emotion/{analysis_id}",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            response_data = response.json()
            assert response_data["id"] == analysis_id
            assert response_data["results"]["dominant_emotion"] == "happy"
    
    @pytest.mark.asyncio
    async def test_get_analysis_not_found(self, async_client: AsyncClient, auth_headers):
        """Test retrieving non-existent analysis."""
        analysis_id = "nonexistent-id"
        
        response = await async_client.get(
            f"/api/v1/emotion/{analysis_id}",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    @pytest.mark.asyncio
    async def test_list_user_analyses(self, async_client: AsyncClient, auth_headers):
        """Test listing user's analyses."""
        with patch('app.services.emotion.EmotionAnalysisService.get_user_analyses') as mock_list:
            mock_list.return_value = [
                {
                    "id": "analysis-1",
                    "analysis_type": "text",
                    "results": {"dominant_emotion": "happy"},
                    "confidence_score": 0.85,
                    "created_at": "2024-01-01T00:00:00Z"
                },
                {
                    "id": "analysis-2",
                    "analysis_type": "video",
                    "results": {"dominant_emotion": "neutral"},
                    "confidence_score": 0.70,
                    "created_at": "2024-01-01T01:00:00Z"
                }
            ]
            
            response = await async_client.get(
                "/api/v1/emotion/",
                headers=auth_headers,
                params={"skip": 0, "limit": 20}
            )
            
            assert response.status_code == status.HTTP_200_OK
            response_data = response.json()
            assert len(response_data) == 2
            assert response_data[0]["id"] == "analysis-1"
            assert response_data[1]["id"] == "analysis-2"
