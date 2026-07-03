"""
Emotion Analysis Retrieval Service for EmoSense Backend API

Provides read access to stored emotion analyses: fetching a single analysis
(with its segment/frame details) and listing a user's analyses with pagination
and optional type filtering.
"""

from typing import List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.emotion import AnalysisType, EmotionAnalysis
from app.schemas.emotion import (
    DetailedAnalysisResponse,
    EmotionAnalysisResponse,
)


class EmotionAnalysisService:
    """Service for retrieving persisted emotion analyses."""

    def __init__(self, db: AsyncSession):
        """
        Initialize the analysis service.

        Args:
            db: Database session
        """
        self.db = db

    async def get_analysis_by_id(
        self,
        analysis_id: UUID,
        user_id: UUID,
    ) -> Optional[DetailedAnalysisResponse]:
        """
        Fetch a single analysis owned by the given user, including its
        segment/frame level details.

        Args:
            analysis_id: Unique analysis identifier
            user_id: Owner of the analysis (enforces access control)

        Returns:
            DetailedAnalysisResponse if found, otherwise None
        """
        stmt = (
            select(EmotionAnalysis)
            .where(
                EmotionAnalysis.id == analysis_id,
                EmotionAnalysis.user_id == user_id,
            )
            .options(
                selectinload(EmotionAnalysis.text_segments),
                selectinload(EmotionAnalysis.video_frames),
                selectinload(EmotionAnalysis.audio_segments),
            )
        )
        result = await self.db.execute(stmt)
        analysis = result.scalar_one_or_none()

        if analysis is None:
            return None

        return DetailedAnalysisResponse.model_validate(analysis)

    async def get_user_analyses(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 20,
        analysis_type: Optional[AnalysisType] = None,
    ) -> List[EmotionAnalysisResponse]:
        """
        List a user's analyses, newest first, with pagination and optional
        filtering by analysis type.

        Args:
            user_id: Owner of the analyses
            skip: Number of records to skip
            limit: Maximum number of records to return
            analysis_type: Optional filter by analysis type

        Returns:
            List of EmotionAnalysisResponse
        """
        stmt = select(EmotionAnalysis).where(EmotionAnalysis.user_id == user_id)

        if analysis_type is not None:
            stmt = stmt.where(EmotionAnalysis.analysis_type == analysis_type)

        stmt = (
            stmt.order_by(EmotionAnalysis.created_at.desc())
            .offset(max(skip, 0))
            .limit(max(min(limit, 100), 1))
        )

        result = await self.db.execute(stmt)
        analyses = result.scalars().all()

        return [EmotionAnalysisResponse.model_validate(a) for a in analyses]
