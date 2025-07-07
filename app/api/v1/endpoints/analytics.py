"""
Analytics API Endpoints for EmoSense Backend API

Provides endpoints for analytics data, reports, and statistics
about emotion analysis usage and results.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db_session
from app.dependencies import get_current_user, get_current_superuser
from app.services.analytics import AnalyticsService


# Create router for analytics endpoints
router = APIRouter()


@router.get(
    "/dashboard",
    summary="Get Dashboard Data",
    description="Get dashboard analytics data for the current user"
)
async def get_dashboard(
    db: AsyncSession = Depends(get_db_session),
    current_user = Depends(get_current_user)
):
    """
    Get dashboard analytics data for the authenticated user.
    
    Returns comprehensive dashboard data including:
    - User statistics
    - Recent analyses
    - Emotion trends
    - Analysis type distribution
    """
    try:
        analytics_service = AnalyticsService(db=db)
        dashboard_data = await analytics_service.get_dashboard_data(current_user.id)
        
        return {
            "status": "success",
            "data": dashboard_data
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dashboard data: {str(e)}"
        )


@router.get(
    "/reports",
    summary="Generate Analytics Report",
    description="Generate a detailed analytics report for the current user"
)
async def get_user_report(
    start_date: Optional[datetime] = Query(None, description="Report start date"),
    end_date: Optional[datetime] = Query(None, description="Report end date"),
    db: AsyncSession = Depends(get_db_session),
    current_user = Depends(get_current_user)
):
    """
    Generate a detailed analytics report for the authenticated user.
    
    Args:
        start_date: Optional start date for the report period
        end_date: Optional end date for the report period
        
    Returns:
        Detailed user analytics report with patterns and insights
    """
    try:
        analytics_service = AnalyticsService(db=db)
        report = await analytics_service.generate_user_report(
            user_id=current_user.id,
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "status": "success",
            "data": report
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate report: {str(e)}"
        )


@router.get(
    "/stats",
    summary="Get Usage Statistics", 
    description="Get system-wide usage statistics (admin only)"
)
async def get_usage_statistics(
    admin_only: bool = Query(True, description="Include admin-only statistics"),
    db: AsyncSession = Depends(get_db_session),
    current_user = Depends(get_current_superuser)
):
    """
    Get system-wide usage statistics.
    
    Requires admin/superuser privileges.
    
    Args:
        admin_only: Whether to include detailed admin statistics
        
    Returns:
        System-wide usage statistics and metrics
    """
    try:
        analytics_service = AnalyticsService(db=db)
        stats = await analytics_service.get_usage_statistics(admin_only=admin_only)
        
        return {
            "status": "success",
            "data": stats
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve usage statistics: {str(e)}"
        )


@router.get(
    "/trends",
    summary="Get Emotion Trends",
    description="Get emotion trends over time for the current user"
)
async def get_emotion_trends(
    days: int = Query(30, description="Number of days to analyze", ge=1, le=365),
    db: AsyncSession = Depends(get_db_session),
    current_user = Depends(get_current_user)
):
    """
    Get emotion trends over time for the authenticated user.
    
    Args:
        days: Number of days to include in the trend analysis
        
    Returns:
        Emotion trends and patterns over the specified time period
    """
    try:
        analytics_service = AnalyticsService(db=db)
        
        # Get analyses for the specified period
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Use the internal method to get trends
        trends = await analytics_service._get_emotion_trends(current_user.id, days)
        
        return {
            "status": "success",
            "data": trends
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve emotion trends: {str(e)}"
        )
