"""
Analytics Service for EmoSense Backend API

Provides analytics and reporting functionality for emotion analysis data.
Generates insights, trends, and statistics from user emotion analysis history.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, and_, desc
from sqlalchemy.future import select

from app.models.emotion import EmotionAnalysis, AnalysisType
from app.models.user import User


class AnalyticsService:
    """
    Analytics service for emotion analysis data.
    
    Provides methods to generate reports, statistics, and insights
    from emotion analysis history and user data.
    """
    
    def __init__(self, db: AsyncSession):
        """
        Initialize analytics service.
        
        Args:
            db: Database session
        """
        self.db = db
    
    async def get_dashboard_data(self, user_id: int) -> Dict[str, Any]:
        """
        Get dashboard data for a specific user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with dashboard statistics
        """
        try:
            # Get basic statistics
            stats = await self._get_user_statistics(user_id)
            
            # Get recent analyses
            recent_analyses = await self._get_recent_analyses(user_id, limit=10)
            
            # Get emotion trends
            emotion_trends = await self._get_emotion_trends(user_id, days=30)
            
            # Get analysis type distribution
            type_distribution = await self._get_analysis_type_distribution(user_id)
            
            return {
                "statistics": stats,
                "recent_analyses": recent_analyses,
                "emotion_trends": emotion_trends,
                "type_distribution": type_distribution,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise Exception(f"Failed to generate dashboard data: {str(e)}")
    
    async def _get_user_statistics(self, user_id: int) -> Dict[str, Any]:
        """Get basic user statistics."""
        try:
            # Total analyses count
            total_query = select(func.count(EmotionAnalysis.id)).where(
                EmotionAnalysis.user_id == user_id
            )
            total_result = await self.db.execute(total_query)
            total_analyses = total_result.scalar() or 0
            
            # Analyses by type
            type_counts = {}
            for analysis_type in AnalysisType:
                type_query = select(func.count(EmotionAnalysis.id)).where(
                    and_(
                        EmotionAnalysis.user_id == user_id,
                        EmotionAnalysis.analysis_type == analysis_type
                    )
                )
                type_result = await self.db.execute(type_query)
                type_counts[analysis_type.value] = type_result.scalar() or 0
            
            # Analyses this month
            this_month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            month_query = select(func.count(EmotionAnalysis.id)).where(
                and_(
                    EmotionAnalysis.user_id == user_id,
                    EmotionAnalysis.created_at >= this_month_start
                )
            )
            month_result = await self.db.execute(month_query)
            this_month_analyses = month_result.scalar() or 0
            
            # Average confidence score
            confidence_query = select(func.avg(EmotionAnalysis.confidence_score)).where(
                EmotionAnalysis.user_id == user_id
            )
            confidence_result = await self.db.execute(confidence_query)
            avg_confidence = confidence_result.scalar() or 0.0
            
            return {
                "total_analyses": total_analyses,
                "analyses_by_type": type_counts,
                "this_month_analyses": this_month_analyses,
                "average_confidence": float(avg_confidence),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise Exception(f"Failed to get user statistics: {str(e)}")
    
    async def _get_recent_analyses(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent analyses for a user."""
        try:
            query = select(EmotionAnalysis).where(
                EmotionAnalysis.user_id == user_id
            ).order_by(desc(EmotionAnalysis.created_at)).limit(limit)
            
            result = await self.db.execute(query)
            analyses = result.scalars().all()
            
            return [
                {
                    "id": str(analysis.id),
                    "analysis_type": analysis.analysis_type.value,
                    "dominant_emotion": analysis.results.get("dominant_emotion", "unknown"),
                    "confidence_score": analysis.confidence_score,
                    "created_at": analysis.created_at.isoformat()
                }
                for analysis in analyses
            ]
            
        except Exception as e:
            raise Exception(f"Failed to get recent analyses: {str(e)}")
    
    async def _get_emotion_trends(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get emotion trends over time."""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            query = select(EmotionAnalysis).where(
                and_(
                    EmotionAnalysis.user_id == user_id,
                    EmotionAnalysis.created_at >= start_date
                )
            ).order_by(EmotionAnalysis.created_at)
            
            result = await self.db.execute(query)
            analyses = result.scalars().all()
            
            # Group by day and emotion
            daily_emotions = {}
            emotion_counts = {}
            
            for analysis in analyses:
                date_key = analysis.created_at.date().isoformat()
                dominant_emotion = analysis.results.get("dominant_emotion", "unknown")
                
                if date_key not in daily_emotions:
                    daily_emotions[date_key] = {}
                
                if dominant_emotion not in daily_emotions[date_key]:
                    daily_emotions[date_key][dominant_emotion] = 0
                
                daily_emotions[date_key][dominant_emotion] += 1
                
                # Overall emotion counts
                if dominant_emotion not in emotion_counts:
                    emotion_counts[dominant_emotion] = 0
                emotion_counts[dominant_emotion] += 1
            
            return {
                "daily_trends": daily_emotions,
                "overall_distribution": emotion_counts,
                "period_days": days,
                "start_date": start_date.isoformat(),
                "end_date": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise Exception(f"Failed to get emotion trends: {str(e)}")
    
    async def _get_analysis_type_distribution(self, user_id: int) -> Dict[str, int]:
        """Get distribution of analysis types."""
        try:
            distribution = {}
            
            for analysis_type in AnalysisType:
                query = select(func.count(EmotionAnalysis.id)).where(
                    and_(
                        EmotionAnalysis.user_id == user_id,
                        EmotionAnalysis.analysis_type == analysis_type
                    )
                )
                result = await self.db.execute(query)
                count = result.scalar() or 0
                distribution[analysis_type.value] = count
            
            return distribution
            
        except Exception as e:
            raise Exception(f"Failed to get analysis type distribution: {str(e)}")
    
    async def get_usage_statistics(self, admin_only: bool = True) -> Dict[str, Any]:
        """
        Get system-wide usage statistics.
        
        Args:
            admin_only: Whether this is for admin access only
            
        Returns:
            Dictionary with usage statistics
        """
        try:
            # Total users
            users_query = select(func.count(User.id))
            users_result = await self.db.execute(users_query)
            total_users = users_result.scalar() or 0
            
            # Total analyses
            analyses_query = select(func.count(EmotionAnalysis.id))
            analyses_result = await self.db.execute(analyses_query)
            total_analyses = analyses_result.scalar() or 0
            
            # Analyses by type
            type_stats = {}
            for analysis_type in AnalysisType:
                type_query = select(func.count(EmotionAnalysis.id)).where(
                    EmotionAnalysis.analysis_type == analysis_type
                )
                type_result = await self.db.execute(type_query)
                type_stats[analysis_type.value] = type_result.scalar() or 0
            
            # Recent activity (last 24 hours)
            yesterday = datetime.utcnow() - timedelta(hours=24)
            recent_query = select(func.count(EmotionAnalysis.id)).where(
                EmotionAnalysis.created_at >= yesterday
            )
            recent_result = await self.db.execute(recent_query)
            recent_analyses = recent_result.scalar() or 0
            
            # Average analyses per user
            avg_per_user = total_analyses / total_users if total_users > 0 else 0
            
            stats = {
                "total_users": total_users,
                "total_analyses": total_analyses,
                "analyses_by_type": type_stats,
                "recent_analyses_24h": recent_analyses,
                "average_analyses_per_user": avg_per_user,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            if admin_only:
                # Add more detailed admin statistics
                stats.update(await self._get_admin_statistics())
            
            return stats
            
        except Exception as e:
            raise Exception(f"Failed to get usage statistics: {str(e)}")
    
    async def _get_admin_statistics(self) -> Dict[str, Any]:
        """Get additional statistics for admin users."""
        try:
            # Most active users
            active_users_query = select(
                EmotionAnalysis.user_id,
                func.count(EmotionAnalysis.id).label('analysis_count')
            ).group_by(
                EmotionAnalysis.user_id
            ).order_by(
                desc('analysis_count')
            ).limit(10)
            
            active_result = await self.db.execute(active_users_query)
            active_users = [
                {"user_id": row.user_id, "analysis_count": row.analysis_count}
                for row in active_result
            ]
            
            # System performance metrics
            # Average confidence score
            confidence_query = select(func.avg(EmotionAnalysis.confidence_score))
            confidence_result = await self.db.execute(confidence_query)
            avg_confidence = confidence_result.scalar() or 0.0
            
            return {
                "most_active_users": active_users,
                "system_average_confidence": float(avg_confidence),
                "performance_metrics": {
                    "avg_confidence_score": float(avg_confidence),
                    "total_processing_hours": 0  # TODO: Implement processing time tracking
                }
            }
            
        except Exception as e:
            raise Exception(f"Failed to get admin statistics: {str(e)}")
    
    async def generate_user_report(
        self,
        user_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate a detailed report for a specific user.
        
        Args:
            user_id: User ID
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Detailed user report
        """
        try:
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()
            
            # Get analyses in date range
            query = select(EmotionAnalysis).where(
                and_(
                    EmotionAnalysis.user_id == user_id,
                    EmotionAnalysis.created_at >= start_date,
                    EmotionAnalysis.created_at <= end_date
                )
            ).order_by(EmotionAnalysis.created_at)
            
            result = await self.db.execute(query)
            analyses = result.scalars().all()
            
            # Analyze patterns
            emotion_patterns = self._analyze_emotion_patterns(analyses)
            temporal_patterns = self._analyze_temporal_patterns(analyses)
            confidence_analysis = self._analyze_confidence_patterns(analyses)
            
            return {
                "user_id": user_id,
                "report_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "total_days": (end_date - start_date).days
                },
                "summary": {
                    "total_analyses": len(analyses),
                    "analyses_by_type": {
                        analysis_type.value: len([a for a in analyses if a.analysis_type == analysis_type])
                        for analysis_type in AnalysisType
                    }
                },
                "emotion_patterns": emotion_patterns,
                "temporal_patterns": temporal_patterns,
                "confidence_analysis": confidence_analysis,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise Exception(f"Failed to generate user report: {str(e)}")
    
    def _analyze_emotion_patterns(self, analyses: List[EmotionAnalysis]) -> Dict[str, Any]:
        """Analyze emotion patterns in analyses."""
        emotion_counts = {}
        emotion_confidence = {}
        
        for analysis in analyses:
            dominant_emotion = analysis.results.get("dominant_emotion", "unknown")
            confidence = analysis.confidence_score
            
            if dominant_emotion not in emotion_counts:
                emotion_counts[dominant_emotion] = 0
                emotion_confidence[dominant_emotion] = []
            
            emotion_counts[dominant_emotion] += 1
            emotion_confidence[dominant_emotion].append(confidence)
        
        # Calculate averages
        emotion_avg_confidence = {
            emotion: sum(scores) / len(scores) if scores else 0
            for emotion, scores in emotion_confidence.items()
        }
        
        # Find most common emotion
        most_common = max(emotion_counts.items(), key=lambda x: x[1]) if emotion_counts else ("unknown", 0)
        
        return {
            "emotion_distribution": emotion_counts,
            "emotion_confidence_averages": emotion_avg_confidence,
            "most_common_emotion": {
                "emotion": most_common[0],
                "count": most_common[1],
                "percentage": (most_common[1] / len(analyses) * 100) if analyses else 0
            }
        }
    
    def _analyze_temporal_patterns(self, analyses: List[EmotionAnalysis]) -> Dict[str, Any]:
        """Analyze temporal patterns in analyses."""
        hourly_counts = {hour: 0 for hour in range(24)}
        daily_counts = {}
        
        for analysis in analyses:
            hour = analysis.created_at.hour
            date = analysis.created_at.date().isoformat()
            
            hourly_counts[hour] += 1
            
            if date not in daily_counts:
                daily_counts[date] = 0
            daily_counts[date] += 1
        
        # Find peak activity hour
        peak_hour = max(hourly_counts.items(), key=lambda x: x[1])
        
        return {
            "hourly_distribution": hourly_counts,
            "daily_counts": daily_counts,
            "peak_activity_hour": {
                "hour": peak_hour[0],
                "count": peak_hour[1]
            },
            "total_active_days": len(daily_counts)
        }
    
    def _analyze_confidence_patterns(self, analyses: List[EmotionAnalysis]) -> Dict[str, Any]:
        """Analyze confidence score patterns."""
        if not analyses:
            return {"average_confidence": 0, "confidence_trend": []}
        
        confidences = [analysis.confidence_score for analysis in analyses]
        
        return {
            "average_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "confidence_distribution": {
                "high": len([c for c in confidences if c >= 0.8]),
                "medium": len([c for c in confidences if 0.5 <= c < 0.8]),
                "low": len([c for c in confidences if c < 0.5])
            }
        }
