"""
System API Endpoints for EmoSense Backend API

Provides endpoints for system administration, health checks,
and monitoring functionality.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db_session
from app.dependencies import get_current_superuser, get_optional_user
from app.services.system import SystemHealthService


# Create router for system endpoints
router = APIRouter()


@router.get(
    "/health",
    summary="Health Check",
    description="Check system health and service availability"
)
async def health_check(
    db: AsyncSession = Depends(get_db_session),
    current_user = Depends(get_optional_user)
):
    """
    Perform a comprehensive health check of the system.
    
    This endpoint checks:
    - Database connectivity
    - Redis availability (if configured)
    - System resource usage
    - Application metrics
    
    Returns:
        Health status and system metrics
    """
    try:
        health_service = SystemHealthService(db=db)
        health_status = await health_service.get_health_status()
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "error"
        }


@router.get(
    "/metrics",
    summary="System Metrics",
    description="Get detailed system metrics and performance data"
)
async def get_system_metrics(
    db: AsyncSession = Depends(get_db_session),
    current_user = Depends(get_current_superuser)
):
    """
    Get detailed system metrics and performance data.
    
    Requires admin/superuser privileges.
    
    Returns:
        Comprehensive system and application metrics
    """
    try:
        health_service = SystemHealthService(db=db)
        metrics = await health_service.get_detailed_metrics()
        
        return {
            "status": "success",
            "data": metrics
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system metrics: {str(e)}"
        )


@router.get(
    "/info",
    summary="System Information",
    description="Get system information and configuration details"
)
async def get_system_info(
    db: AsyncSession = Depends(get_db_session),
    current_user = Depends(get_current_superuser)
):
    """
    Get detailed system information and configuration.
    
    Requires admin/superuser privileges.
    
    Returns:
        System information including platform details and configuration
    """
    try:
        health_service = SystemHealthService(db=db)
        system_info = await health_service.get_system_info()
        
        return {
            "status": "success",
            "data": system_info
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system information: {str(e)}"
        )


@router.post(
    "/cache/clear",
    summary="Clear Cache",
    description="Clear application caches (admin only)"
)
async def clear_cache(
    db: AsyncSession = Depends(get_db_session),
    current_user = Depends(get_current_superuser)
):
    """
    Clear application caches.
    
    Requires admin/superuser privileges.
    
    This endpoint clears various application caches including:
    - Redis cache
    - Model caches
    - File caches
    
    Returns:
        Cache clearing results
    """
    try:
        health_service = SystemHealthService(db=db)
        result = await health_service.clear_cache()
        
        return {
            "status": "success",
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.get(
    "/metrics/prometheus",
    summary="Prometheus Metrics",
    description="Get metrics in Prometheus format for monitoring",
    response_class="text/plain"
)
async def get_prometheus_metrics(
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get system metrics in Prometheus format.
    
    This endpoint provides metrics in a format compatible with
    Prometheus monitoring system.
    
    Returns:
        Metrics in Prometheus text format
    """
    try:
        health_service = SystemHealthService(db=db)
        metrics_text = health_service.get_prometheus_metrics()
        
        return metrics_text
        
    except Exception as e:
        return f"# Error generating metrics: {str(e)}"
