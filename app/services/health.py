"""
Health Check Service for EmoSense Backend API

Provides health check functionality for monitoring application
and dependency status including database, cache, and external services.
"""

import asyncio
import time
from typing import Dict, Any

import redis.asyncio as redis

from app.config import get_settings
from app.database import DatabaseManager


settings = get_settings()


class HealthCheckService:
    """
    Service for performing comprehensive health checks.
    
    Checks the status of various application components including
    database connectivity, cache availability, and system resources.
    """
    
    def __init__(self):
        """Initialize health check service."""
        self.db_manager = DatabaseManager()
    
    async def check_database(self) -> Dict[str, Any]:
        """
        Check database connectivity and status.
        
        Returns:
            Dict with database health information
        """
        try:
            start_time = time.time()
            is_connected = await self.db_manager.test_connection()
            response_time = time.time() - start_time
            
            if is_connected:
                connection_info = await self.db_manager.get_connection_info()
                return {
                    "status": "healthy",
                    "response_time": round(response_time * 1000, 2),  # ms
                    "details": connection_info
                }
            else:
                return {
                    "status": "unhealthy",
                    "response_time": round(response_time * 1000, 2),
                    "error": "Cannot connect to database"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def check_redis(self) -> Dict[str, Any]:
        """
        Check Redis connectivity and status.
        
        Returns:
            Dict with Redis health information
        """
        try:
            start_time = time.time()
            redis_client = redis.from_url(settings.REDIS_URL)
            
            # Test basic connectivity
            await redis_client.ping()
            response_time = time.time() - start_time
            
            # Get Redis info
            info = await redis_client.info()
            await redis_client.close()
            
            return {
                "status": "healthy",
                "response_time": round(response_time * 1000, 2),
                "details": {
                    "version": info.get("redis_version"),
                    "used_memory": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                    "uptime": info.get("uptime_in_seconds")
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def check_disk_space(self) -> Dict[str, Any]:
        """
        Check available disk space.
        
        Returns:
            Dict with disk space information
        """
        try:
            import shutil
            
            total, used, free = shutil.disk_usage("/")
            free_percent = (free / total) * 100
            
            status = "healthy" if free_percent > 10 else "warning" if free_percent > 5 else "critical"
            
            return {
                "status": status,
                "details": {
                    "total_gb": round(total / (1024**3), 2),
                    "used_gb": round(used / (1024**3), 2),
                    "free_gb": round(free / (1024**3), 2),
                    "free_percent": round(free_percent, 2)
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def check_memory_usage(self) -> Dict[str, Any]:
        """
        Check system memory usage.
        
        Returns:
            Dict with memory usage information
        """
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            available_percent = memory.available / memory.total * 100
            
            status = "healthy" if available_percent > 20 else "warning" if available_percent > 10 else "critical"
            
            return {
                "status": status,
                "details": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_percent": round(memory.percent, 2),
                    "available_percent": round(available_percent, 2)
                }
            }
        except ImportError:
            return {
                "status": "unknown",
                "error": "psutil not available"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def get_comprehensive_health(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of all components.
        
        Returns:
            Dict with overall health information
        """
        start_time = time.time()
        
        # Run all health checks concurrently
        database_check, redis_check, disk_check, memory_check = await asyncio.gather(
            self.check_database(),
            self.check_redis(),
            self.check_disk_space(),
            self.check_memory_usage(),
            return_exceptions=True
        )
        
        # Handle any exceptions from the checks
        def safe_check_result(check_result, component_name):
            if isinstance(check_result, Exception):
                return {
                    "status": "unhealthy",
                    "error": f"{component_name} check failed: {str(check_result)}"
                }
            return check_result
        
        database_status = safe_check_result(database_check, "Database")
        redis_status = safe_check_result(redis_check, "Redis")
        disk_status = safe_check_result(disk_check, "Disk")
        memory_status = safe_check_result(memory_check, "Memory")
        
        # Determine overall status
        component_statuses = [
            database_status["status"],
            redis_status["status"],
            disk_status["status"],
            memory_status["status"]
        ]
        
        if "unhealthy" in component_statuses:
            overall_status = "unhealthy"
        elif "critical" in component_statuses:
            overall_status = "critical"
        elif "warning" in component_statuses:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        total_time = time.time() - start_time
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "check_duration": round(total_time * 1000, 2),  # ms
            "components": {
                "database": database_status,
                "redis": redis_status,
                "disk": disk_status,
                "memory": memory_status
            }
        }


# Global health check service instance
health_service = HealthCheckService()


async def get_health_status() -> Dict[str, Any]:
    """
    Convenience function to get health status.
    
    Returns:
        Dict with health status information
    """
    return await health_service.get_comprehensive_health()
