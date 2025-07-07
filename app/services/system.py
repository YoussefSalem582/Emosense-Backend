"""
System Health Service for EmoSense Backend API

Provides system health monitoring, metrics collection, and status checking
functionality for monitoring application performance and availability.
"""

import os
import psutil
from datetime import datetime
from typing import Dict, Any, List
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.config import get_settings


class SystemHealthService:
    """
    System health monitoring service.
    
    Provides methods to check system health, collect metrics,
    and monitor application performance.
    """
    
    def __init__(self, db: AsyncSession):
        """
        Initialize system health service.
        
        Args:
            db: Database session
        """
        self.db = db
        self.settings = get_settings()
    
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            Dictionary with health status information
        """
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": self.settings.VERSION,
                "environment": self.settings.ENVIRONMENT,
                "services": {},
                "metrics": {}
            }
            
            # Check database connectivity
            db_health = await self._check_database_health()
            health_status["services"]["database"] = db_health
            
            # Check Redis connectivity (if configured)
            redis_health = await self._check_redis_health()
            health_status["services"]["redis"] = redis_health
            
            # Get system metrics
            system_metrics = self._get_system_metrics()
            health_status["metrics"]["system"] = system_metrics
            
            # Get application metrics
            app_metrics = await self._get_application_metrics()
            health_status["metrics"]["application"] = app_metrics
            
            # Determine overall status
            service_statuses = [service["status"] for service in health_status["services"].values()]
            if "unhealthy" in service_statuses:
                health_status["status"] = "unhealthy"
            elif "degraded" in service_statuses:
                health_status["status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "version": self.settings.VERSION
            }
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            start_time = datetime.utcnow()
            
            # Simple connectivity test
            result = await self.db.execute(text("SELECT 1"))
            result.scalar()
            
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds() * 1000  # ms
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "last_checked": end_time.isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_checked": datetime.utcnow().isoformat()
            }
    
    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance."""
        try:
            # TODO: Implement Redis health check
            # This would require redis client initialization
            
            return {
                "status": "healthy",
                "note": "Redis health check not implemented",
                "last_checked": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_checked": datetime.utcnow().isoformat()
            }
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available
            memory_total = memory.total
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free = disk.free
            disk_total = disk.total
            
            # Network metrics (if available)
            try:
                network = psutil.net_io_counters()
                network_metrics = {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }
            except:
                network_metrics = {"error": "Network metrics unavailable"}
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count
                },
                "memory": {
                    "percent": memory_percent,
                    "available_bytes": memory_available,
                    "total_bytes": memory_total,
                    "available_gb": round(memory_available / (1024**3), 2),
                    "total_gb": round(memory_total / (1024**3), 2)
                },
                "disk": {
                    "percent": disk_percent,
                    "free_bytes": disk_free,
                    "total_bytes": disk_total,
                    "free_gb": round(disk_free / (1024**3), 2),
                    "total_gb": round(disk_total / (1024**3), 2)
                },
                "network": network_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _get_application_metrics(self) -> Dict[str, Any]:
        """Get application-level metrics."""
        try:
            # Process metrics
            process = psutil.Process()
            
            app_metrics = {
                "process": {
                    "pid": process.pid,
                    "memory_percent": process.memory_percent(),
                    "cpu_percent": process.cpu_percent(),
                    "create_time": process.create_time(),
                    "num_threads": process.num_threads()
                },
                "python": {
                    "version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                    "executable": os.sys.executable
                },
                "environment": {
                    "name": self.settings.ENVIRONMENT,
                    "debug": self.settings.DEBUG,
                    "host": self.settings.HOST,
                    "port": self.settings.PORT
                },
                "uptime_seconds": datetime.utcnow().timestamp() - process.create_time()
            }
            
            return app_metrics
            
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_detailed_metrics(self) -> Dict[str, Any]:
        """
        Get detailed system and application metrics.
        
        Returns:
            Comprehensive metrics dictionary
        """
        try:
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "system": self._get_system_metrics(),
                "application": await self._get_application_metrics(),
                "database": await self._get_database_metrics(),
                "performance": await self._get_performance_metrics()
            }
            
            return metrics
            
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _get_database_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics."""
        try:
            # Get database connection info
            db_metrics = {
                "connection_url": self.settings.DATABASE_URL.split('@')[1] if '@' in self.settings.DATABASE_URL else "hidden",
                "pool_size": self.settings.DATABASE_POOL_SIZE,
                "max_overflow": self.settings.DATABASE_MAX_OVERFLOW
            }
            
            # TODO: Add more detailed database metrics
            # - Active connections
            # - Query performance
            # - Table sizes
            
            return db_metrics
            
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get application performance metrics."""
        try:
            # TODO: Implement performance tracking
            # This could include:
            # - Request response times
            # - Analysis processing times
            # - Error rates
            # - Throughput metrics
            
            performance_metrics = {
                "note": "Performance metrics collection not implemented",
                "suggestions": [
                    "Implement request timing middleware",
                    "Add analysis processing time tracking",
                    "Monitor error rates and types",
                    "Track API endpoint usage patterns"
                ]
            }
            
            return performance_metrics
            
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_system_info(self) -> Dict[str, Any]:
        """
        Get detailed system information.
        
        Returns:
            System information dictionary
        """
        try:
            import platform
            
            system_info = {
                "platform": {
                    "system": platform.system(),
                    "node": platform.node(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                    "python_version": platform.python_version()
                },
                "application": {
                    "name": self.settings.APP_NAME,
                    "version": self.settings.VERSION,
                    "environment": self.settings.ENVIRONMENT,
                    "debug_mode": self.settings.DEBUG
                },
                "configuration": {
                    "host": self.settings.HOST,
                    "port": self.settings.PORT,
                    "database_configured": bool(self.settings.DATABASE_URL),
                    "redis_configured": bool(self.settings.REDIS_URL),
                    "cors_origins": len(self.settings.cors_origins_list),
                    "max_file_size_mb": self.settings.MAX_FILE_SIZE / (1024 * 1024)
                },
                "features": {
                    "text_analysis": True,
                    "video_analysis": True,
                    "audio_analysis": True,
                    "batch_processing": True,
                    "analytics": True,
                    "user_management": True
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return system_info
            
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def clear_cache(self) -> Dict[str, Any]:
        """
        Clear application caches.
        
        Returns:
            Cache clearing results
        """
        try:
            # TODO: Implement cache clearing
            # This could include:
            # - Redis cache clearing
            # - Model cache clearing
            # - File cache clearing
            
            result = {
                "status": "success",
                "message": "Cache clearing not implemented",
                "timestamp": datetime.utcnow().isoformat(),
                "caches_cleared": []
            }
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_prometheus_metrics(self) -> str:
        """
        Get metrics in Prometheus format.
        
        Returns:
            Prometheus-formatted metrics string
        """
        try:
            # TODO: Implement Prometheus metrics collection
            # This would typically use prometheus_client library
            
            metrics_lines = [
                "# HELP emosense_info Application information",
                "# TYPE emosense_info gauge",
                f'emosense_info{{version="{self.settings.VERSION}",environment="{self.settings.ENVIRONMENT}"}} 1',
                "",
                "# HELP emosense_uptime_seconds Application uptime in seconds",
                "# TYPE emosense_uptime_seconds counter",
                "emosense_uptime_seconds 0",  # TODO: Calculate actual uptime
                "",
                "# HELP emosense_health_status Application health status (1=healthy, 0=unhealthy)",
                "# TYPE emosense_health_status gauge",
                "emosense_health_status 1",  # TODO: Use actual health status
            ]
            
            return "\\n".join(metrics_lines)
            
        except Exception as e:
            return f"# Error generating metrics: {str(e)}"
