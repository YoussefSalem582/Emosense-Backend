"""
EmoSense Backend API - Main Application Module

FastAPI application for emotion analysis providing endpoints for text, video,
and audio emotion detection with user authentication and analytics.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from app.api.v1.router import api_router
from app.config import get_settings
from app.core.exceptions import CustomHTTPException
from app.database import create_tables, get_engine


# Configure structured logging
logger = structlog.get_logger(__name__)

# Get application settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler for startup and shutdown events.
    
    Args:
        app: FastAPI application instance
        
    Yields:
        None during application runtime
    """
    # Startup
    logger.info("Starting EmoSense Backend API", version=settings.VERSION)
    
    # Initialize database tables
    try:
        await create_tables()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise
    
    # Initialize ML models (if needed)
    # TODO: Initialize emotion analysis models here
    
    yield
    
    # Shutdown
    logger.info("Shutting down EmoSense Backend API")
    
    # Close database connections
    engine = get_engine()
    await engine.dispose()
    logger.info("Database connections closed")


# Create FastAPI application instance
app = FastAPI(
    title="EmoSense Backend API",
    description="Comprehensive emotion analysis API for text, video, and audio processing",
    version=settings.VERSION,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
    lifespan=lifespan,
    openapi_url="/openapi.json" if settings.ENVIRONMENT != "production" else None,
)

# Add security middleware
if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(CustomHTTPException)
async def custom_http_exception_handler(
    request: Request, 
    exc: CustomHTTPException
) -> JSONResponse:
    """
    Custom HTTP exception handler for structured error responses.
    
    Args:
        request: The incoming request
        exc: The custom HTTP exception
        
    Returns:
        JSON response with error details
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.detail,
                "details": exc.details,
                "timestamp": exc.timestamp.isoformat(),
            }
        },
        headers=exc.headers,
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all HTTP requests for monitoring and debugging.
    
    Args:
        request: The incoming request
        call_next: The next middleware/handler in the chain
        
    Returns:
        The response from the next handler
    """
    start_time = time.time()
    
    # Log request
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None,
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    logger.info(
        "Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=f"{process_time:.4f}s",
    )
    
    # Add processing time header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Include API routes
app.include_router(api_router, prefix="/api")

# Add metrics endpoint for monitoring
if settings.ENABLE_METRICS:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)


@app.get("/", tags=["Root"])
async def root() -> dict:
    """
    Root endpoint providing basic API information.
    
    Returns:
        Basic API information and status
    """
    return {
        "name": "EmoSense Backend API",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "status": "healthy",
        "docs_url": "/docs" if settings.ENVIRONMENT != "production" else None,
    }


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns:
        Health status of the application and its dependencies
    """
    from app.services.health import get_health_status
    
    try:
        health_status = await get_health_status()
        return health_status
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e),
        }


if __name__ == "__main__":
    import time
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning",
    )
