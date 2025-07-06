"""
API v1 Router for EmoSense Backend API

Main router that combines all API endpoints for version 1 of the API.
Organizes routes by feature area and provides consistent URL structure.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import auth, emotion, users, analytics, system


# Create the main API router for version 1
api_router = APIRouter(prefix="/v1")

# Include all endpoint routers
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"],
)

api_router.include_router(
    users.router,
    prefix="/users",
    tags=["Users"],
)

api_router.include_router(
    emotion.router,
    prefix="/emotion",
    tags=["Emotion Analysis"],
)

api_router.include_router(
    analytics.router,
    prefix="/analytics",
    tags=["Analytics"],
)

api_router.include_router(
    system.router,
    prefix="/system",
    tags=["System"],
)
