"""
Vercel deployment entry point for EmoSense Backend API
"""

# Import the simplified FastAPI app for Vercel
from vercel_app import app

# This is the ASGI app that Vercel will serve
handler = app
