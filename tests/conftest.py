"""
Pytest configuration and fixtures for EmoSense Backend API tests.

Provides common test fixtures, database setup, and testing utilities
for comprehensive testing of the emotion analysis API.
"""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.database import Base, get_db_session
from app.config import get_settings
from app.core.security import create_access_token
from app.models.user import User


# Test database URL (SQLite for testing)
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"


@pytest.fixture(scope="session")
def test_settings():
    """Override settings for testing."""
    from app.config import Settings
    import os
    
    # Set test environment variables
    os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only"
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["DEBUG"] = "True"
    
    return Settings()


@pytest.fixture(autouse=True)
def override_settings(test_settings):
    """Automatically override settings for all tests."""
    from app.config import get_settings
    from app.main import app
    
    app.dependency_overrides[get_settings] = lambda: test_settings
    yield
    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def test_db() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    TestSessionLocal = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with TestSessionLocal() as session:
        yield session
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture
def client(test_db):
    """Create a test client with dependency overrides."""
    def override_get_db():
        yield test_db
    
    app.dependency_overrides[get_db_session] = override_get_db
    
    with TestClient(app) as c:
        yield c
    
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def async_client(test_db) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client."""
    from app.config import get_settings
    
    async def override_get_db():
        yield test_db
    
    def override_get_settings():
        from app.config import Settings
        import os
        
        # Set test environment variables
        os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only"
        os.environ["DATABASE_URL"] = TEST_DATABASE_URL
        os.environ["ENVIRONMENT"] = "testing"
        os.environ["DEBUG"] = "True"
        
        return Settings()
    
    app.dependency_overrides[get_db_session] = override_get_db
    app.dependency_overrides[get_settings] = override_get_settings
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def test_user(test_db: AsyncSession) -> User:
    """Create a test user."""
    from app.core.security import create_password_hash
    
    user = User(
        email="test@example.com",
        first_name="Test",
        last_name="User",
        hashed_password=create_password_hash("testpassword"),
        is_active=True
    )
    
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    
    return user


@pytest_asyncio.fixture
async def test_superuser(test_db: AsyncSession) -> User:
    """Create a test superuser."""
    from app.core.security import create_password_hash
    
    user = User(
        email="admin@example.com",
        first_name="Admin",
        last_name="User",
        hashed_password=create_password_hash("adminpassword"),
        is_active=True,
        is_superuser=True
    )
    
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    
    return user


@pytest.fixture
def auth_headers(test_user: User) -> dict:
    """Create authentication headers for a test user."""
    token = create_access_token(data={"sub": str(test_user.id)})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers(test_superuser: User) -> dict:
    """Create authentication headers for a test superuser."""
    token = create_access_token(data={"sub": str(test_superuser.id)})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_text_data() -> dict:
    """Sample text data for testing."""
    return {
        "text": "I am feeling very happy today! This is wonderful news.",
        "language": "en",
        "confidence_threshold": 0.5,
        "segment_analysis": True
    }


@pytest.fixture
def sample_video_file() -> str:
    """Path to a sample video file for testing."""
    # In a real test environment, this would point to a test video file
    return "tests/fixtures/sample_video.mp4"


@pytest.fixture
def sample_audio_file() -> str:
    """Path to a sample audio file for testing."""
    # In a real test environment, this would point to a test audio file
    return "tests/fixtures/sample_audio.wav"


@pytest.fixture
def mock_emotion_results() -> dict:
    """Mock emotion analysis results for testing."""
    return {
        "dominant_emotion": "happy",
        "emotions": {
            "happy": 0.85,
            "neutral": 0.10,
            "sad": 0.05
        },
        "confidence_score": 0.85,
        "metadata": {
            "processing_time": 1.23,
            "model_version": "test-1.0"
        }
    }


# Test utilities
class TestHelpers:
    """Helper functions for testing."""
    
    @staticmethod
    def assert_emotion_response(response_data: dict):
        """Assert that a response contains valid emotion analysis data."""
        assert "id" in response_data
        assert "analysis_type" in response_data
        assert "results" in response_data
        assert "confidence_score" in response_data
        assert "created_at" in response_data
        
        results = response_data["results"]
        assert "dominant_emotion" in results
        assert "emotions" in results
        assert isinstance(results["emotions"], dict)
        assert 0 <= response_data["confidence_score"] <= 1
    
    @staticmethod
    def assert_error_response(response_data: dict, expected_status: int = None):
        """Assert that a response contains valid error information."""
        assert "detail" in response_data
        if expected_status:
            assert response_data.get("status_code") == expected_status
    
    @staticmethod
    def assert_user_response(response_data: dict):
        """Assert that a response contains valid user data."""
        assert "id" in response_data
        assert "email" in response_data
        assert "first_name" in response_data
        assert "last_name" in response_data
        assert "is_active" in response_data
        # Should not contain password
        assert "hashed_password" not in response_data
        assert "password" not in response_data


@pytest.fixture
def test_helpers() -> TestHelpers:
    """Provide test helper functions."""
    return TestHelpers()
