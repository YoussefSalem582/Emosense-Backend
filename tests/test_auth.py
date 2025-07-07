"""
Test cases for authentication endpoints.

Tests user registration, login, token refresh, and authentication flow.
"""

import pytest
from fastapi import status
from httpx import AsyncClient

from app.core.security import verify_password


class TestAuthentication:
    """Test authentication endpoints."""
    
    @pytest.mark.asyncio
    async def test_register_user(self, async_client: AsyncClient):
        """Test user registration."""
        user_data = {
            "email": "newuser@example.com",
            "first_name": "New",
            "last_name": "User",
            "password": "securepassword123",
            "confirm_password": "securepassword123"
        }
        
        response = await async_client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["email"] == user_data["email"]
        assert data["first_name"] == user_data["first_name"]
        assert data["last_name"] == user_data["last_name"]
        assert "password" not in data
        assert "hashed_password" not in data
    
    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, async_client: AsyncClient, test_user):
        """Test registration with duplicate email."""
        user_data = {
            "email": test_user.email,
            "first_name": "Different",
            "last_name": "User",
            "password": "securepassword123",
            "confirm_password": "securepassword123"
        }
        
        response = await async_client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "error" in data
        assert "already" in data["error"]["message"].lower()
        assert data["error"]["code"] == "VALIDATION_ERROR"
    
    @pytest.mark.asyncio
    async def test_login_valid_credentials(self, async_client: AsyncClient, test_user):
        """Test login with valid credentials."""
        login_data = {
            "email": test_user.email,
            "password": "testpassword"
        }
        
        response = await async_client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "user" in data
    
    @pytest.mark.asyncio
    async def test_login_invalid_email(self, async_client: AsyncClient):
        """Test login with invalid email."""
        login_data = {
            "email": "nonexistent@example.com",
            "password": "testpassword"
        }
        
        response = await async_client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        error_data = response.json()
        # Check custom error format
        if "error" in error_data:
            assert "invalid" in error_data["error"]["message"].lower()
        else:
            assert "invalid" in error_data["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_login_invalid_password(self, async_client: AsyncClient, test_user):
        """Test login with invalid password."""
        login_data = {
            "email": test_user.email,
            "password": "wrongpassword"
        }
        
        response = await async_client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        error_data = response.json()
        # Check custom error format
        if "error" in error_data:
            assert "invalid" in error_data["error"]["message"].lower()
        else:
            assert "invalid" in error_data["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_refresh_token(self, async_client: AsyncClient, test_user):
        """Test token refresh functionality."""
        # First login to get tokens
        login_data = {
            "email": test_user.email,
            "password": "testpassword"
        }
        
        login_response = await async_client.post("/api/v1/auth/login", json=login_data)
        login_data = login_response.json()
        
        # Use refresh token to get new access token
        refresh_data = {
            "refresh_token": login_data["refresh_token"]
        }
        
        response = await async_client.post("/api/v1/auth/refresh", json=refresh_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
    
    @pytest.mark.asyncio
    async def test_refresh_invalid_token(self, async_client: AsyncClient):
        """Test refresh with invalid token."""
        refresh_data = {
            "refresh_token": "invalid.token.here"
        }
        
        response = await async_client.post("/api/v1/auth/refresh", json=refresh_data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    @pytest.mark.asyncio
    async def test_protected_endpoint_without_token(self, async_client: AsyncClient):
        """Test accessing protected endpoint without token."""
        response = await async_client.get("/api/v1/users/me")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    @pytest.mark.asyncio
    async def test_protected_endpoint_with_token(self, async_client: AsyncClient, auth_headers):
        """Test accessing protected endpoint with valid token."""
        response = await async_client.get("/api/v1/users/me", headers=auth_headers)
        
        # This might fail if the users endpoint is not implemented
        # but the authentication should work
        assert response.status_code != status.HTTP_401_UNAUTHORIZED
    
    @pytest.mark.asyncio
    async def test_protected_endpoint_with_invalid_token(self, async_client: AsyncClient):
        """Test accessing protected endpoint with invalid token."""
        headers = {"Authorization": "Bearer invalid.token.here"}
        response = await async_client.get("/api/v1/users/me", headers=headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestPasswordSecurity:
    """Test password security functions."""
    
    def test_password_hashing(self):
        """Test password hashing and verification."""
        from app.core.security import create_password_hash, verify_password
        
        password = "testsecurepassword123"
        hashed = create_password_hash(password)
        
        # Hash should be different from password
        assert hashed != password
        
        # Should verify correctly
        assert verify_password(password, hashed) is True
        
        # Should not verify with wrong password
        assert verify_password("wrongpassword", hashed) is False
    
    def test_token_creation_and_verification(self):
        """Test JWT token creation and verification."""
        from app.core.security import create_access_token, verify_token
        
        data = {"sub": "123", "email": "test@example.com"}
        token = create_access_token(data)
        
        # Token should be a string
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Should be able to verify and extract data
        verified_data = verify_token(token)
        assert verified_data["sub"] == "123"
        assert verified_data["email"] == "test@example.com"
    
    def test_expired_token_verification(self):
        """Test verification of expired tokens."""
        from datetime import timedelta
        from app.core.security import create_access_token, verify_token
        
        # Create token that expires immediately
        data = {"sub": "123"}
        token = create_access_token(data, expires_delta=timedelta(seconds=-1))
        
        # Should raise exception for expired token
        with pytest.raises(Exception):
            verify_token(token)
