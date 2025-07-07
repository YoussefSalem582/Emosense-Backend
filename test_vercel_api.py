import requests
import json

# Test the deployed API
base_url = "https://emosense-backend-dmi6hxw0c-youssefsalem582s-projects.vercel.app"

def test_api():
    print("Testing EmoSense Backend API on Vercel...")
    
    # Test root endpoint
    print("\n1. Testing root endpoint...")
    response = requests.get(f"{base_url}/")
    print(f"Status: {response.status_code}")
    try:
        print(f"Response: {response.json()}")
    except:
        print(f"Response (text): {response.text[:200]}...")
    
    # Test health endpoint
    print("\n2. Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"Status: {response.status_code}")
    try:
        print(f"Response: {response.json()}")
    except:
        print(f"Response (text): {response.text[:200]}...")
    
    # Test user registration
    print("\n3. Testing user registration...")
    user_data = {
        "email": "test@example.com",
        "username": "testuser",
        "password": "testpass123"
    }
    response = requests.post(f"{base_url}/api/v1/auth/register", json=user_data)
    print(f"Status: {response.status_code}")
    try:
        print(f"Response: {response.json()}")
    except:
        print(f"Response (text): {response.text[:200]}...")
    
    if response.status_code == 200:
        try:
            token = response.json().get("access_token")
        except:
            token = None
        
        # Test emotion analysis
        print("\n4. Testing emotion analysis...")
        emotion_data = {"text": "I am so happy today!"}
        response = requests.post(f"{base_url}/api/v1/emotion/analyze", json=emotion_data)
        print(f"Status: {response.status_code}")
        try:
            print(f"Response: {response.json()}")
        except:
            print(f"Response (text): {response.text[:200]}...")
        
        # Test analytics
        print("\n5. Testing analytics...")
        response = requests.get(f"{base_url}/api/v1/analytics/dashboard")
        print(f"Status: {response.status_code}")
        try:
            print(f"Response: {response.json()}")
        except:
            print(f"Response (text): {response.text[:200]}...")

if __name__ == "__main__":
    test_api()
