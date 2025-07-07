import requests
import json
import base64

# Test the deployed API
base_url = "https://emosense-backend-youssefsalem582s-projects.vercel.app"

def test_advanced_api():
    print("Testing EmoSense Backend API v2.0 on Vercel...")
    
    # Test root endpoint
    print("\n1. Testing root endpoint...")
    response = requests.get(f"{base_url}/")
    print(f"Status: {response.status_code}")
    try:
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
    except:
        print(f"Response (text): {response.text[:200]}...")
    
    # Test health endpoint
    print("\n2. Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"Status: {response.status_code}")
    try:
        data = response.json()
        print(f"Models available: {list(data.get('models', {}).keys())}")
        print(f"Total analyses: {data.get('total_analyses', 0)}")
    except:
        print(f"Response (text): {response.text[:200]}...")
    
    # Test models endpoint
    print("\n3. Testing models endpoint...")
    response = requests.get(f"{base_url}/api/v1/models")
    print(f"Status: {response.status_code}")
    try:
        data = response.json()
        print(f"Available models: {json.dumps(data, indent=2)}")
    except:
        print(f"Response (text): {response.text[:200]}...")
    
    # Test user registration
    print("\n4. Testing user registration...")
    user_data = {
        "email": "test@example.com",
        "username": "testuser",
        "password": "testpass123"
    }
    response = requests.post(f"{base_url}/api/v1/auth/register", json=user_data)
    print(f"Status: {response.status_code}")
    try:
        data = response.json()
        print(f"User registered: {data.get('user', {}).get('username')}")
        token = data.get("access_token")
    except:
        print(f"Response (text): {response.text[:200]}...")
        token = None
    
    # Test advanced text emotion analysis
    print("\n5. Testing advanced text emotion analysis...")
    text_data = {
        "text": "I am absolutely thrilled and excited about this amazing opportunity!",
        "model_type": "text"
    }
    response = requests.post(f"{base_url}/api/v1/emotion/analyze/text", json=text_data)
    print(f"Status: {response.status_code}")
    try:
        data = response.json()
        print(f"Dominant emotion: {data.get('dominant_emotion')}")
        print(f"Confidence: {data.get('confidence'):.3f}")
        print(f"Processing time: {data.get('processing_time'):.3f}s")
        print(f"Analysis ID: {data.get('analysis_id')}")
        print(f"Emotions: {json.dumps(data.get('emotions', {}), indent=2)}")
        analysis_id = data.get('analysis_id')
    except:
        print(f"Response (text): {response.text[:200]}...")
        analysis_id = None
    
    # Test multimodal analysis
    print("\n6. Testing multimodal emotion analysis...")
    multimodal_data = {
        "text": "This is such a sad and depressing situation",
        "model_type": "multimodal"
    }
    response = requests.post(f"{base_url}/api/v1/emotion/analyze/multimodal", json=multimodal_data)
    print(f"Status: {response.status_code}")
    try:
        data = response.json()
        print(f"Dominant emotion: {data.get('dominant_emotion')}")
        print(f"Modalities used: {data.get('metadata', {}).get('modalities_used', [])}")
        print(f"Processing time: {data.get('processing_time'):.3f}s")
    except:
        print(f"Response (text): {response.text[:200]}...")
    
    # Test analysis history
    print("\n7. Testing analysis history...")
    response = requests.get(f"{base_url}/api/v1/emotion/history?limit=5")
    print(f"Status: {response.status_code}")
    try:
        data = response.json()
        print(f"Total analyses in history: {data.get('total_count', 0)}")
        print(f"Recent analyses: {len(data.get('analyses', []))}")
    except:
        print(f"Response (text): {response.text[:200]}...")
    
    # Test specific analysis result
    if analysis_id:
        print(f"\n8. Testing specific analysis result ({analysis_id})...")
        response = requests.get(f"{base_url}/api/v1/emotion/analysis/{analysis_id}")
        print(f"Status: {response.status_code}")
        try:
            data = response.json()
            print(f"Analysis found: {data.get('id') == analysis_id}")
            print(f"Timestamp: {data.get('timestamp')}")
        except:
            print(f"Response (text): {response.text[:200]}...")
    
    # Test analytics dashboard
    print("\n9. Testing analytics dashboard...")
    response = requests.get(f"{base_url}/api/v1/analytics/dashboard")
    print(f"Status: {response.status_code}")
    try:
        data = response.json()
        stats = data.get('analysis_stats', {})
        print(f"Total analyses: {stats.get('total_analyses', 0)}")
        print(f"Models available: {stats.get('models_available', 0)}")
        print(f"Emotion distribution: {json.dumps(stats.get('emotion_distribution', {}), indent=2)}")
    except:
        print(f"Response (text): {response.text[:200]}...")
    
    # Test legacy endpoint for backward compatibility
    print("\n10. Testing legacy emotion analysis endpoint...")
    legacy_data = {"text": "I feel really angry about this situation!"}
    response = requests.post(f"{base_url}/api/v1/emotion/analyze", json=legacy_data)
    print(f"Status: {response.status_code}")
    try:
        data = response.json()
        print(f"Legacy endpoint works: {data.get('dominant_emotion') is not None}")
        print(f"Dominant emotion: {data.get('dominant_emotion')}")
    except:
        print(f"Response (text): {response.text[:200]}...")

def test_file_upload_simulation():
    """Test file upload endpoint with simulated data"""
    print("\n11. Testing file upload simulation...")
    
    # Create a dummy audio file content (base64 encoded)
    dummy_audio = b"fake_audio_data_for_testing"
    audio_base64 = base64.b64encode(dummy_audio).decode()
    
    # Prepare multipart form data
    files = {
        'file': ('test_audio.wav', dummy_audio, 'audio/wav')
    }
    data = {
        'analysis_type': 'audio',
        'text': 'This is accompanying text'
    }
    
    response = requests.post(f"{base_url}/api/v1/emotion/analyze/file", files=files, data=data)
    print(f"Status: {response.status_code}")
    try:
        result = response.json()
        print(f"File analysis result: {result.get('dominant_emotion')}")
        print(f"File metadata: {result.get('metadata', {})}")
    except:
        print(f"Response (text): {response.text[:200]}...")

def test_different_emotions():
    """Test various emotion types"""
    print("\n12. Testing different emotion types...")
    
    test_texts = [
        ("I'm so happy and joyful today!", "joy"),
        ("This is absolutely terrible and awful", "sadness"),
        ("I'm furious and really angry about this", "anger"),
        ("I'm scared and really afraid", "fear"),
        ("Wow, this is so surprising and amazing!", "surprise"),
        ("This is disgusting and revolting", "disgust"),
        ("The weather is okay today", "neutral")
    ]
    
    for text, expected in test_texts:
        data = {"text": text, "model_type": "text"}
        response = requests.post(f"{base_url}/api/v1/emotion/analyze/text", json=data)
        
        if response.status_code == 200:
            try:
                result = response.json()
                detected = result.get('dominant_emotion')
                confidence = result.get('confidence', 0)
                print(f"Text: '{text[:30]}...' -> {detected} ({confidence:.2f})")
            except:
                print(f"Failed to parse response for: {text[:30]}...")
        else:
            print(f"Failed request for: {text[:30]}... (Status: {response.status_code})")

if __name__ == "__main__":
    test_advanced_api()
    test_file_upload_simulation()
    test_different_emotions()
    
    print("\n" + "="*60)
    print("✅ EmoSense Backend API v2.0 Testing Complete!")
    print("="*60)
    print("\n🚀 Ready for real model integration!")
    print("📚 Check model_integration_guide.py for implementation details")
    print("🔧 Update emotion_models.py with your preferred models")
    print("📖 Visit /docs for interactive API documentation")
