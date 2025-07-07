#!/usr/bin/env python3
"""
Comprehensive test script for EmoSense Backend with Real Models
Tests all emotion analysis capabilities including text, audio, video, and multimodal
"""

import asyncio
import requests
import json
import base64
import time
from typing import Dict, Any
import io

# Test configuration
BASE_URL = "https://your-emosense-backend.vercel.app"  # Update with your deployment URL
LOCAL_URL = "http://localhost:8000"  # For local testing

# Use local for testing, change to BASE_URL for production
API_URL = LOCAL_URL

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_result(test_name: str, result: Dict[str, Any]):
    """Print formatted test results"""
    print(f"\n🧪 {test_name}")
    print("-" * 40)
    
    if "emotions" in result:
        print(f"✅ Dominant Emotion: {result.get('dominant_emotion', 'N/A')}")
        print(f"📊 Confidence: {result.get('confidence', 0):.3f}")
        print(f"⏱️  Processing Time: {result.get('processing_time', 0):.3f}s")
        print(f"🤖 Model Version: {result.get('model_version', 'N/A')}")
        
        print("\n📈 Emotion Breakdown:")
        emotions = result.get('emotions', {})
        for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 20)
            print(f"  {emotion:>10}: {score:.3f} {bar}")
        
        if "metadata" in result:
            print(f"\n🔧 Metadata: {json.dumps(result['metadata'], indent=2)}")
    else:
        print(f"❌ Error: {result}")

def test_authentication():
    """Test user registration and login"""
    print_section("AUTHENTICATION TESTS")
    
    # Test registration
    register_data = {
        "email": "test@emosense.ai",
        "password": "SecurePassword123",
        "username": "TestUser"
    }
    
    try:
        response = requests.post(f"{API_URL}/api/v1/auth/register", json=register_data)
        if response.status_code == 200:
            result = response.json()
            print("✅ Registration successful")
            print(f"🔑 Access Token: {result.get('access_token', 'N/A')[:20]}...")
            return result.get('access_token')
        else:
            print(f"⚠️ Registration failed or user exists: {response.status_code}")
    except Exception as e:
        print(f"❌ Registration error: {e}")
    
    # Test login
    login_data = {
        "email": register_data["email"],
        "password": register_data["password"]
    }
    
    try:
        response = requests.post(f"{API_URL}/api/v1/auth/login", json=login_data)
        if response.status_code == 200:
            result = response.json()
            print("✅ Login successful")
            print(f"👤 User: {result.get('user', {}).get('username', 'N/A')}")
            return result.get('access_token')
        else:
            print(f"❌ Login failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Login error: {e}")
    
    return None

def test_health_and_models():
    """Test health check and model availability"""
    print_section("HEALTH CHECK & MODEL AVAILABILITY")
    
    try:
        # Health check
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            result = response.json()
            print("✅ API Health: OK")
            print(f"📊 Users: {result.get('users_count', 0)}")
            print(f"🔗 Sessions: {result.get('sessions_count', 0)}")
            print(f"📈 Total Analyses: {result.get('total_analyses', 0)}")
        
        # Available models
        response = requests.get(f"{API_URL}/api/v1/models")
        if response.status_code == 200:
            result = response.json()
            print("\n🤖 Available Models:")
            models = result.get('models', {})
            for name, info in models.items():
                status = "✅ Loaded" if info.get('loaded') else "⏳ Not Loaded"
                print(f"  {name:>12}: {info.get('name', 'N/A')} v{info.get('version', '0.0.0')} {status}")
                print(f"               Emotions: {', '.join(info.get('supported_emotions', []))}")
        
    except Exception as e:
        print(f"❌ Health check error: {e}")

def test_text_emotion_analysis():
    """Test text emotion analysis with various examples"""
    print_section("TEXT EMOTION ANALYSIS")
    
    test_texts = [
        {
            "name": "Joy/Happiness",
            "text": "I'm absolutely thrilled and excited about this amazing opportunity! This is the best day ever!"
        },
        {
            "name": "Sadness",
            "text": "I feel so lonely and heartbroken. Everything seems dark and hopeless right now."
        },
        {
            "name": "Anger",
            "text": "I'm absolutely furious and enraged! This is completely unacceptable and makes me mad!"
        },
        {
            "name": "Fear/Anxiety",
            "text": "I'm terrified and really worried about what might happen. This situation scares me so much."
        },
        {
            "name": "Surprise",
            "text": "Wow! I can't believe this just happened! This is so unexpected and shocking!"
        },
        {
            "name": "Disgust",
            "text": "This is absolutely revolting and disgusting. I feel sick just thinking about it."
        },
        {
            "name": "Neutral/Mixed",
            "text": "Today I went to the store and bought some groceries. The weather was okay."
        },
        {
            "name": "Complex Emotions",
            "text": "I'm excited about the new job but also nervous and scared about the challenges ahead."
        }
    ]
    
    for test_case in test_texts:
        try:
            response = requests.post(
                f"{API_URL}/api/v1/emotion/analyze/text",
                json={
                    "text": test_case["text"],
                    "model_type": "text"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print_result(test_case["name"], result)
            else:
                print(f"❌ {test_case['name']} failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ {test_case['name']} error: {e}")

def create_synthetic_audio_data() -> str:
    """Create synthetic audio data for testing"""
    # Create a simple synthetic WAV file as bytes
    import struct
    import math
    
    sample_rate = 44100
    duration = 2  # seconds
    frequency = 440  # A4 note
    
    # Generate sine wave
    samples = []
    for i in range(int(sample_rate * duration)):
        time_point = i / sample_rate
        amplitude = 0.3 * math.sin(2 * math.pi * frequency * time_point)
        samples.append(struct.pack('<h', int(amplitude * 32767)))
    
    # Create WAV header
    wav_data = b'RIFF'
    wav_data += struct.pack('<I', 36 + len(samples) * 2)
    wav_data += b'WAVE'
    wav_data += b'fmt '
    wav_data += struct.pack('<I', 16)
    wav_data += struct.pack('<H', 1)  # PCM format
    wav_data += struct.pack('<H', 1)  # Mono
    wav_data += struct.pack('<I', sample_rate)
    wav_data += struct.pack('<I', sample_rate * 2)
    wav_data += struct.pack('<H', 2)
    wav_data += struct.pack('<H', 16)
    wav_data += b'data'
    wav_data += struct.pack('<I', len(samples) * 2)
    wav_data += b''.join(samples)
    
    return base64.b64encode(wav_data).decode()

def test_audio_emotion_analysis():
    """Test audio emotion analysis"""
    print_section("AUDIO EMOTION ANALYSIS")
    
    try:
        # Create synthetic audio for testing
        audio_data = create_synthetic_audio_data()
        
        response = requests.post(
            f"{API_URL}/api/v1/emotion/analyze/multimodal",
            json={
                "audio_base64": audio_data,
                "model_type": "audio"
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print_result("Synthetic Audio Analysis", result)
        else:
            print(f"❌ Audio analysis failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Audio analysis error: {e}")

def create_synthetic_image_data() -> str:
    """Create synthetic image data for testing"""
    # Create a simple colored rectangle as PNG
    from PIL import Image
    import io
    
    # Create a 100x100 image with a simple pattern
    image = Image.new('RGB', (100, 100), color='lightblue')
    
    # Add some patterns to make it more interesting
    pixels = image.load()
    for x in range(100):
        for y in range(100):
            if (x + y) % 20 < 10:
                pixels[x, y] = (255, 200, 200)  # Light red
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return base64.b64encode(img_byte_arr).decode()

def test_video_emotion_analysis():
    """Test video/image emotion analysis"""
    print_section("VIDEO/IMAGE EMOTION ANALYSIS")
    
    try:
        # Try to create synthetic image
        try:
            image_data = create_synthetic_image_data()
            
            response = requests.post(
                f"{API_URL}/api/v1/emotion/analyze/multimodal",
                json={
                    "video_base64": image_data,
                    "model_type": "video"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print_result("Synthetic Image Analysis", result)
            else:
                print(f"❌ Video analysis failed: {response.status_code}")
                print(f"Response: {response.text}")
                
        except ImportError:
            print("⚠️ PIL not available, testing with dummy data")
            
            # Use dummy binary data
            dummy_image = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100  # Minimal PNG-like data
            image_data = base64.b64encode(dummy_image).decode()
            
            response = requests.post(
                f"{API_URL}/api/v1/emotion/analyze/multimodal",
                json={
                    "video_base64": image_data,
                    "model_type": "video"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print_result("Dummy Image Analysis", result)
            else:
                print(f"❌ Video analysis with dummy data failed: {response.status_code}")
                
    except Exception as e:
        print(f"❌ Video analysis error: {e}")

def test_multimodal_analysis():
    """Test multimodal emotion analysis"""
    print_section("MULTIMODAL EMOTION ANALYSIS")
    
    try:
        # Create test data
        audio_data = create_synthetic_audio_data()
        
        try:
            image_data = create_synthetic_image_data()
        except ImportError:
            # Fallback to dummy data
            dummy_image = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
            image_data = base64.b64encode(dummy_image).decode()
        
        # Test multimodal with all modalities
        multimodal_tests = [
            {
                "name": "Text + Audio + Video",
                "data": {
                    "text": "I'm feeling really happy and excited about this project!",
                    "audio_base64": audio_data,
                    "video_base64": image_data,
                    "model_type": "multimodal"
                }
            },
            {
                "name": "Text + Audio",
                "data": {
                    "text": "This is such a sad and disappointing situation.",
                    "audio_base64": audio_data,
                    "model_type": "multimodal"
                }
            },
            {
                "name": "Audio + Video",
                "data": {
                    "audio_base64": audio_data,
                    "video_base64": image_data,
                    "model_type": "multimodal"
                }
            }
        ]
        
        for test_case in multimodal_tests:
            try:
                response = requests.post(
                    f"{API_URL}/api/v1/emotion/analyze/multimodal",
                    json=test_case["data"]
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print_result(test_case["name"], result)
                else:
                    print(f"❌ {test_case['name']} failed: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ {test_case['name']} error: {e}")
                
    except Exception as e:
        print(f"❌ Multimodal analysis setup error: {e}")

def test_file_upload_analysis():
    """Test file upload emotion analysis"""
    print_section("FILE UPLOAD ANALYSIS")
    
    try:
        # Create a temporary test file
        test_content = "This is a test file for emotion analysis. I'm feeling very happy today!"
        
        files = {
            'file': ('test.txt', test_content, 'text/plain')
        }
        data = {
            'analysis_type': 'auto',
            'text': test_content
        }
        
        response = requests.post(
            f"{API_URL}/api/v1/emotion/analyze/file",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            print_result("Text File Upload", result)
        else:
            print(f"❌ File upload failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ File upload error: {e}")

def test_analytics_and_history():
    """Test analytics and history endpoints"""
    print_section("ANALYTICS & HISTORY")
    
    try:
        # Get analysis history
        response = requests.get(f"{API_URL}/api/v1/emotion/history?limit=5")
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis History:")
            print(f"📊 Total Analyses: {result.get('total_count', 0)}")
            
            analyses = result.get('analyses', [])
            for i, analysis in enumerate(analyses[:3]):  # Show first 3
                emotion = analysis.get('result', {}).get('dominant_emotion', 'N/A')
                timestamp = analysis.get('timestamp', 0)
                print(f"  {i+1}. {emotion} (ID: {analysis.get('id', 'N/A')[:8]}...)")
        
        # Get analytics dashboard
        response = requests.get(f"{API_URL}/api/v1/analytics/dashboard")
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Analytics Dashboard:")
            print(f"👥 Total Users: {result.get('total_users', 0)}")
            print(f"🔗 Active Sessions: {result.get('total_sessions', 0)}")
            
            stats = result.get('analysis_stats', {})
            print(f"📈 Total Analyses: {stats.get('total_analyses', 0)}")
            print(f"🤖 Models Available: {stats.get('models_available', 0)}")
            print(f"⏰ Recent (1h): {stats.get('recent_analyses', 0)}")
            
            emotion_dist = stats.get('emotion_distribution', {})
            if emotion_dist:
                print("\n📊 Emotion Distribution:")
                for emotion, count in sorted(emotion_dist.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {emotion:>10}: {count}")
        
    except Exception as e:
        print(f"❌ Analytics error: {e}")

def test_performance_benchmark():
    """Run performance benchmarks"""
    print_section("PERFORMANCE BENCHMARK")
    
    test_text = "I'm feeling really excited about this performance test!"
    
    # Test multiple requests to measure performance
    times = []
    successes = 0
    
    print("🚀 Running 10 text analysis requests...")
    
    for i in range(10):
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{API_URL}/api/v1/emotion/analyze/text",
                json={"text": test_text, "model_type": "text"}
            )
            
            end_time = time.time()
            request_time = end_time - start_time
            times.append(request_time)
            
            if response.status_code == 200:
                successes += 1
            
            print(f"  Request {i+1}: {request_time:.3f}s {'✅' if response.status_code == 200 else '❌'}")
            
        except Exception as e:
            print(f"  Request {i+1}: Error - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📊 Performance Summary:")
        print(f"  Success Rate: {successes}/10 ({successes*10}%)")
        print(f"  Average Time: {avg_time:.3f}s")
        print(f"  Min Time: {min_time:.3f}s")
        print(f"  Max Time: {max_time:.3f}s")
        print(f"  Requests/sec: {1/avg_time:.2f}")

def main():
    """Run all tests"""
    print("🧪 EmoSense Backend - Real Models Test Suite")
    print(f"🔗 Testing API at: {API_URL}")
    
    # Test authentication
    token = test_authentication()
    
    # Test health and models
    test_health_and_models()
    
    # Test emotion analysis capabilities
    test_text_emotion_analysis()
    test_audio_emotion_analysis()
    test_video_emotion_analysis()
    test_multimodal_analysis()
    test_file_upload_analysis()
    
    # Test analytics
    test_analytics_and_history()
    
    # Performance benchmark
    test_performance_benchmark()
    
    print_section("TEST COMPLETE")
    print("✅ All tests completed! Check the results above.")
    print("\n💡 Tips:")
    print("  - For production use, deploy with real ML models")
    print("  - Monitor performance and accuracy metrics")
    print("  - Consider GPU acceleration for better performance")
    print("  - Implement model caching for faster response times")

if __name__ == "__main__":
    main()
