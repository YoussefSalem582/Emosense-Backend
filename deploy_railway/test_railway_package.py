"""
Local test script for Railway deployment package
Run this to verify the deployment works before pushing to Railway
"""

import asyncio
import sys
import os
import time
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def test_models():
    """Test that the Railway models can be loaded and used"""
    print("🧪 Testing Railway deployment package...")
    
    try:
        # Test model imports
        print("📦 Testing model imports...")
        from api.emotion_models_railway import (
            EmotionAnalysisRequest,
            EmotionResult,
            RailwayTextEmotionModel,
            MockAudioEmotionModel,
            MockVideoEmotionModel,
            initialize_models
        )
        print("✅ Models imported successfully")
        
        # Test model initialization
        print("🔧 Testing model initialization...")
        success = await initialize_models()
        if success:
            print("✅ Models initialized successfully")
        else:
            print("❌ Model initialization failed")
            return False
        
        # Test text emotion analysis
        print("📝 Testing text emotion analysis...")
        from api.emotion_models_railway import railway_text_model
        
        test_texts = [
            "I am very happy today!",
            "This is terrible and makes me angry",
            "I'm scared about the upcoming exam",
            "What a wonderful surprise!",
            "This is just a normal day"
        ]
        
        for text in test_texts:
            result = await railway_text_model.analyze_emotion(text)
            print(f"   Text: '{text}'")
            print(f"   Emotion: {result.emotion} ({result.confidence:.2f})")
            print(f"   Model: {result.model_used}")
            print(f"   Time: {result.processing_time:.3f}s")
            print()
        
        # Test audio mock
        print("🎵 Testing mock audio analysis...")
        from api.emotion_models_railway import mock_audio_model
        fake_audio = b"fake audio data" * 1000
        audio_result = await mock_audio_model.analyze_emotion(fake_audio)
        print(f"   Audio emotion: {audio_result.emotion} ({audio_result.confidence:.2f})")
        print(f"   Model: {audio_result.model_used}")
        
        # Test video mock
        print("🎬 Testing mock video analysis...")
        from api.emotion_models_railway import mock_video_model
        fake_video = b"fake video data" * 5000
        video_result = await mock_video_model.analyze_emotion(fake_video)
        print(f"   Video emotion: {video_result.emotion} ({video_result.confidence:.2f})")
        print(f"   Model: {video_result.model_used}")
        
        print("🎯 All tests passed! Railway package is ready for deployment.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_server():
    """Test that the FastAPI server can start"""
    print("🚀 Testing FastAPI server startup...")
    
    try:
        # Import server
        import production_server_railway
        print("✅ Server module imported successfully")
        
        # Test that app is created
        if hasattr(production_server_railway, 'app'):
            print("✅ FastAPI app created successfully")
        else:
            print("❌ FastAPI app not found")
            return False
        
        print("🎯 Server test passed! Railway server is ready.")
        return True
        
    except Exception as e:
        print(f"❌ Server test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("=" * 60)
    print("🚂 RAILWAY DEPLOYMENT PACKAGE TEST")
    print("=" * 60)
    
    start_time = time.time()
    
    # Test models
    models_ok = await test_models()
    
    print("\n" + "-" * 40 + "\n")
    
    # Test server
    server_ok = await test_server()
    
    print("\n" + "=" * 60)
    
    total_time = time.time() - start_time
    
    if models_ok and server_ok:
        print("🎉 ALL TESTS PASSED!")
        print(f"⏱️  Total test time: {total_time:.2f}s")
        print("\n✅ Your Railway deployment package is ready!")
        print("📋 Next steps:")
        print("   1. Push this code to your GitHub repository")
        print("   2. Connect to Railway at https://railway.app")
        print("   3. Deploy from the deploy_railway folder")
        print("   4. Test your live API endpoints")
    else:
        print("❌ SOME TESTS FAILED!")
        print("🔧 Please fix the issues before deploying to Railway")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
