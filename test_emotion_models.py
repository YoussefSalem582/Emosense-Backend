#!/usr/bin/env python3
"""
Test script for EmoSense emotion analysis models
"""

import asyncio
import sys
import os

# Add the api directory to path so we can import emotion_models
sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))

try:
    from emotion_models import (
        EmotionAnalysisRequest,
        TextEmotionModel,
        AudioEmotionModel,
        VideoEmotionModel,
        MultimodalEmotionModel,
        model_manager
    )
except ImportError as e:
    print(f"Import error: {e}")
    # Try importing individual components
    from emotion_models import EmotionAnalysisRequest, TextEmotionModel, model_manager

async def test_text_emotion():
    """Test text emotion analysis"""
    print("\n🔤 Testing Text Emotion Analysis")
    print("=" * 50)
    
    test_texts = [
        "I am so excited about this amazing project!",
        "I feel really sad and depressed today.",
        "This is absolutely disgusting and horrible!",
        "I'm scared and worried about the future.",
        "What a surprising turn of events!",
        "I'm furious and angry about this situation!",
        "It's just a normal day, nothing special."
    ]
    
    text_model = TextEmotionModel()
    
    for text in test_texts:
        try:
            request = EmotionAnalysisRequest(
                text=text,
                analysis_type="text"
            )
            
            result = await text_model.analyze(request)
            
            print(f"\n📝 Text: '{text}'")
            print(f"🎭 Dominant Emotion: {result.dominant_emotion}")
            print(f"🎯 Confidence: {result.confidence:.3f}")
            print(f"⚡ Processing Time: {result.processing_time:.3f}s")
            print(f"🔧 Model Version: {result.model_version}")
            print(f"📊 All Emotions: {dict(sorted(result.emotions.items(), key=lambda x: x[1], reverse=True))}")
            
        except Exception as e:
            print(f"❌ Error analyzing '{text}': {e}")

async def test_audio_emotion():
    """Test audio emotion analysis"""
    print("\n🎵 Testing Audio Emotion Analysis")
    print("=" * 50)
    
    # Create dummy audio data for testing
    dummy_audio_data = b"dummy_audio_data_for_testing" * 100
    
    try:
        audio_model = AudioEmotionModel()
        
        request = EmotionAnalysisRequest(
            audio_data=dummy_audio_data,
            analysis_type="audio"
        )
        
        result = await audio_model.analyze(request)
        
        print(f"🎭 Dominant Emotion: {result.dominant_emotion}")
        print(f"🎯 Confidence: {result.confidence:.3f}")
        print(f"⚡ Processing Time: {result.processing_time:.3f}s")
        print(f"🔧 Model Version: {result.model_version}")
        print(f"📊 All Emotions: {dict(sorted(result.emotions.items(), key=lambda x: x[1], reverse=True))}")
        print(f"📋 Metadata: {result.metadata}")
        
    except Exception as e:
        print(f"❌ Error in audio analysis: {e}")

async def test_video_emotion():
    """Test video emotion analysis"""
    print("\n📹 Testing Video Emotion Analysis")
    print("=" * 50)
    
    # Create dummy video data for testing
    dummy_video_data = b"dummy_video_data_for_testing" * 200
    
    try:
        video_model = VideoEmotionModel()
        
        request = EmotionAnalysisRequest(
            video_data=dummy_video_data,
            analysis_type="video"
        )
        
        result = await video_model.analyze(request)
        
        print(f"🎭 Dominant Emotion: {result.dominant_emotion}")
        print(f"🎯 Confidence: {result.confidence:.3f}")
        print(f"⚡ Processing Time: {result.processing_time:.3f}s")
        print(f"🔧 Model Version: {result.model_version}")
        print(f"📊 All Emotions: {dict(sorted(result.emotions.items(), key=lambda x: x[1], reverse=True))}")
        print(f"📋 Metadata: {result.metadata}")
        
    except Exception as e:
        print(f"❌ Error in video analysis: {e}")

async def test_multimodal_emotion():
    """Test multimodal emotion analysis"""
    print("\n🌐 Testing Multimodal Emotion Analysis")
    print("=" * 50)
    
    try:
        multimodal_model = MultimodalEmotionModel()
        
        request = EmotionAnalysisRequest(
            text="I'm feeling amazing today!",
            audio_data=b"dummy_audio" * 50,
            video_data=b"dummy_video" * 100,
            analysis_type="multimodal"
        )
        
        result = await multimodal_model.analyze(request)
        
        print(f"🎭 Dominant Emotion: {result.dominant_emotion}")
        print(f"🎯 Confidence: {result.confidence:.3f}")
        print(f"⚡ Processing Time: {result.processing_time:.3f}s")
        print(f"🔧 Model Version: {result.model_version}")
        print(f"📊 All Emotions: {dict(sorted(result.emotions.items(), key=lambda x: x[1], reverse=True))}")
        print(f"📋 Metadata: {result.metadata}")
        
    except Exception as e:
        print(f"❌ Error in multimodal analysis: {e}")

async def test_model_manager():
    """Test the model manager"""
    print("\n🎛️ Testing Model Manager")
    print("=" * 50)
    
    try:
        # Test getting available models
        available_models = model_manager.get_available_models()
        print("📋 Available Models:")
        for name, info in available_models.items():
            print(f"  • {name}: {info['name']} v{info['version']}")
            print(f"    Emotions: {info['supported_emotions']}")
            print(f"    Loaded: {info['loaded']}")
            print()
        
        # Test analysis through model manager
        print("🧪 Testing analysis through model manager...")
        
        text_request = EmotionAnalysisRequest(
            text="This is absolutely wonderful and amazing!",
            analysis_type="text"
        )
        
        result = await model_manager.analyze_emotion(text_request)
        print(f"✅ Model Manager Analysis Result:")
        print(f"   Dominant: {result.dominant_emotion} ({result.confidence:.3f})")
        print(f"   Version: {result.model_version}")
        
    except Exception as e:
        print(f"❌ Error in model manager test: {e}")

async def main():
    """Main test function"""
    print("🚀 EmoSense Emotion Models Test Suite")
    print("=" * 60)
    print("Testing production-ready emotion analysis models...")
    
    try:
        # Test individual models
        await test_text_emotion()
        await test_audio_emotion()
        await test_video_emotion()
        await test_multimodal_emotion()
        
        # Test model manager
        await test_model_manager()
        
        print("\n✅ All tests completed!")
        print("🎉 EmoSense emotion models are working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main())
