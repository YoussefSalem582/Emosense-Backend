#!/usr/bin/env python3
"""
Simple test script for EmoSense text emotion analysis
"""

import asyncio
import sys
import os

# Add the api directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

async def test_text_emotion():
    """Test text emotion analysis"""
    print("🔤 Testing Text Emotion Analysis")
    print("=" * 50)
    
    try:
        # Import here to handle potential import issues
        from emotion_models import EmotionAnalysisRequest, TextEmotionModel
        
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
                
                # Show top 3 emotions
                sorted_emotions = sorted(result.emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"📊 Top Emotions: {dict(sorted_emotions)}")
                
            except Exception as e:
                print(f"❌ Error analyzing '{text}': {e}")
                import traceback
                traceback.print_exc()
        
        print("\n✅ Text emotion analysis test completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the correct directory and the emotion_models.py file exists.")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main test function"""
    print("🚀 EmoSense Simple Text Emotion Test")
    print("=" * 60)
    
    await test_text_emotion()

if __name__ == "__main__":
    asyncio.run(main())
