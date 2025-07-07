#!/usr/bin/env python3
"""
Test client for EmoSense API server
Tests the endpoints that will be used by the Flutter app
"""

import requests
import json
import asyncio
import time

# Server URL
BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🏥 Testing Health Endpoint")
    print("-" * 30)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Status: {data['status']}")
            print(f"📊 API Version: {data['version']}")
            print(f"🤖 Models Available:")
            for model, available in data['models_available'].items():
                status = "✅ Ready" if available else "❌ Not Ready"
                print(f"   • {model}: {status}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Health check error: {e}")

def test_text_analysis():
    """Test text emotion analysis endpoint"""
    print(f"\n📝 Testing Text Analysis Endpoint")
    print("-" * 40)
    
    test_texts = [
        "I'm so excited about this new app!",
        "This is terrible and I hate it.",
        "What a surprising announcement!",
        "I feel scared about the future.",
        "This is a normal sentence."
    ]
    
    for text in test_texts:
        try:
            payload = {
                "text": text,
                "analysis_type": "text"
            }
            
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/api/v1/analyze/text", json=payload)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                if data["success"]:
                    result = data["data"]
                    print(f"\n📱 Text: \"{text}\"")
                    print(f"🎭 Emotion: {result['dominant_emotion']} ({result['confidence']:.1%})")
                    print(f"⚡ API Response Time: {(end_time - start_time):.3f}s")
                    print(f"🔧 Model Time: {result['processing_time']}s")
                    
                    # Show top 3 emotions
                    emotions = sorted(result['emotions'].items(), key=lambda x: x[1], reverse=True)[:3]
                    emotion_str = " | ".join([f"{e}: {s:.1%}" for e, s in emotions])
                    print(f"📊 Top Emotions: {emotion_str}")
                else:
                    print(f"❌ Analysis failed: {data['error']}")
            else:
                print(f"❌ Request failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Text analysis error: {e}")

def test_models_endpoint():
    """Test models information endpoint"""
    print(f"\n🤖 Testing Models Information Endpoint")
    print("-" * 45)
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/models")
        
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                models = data["data"]["models"]
                print(f"📊 API Version: {data['data']['api_version']}")
                print(f"📈 Total Models: {data['data']['total_models']}")
                print(f"\n🔧 Model Details:")
                
                for model_type, info in models.items():
                    status = "🟢 Active" if info["loaded"] else "🟡 Available" if info["available"] else "🔴 Unavailable"
                    print(f"\n• {model_type.upper()} MODEL {status}")
                    print(f"  └── Name: {info['name']}")
                    print(f"  └── Version: {info['version']}")
                    print(f"  └── Emotions: {', '.join(info['supported_emotions'])}")
            else:
                print(f"❌ Models info failed: {data['error']}")
        else:
            print(f"❌ Request failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Models info error: {e}")

def test_batch_analysis():
    """Test batch text analysis"""
    print(f"\n📦 Testing Batch Analysis Endpoint")
    print("-" * 40)
    
    batch_texts = [
        "I love this app!",
        "This is frustrating.",
        "Amazing work!",
        "I'm worried about this.",
        "Neutral statement here."
    ]
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/v1/analyze/batch", json=batch_texts)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                results = data["data"]["results"]
                summary = data["data"]["summary"]
                
                print(f"📊 Batch Summary:")
                print(f"   • Total Texts: {summary['total_texts']}")
                print(f"   • Successful: {summary['successful']}")
                print(f"   • Failed: {summary['failed']}")
                print(f"   • Total Time: {summary['total_processing_time']}s")
                print(f"   • Average Time: {summary['average_time']}s")
                print(f"   • API Response: {(end_time - start_time):.3f}s")
                
                print(f"\n📝 Individual Results:")
                for result in results:
                    if result["success"]:
                        print(f"   {result['index'] + 1}. {result['dominant_emotion']} ({result['confidence']:.1%}) - \"{result['text']}\"")
                    else:
                        print(f"   {result['index'] + 1}. ❌ Error: {result['error']}")
            else:
                print(f"❌ Batch analysis failed: {data['error']}")
        else:
            print(f"❌ Request failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Batch analysis error: {e}")

def test_flutter_integration_example():
    """Show how Flutter app would integrate"""
    print(f"\n📱 Flutter Integration Example")
    print("-" * 35)
    
    print("🔧 Flutter HTTP Request Example:")
    print("""
// Dart code for Flutter app
import 'package:http/http.dart' as http;
import 'dart:convert';

class EmoSenseAPI {
  static const String baseUrl = 'http://localhost:8000';
  
  static Future<Map<String, dynamic>> analyzeText(String text) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/v1/analyze/text'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'text': text,
        'analysis_type': 'text'
      }),
    );
    
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      if (data['success']) {
        return data['data'];
      } else {
        throw Exception(data['error']);
      }
    } else {
      throw Exception('Failed to analyze text');
    }
  }
  
  static Future<Map<String, dynamic>> getModelsInfo() async {
    final response = await http.get(
      Uri.parse('$baseUrl/api/v1/models'),
    );
    
    return jsonDecode(response.body);
  }
}

// Usage in Flutter widget
class EmotionAnalysisWidget extends StatefulWidget {
  @override
  _EmotionAnalysisWidgetState createState() => _EmotionAnalysisWidgetState();
}

class _EmotionAnalysisWidgetState extends State<EmotionAnalysisWidget> {
  String _emotion = '';
  double _confidence = 0.0;
  bool _isAnalyzing = false;
  
  Future<void> _analyzeText(String text) async {
    setState(() {
      _isAnalyzing = true;
    });
    
    try {
      final result = await EmoSenseAPI.analyzeText(text);
      setState(() {
        _emotion = result['dominant_emotion'];
        _confidence = result['confidence'];
        _isAnalyzing = false;
      });
    } catch (e) {
      print('Error: $e');
      setState(() {
        _isAnalyzing = false;
      });
    }
  }
}
""")

def main():
    """Run all API tests"""
    print("🚀 EMOSENSE API INTEGRATION TESTS")
    print("=" * 50)
    print("Testing FastAPI server for Flutter app connection...")
    
    # Wait for server to be ready
    print("⏳ Waiting for server to initialize...")
    time.sleep(5)
    
    # Run tests
    test_health_endpoint()
    test_text_analysis()
    test_models_endpoint()
    test_batch_analysis()
    test_flutter_integration_example()
    
    print(f"\n" + "=" * 50)
    print("✅ API INTEGRATION TESTS COMPLETE!")
    print("🎉 Your Flutter app can now connect to the emotion analysis API!")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔧 API Base URL: http://localhost:8000")
    print("=" * 50)

if __name__ == "__main__":
    main()
