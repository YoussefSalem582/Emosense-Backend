# Railway Deployment Guide for EmoSense Backend

This directory contains the optimized deployment package for Railway's free tier.

## 🚂 Why Railway?

Railway was chosen as the best free deployment platform for EmoSense based on:

- **Always-on deployment** (no cold starts like Heroku/Render)
- **512MB RAM** on free tier (sufficient for lightweight models)
- **1GB storage** (adequate for our minimal setup)
- **AI/ML friendly** environment
- **Simple deployment** with GitHub integration
- **No credit card required** for free tier

## 📦 What's Included

```
deploy_railway/
├── Dockerfile                      # Railway-optimized container
├── requirements_railway.txt        # Minimal dependencies for free tier
├── production_server_railway.py    # Railway-optimized FastAPI server
├── api/
│   └── emotion_models_railway.py   # Lightweight emotion models
└── README.md                       # This file
```

## 🚀 Quick Deployment

### Option 1: Railway Dashboard (Recommended)

1. **Sign up** for Railway at [railway.app](https://railway.app)
2. **Connect your GitHub account**
3. **Create a new project** → "Deploy from GitHub repo"
4. **Select your repository** and the `deploy_railway` folder
5. **Railway will auto-detect** the Dockerfile and deploy

### Option 2: Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Navigate to deployment directory
cd deploy_railway

# Initialize and deploy
railway init
railway up
```

## ⚙️ Configuration

### Environment Variables (Optional)

Railway will automatically set `PORT`, but you can configure:

```env
# Optional environment variables
RAILWAY_ENVIRONMENT=production
LOG_LEVEL=INFO
```

### Automatic Settings

Railway automatically provides:
- `PORT` - Application port (usually 3000+)
- `RAILWAY_ENVIRONMENT` - Environment name
- `RAILWAY_PUBLIC_DOMAIN` - Your app's public URL

## 🔧 Technical Details

### Resource Optimization

This deployment is optimized for Railway's free tier limits:
- **Memory**: Uses <512MB RAM with lightweight models
- **Storage**: Minimal dependencies (<1GB total)
- **CPU**: Uses only rule-based and NLTK models (no PyTorch/Transformers)
- **Network**: Efficient endpoint responses

### Model Capabilities

**Text Analysis:**
- ✅ Rule-based emotion detection
- ✅ NLTK sentiment analysis
- ✅ 7 emotion categories (joy, sadness, anger, fear, surprise, disgust, neutral)

**Audio/Video Analysis:**
- 📝 Mock responses (for demo purposes)
- 🔄 Can be upgraded with real models if needed

### API Endpoints

```
GET  /                          # Welcome message
GET  /health                    # Health check
POST /analyze/text              # Text emotion analysis
POST /analyze/audio             # Audio emotion analysis (mock)
POST /analyze/video             # Video emotion analysis (mock)
POST /analyze/batch             # Batch analysis
GET  /models/status             # Model status
```

## 🧪 Testing Your Deployment

After deployment, test your API:

```bash
# Health check
curl https://your-app.railway.app/health

# Test text analysis
curl -X POST "https://your-app.railway.app/analyze/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "I am very happy today!"}'
```

## 📱 Flutter Integration

Update your Flutter app's API endpoint:

```dart
class ApiConfig {
  static const String baseUrl = 'https://your-app.railway.app';
  // Replace 'your-app' with your actual Railway domain
}
```

## 🔄 Updates and Scaling

### Auto-deployment
- Railway automatically redeploys when you push to your main branch
- Builds typically take 2-3 minutes

### Scaling Options
- **Free tier**: 512MB RAM, shared CPU
- **Paid tiers**: More RAM, dedicated CPU, custom domains
- **Add real models**: Upgrade requirements.txt to include torch/transformers

## 🛠️ Troubleshooting

### Common Issues

**Build Failures:**
```bash
# Check Railway logs
railway logs
```

**Memory Issues:**
- Ensure you're using the minimal requirements.txt
- Models automatically fallback to lighter versions

**Import Errors:**
- Dependencies are optional - the app gracefully handles missing packages
- Rule-based analysis always works as fallback

### Getting Help

1. **Railway Docs**: [docs.railway.app](https://docs.railway.app)
2. **Railway Discord**: Active community support
3. **GitHub Issues**: Report bugs in your repository

## 🎯 Next Steps

1. **Deploy to Railway** using this package
2. **Test all endpoints** with your domain
3. **Update Flutter app** with new API endpoint
4. **Monitor performance** in Railway dashboard
5. **Upgrade models** if needed for production

## 📊 Monitoring

Railway provides built-in monitoring:
- **CPU/Memory usage**
- **Request logs** 
- **Error tracking**
- **Deployment history**

Access via: [railway.app/dashboard](https://railway.app/dashboard)

---

**🎉 Your EmoSense backend is ready for global deployment on Railway!**

For support or questions, check the Railway documentation or create an issue in your repository.
