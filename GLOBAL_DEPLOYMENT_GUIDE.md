# 🌍 EmoSense Global Deployment Guide

Deploy your emotion analysis API globally for worldwide access!

## 🚀 Quick Deploy Options (Choose One)

### Option 1: Railway (Recommended - Fastest)
```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login to Railway
railway login

# 3. Initialize project
railway init

# 4. Deploy
railway up
```

### Option 2: Render (Free Tier Available)
```bash
# 1. Push code to GitHub
git init
git add .
git commit -m "Initial commit"
git remote add origin YOUR_GITHUB_REPO
git push -u origin main

# 2. Go to render.com
# 3. Connect GitHub repo
# 4. Deploy as Web Service
```

### Option 3: Google Cloud Run
```bash
# 1. Install Google Cloud CLI
# 2. Login and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 3. Build and deploy
gcloud run deploy emosense-api --source . --platform managed --region us-central1
```

### Option 4: Heroku
```bash
# 1. Install Heroku CLI
# 2. Login
heroku login

# 3. Create app
heroku create emosense-api

# 4. Deploy
git push heroku main
```

---

## 📋 Step-by-Step: Railway Deployment (Recommended)

### Step 1: Prepare Your Code
```bash
# Make sure all files are ready
ls -la
# Should see: production_server.py, requirements_production.txt, Procfile, railway.toml
```

### Step 2: Install Railway CLI
```bash
# Windows (PowerShell)
iwr https://railway.app/install.ps1 | iex

# macOS/Linux
curl -fsSL https://railway.app/install.sh | sh

# Or via npm
npm install -g @railway/cli
```

### Step 3: Deploy to Railway
```bash
# Login to Railway
railway login

# Initialize project in current directory
railway init

# Deploy your API
railway up

# Get your deployed URL
railway status
```

### Step 4: Set Environment Variables (Optional)
```bash
# Set production environment
railway variables set ENVIRONMENT=production
railway variables set EMOTION_MODEL_DEVICE=cpu
```

---

## 📋 Step-by-Step: Render Deployment (Free Option)

### Step 1: Push to GitHub
```bash
# Initialize git repo
git init

# Add all files
git add .

# Commit
git commit -m "EmoSense API for global deployment"

# Add remote (replace with your repo)
git remote add origin https://github.com/yourusername/emosense-api.git

# Push
git push -u origin main
```

### Step 2: Deploy on Render
1. Go to [render.com](https://render.com)
2. Sign up/Login with GitHub
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Name**: `emosense-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements_production.txt`
   - **Start Command**: `gunicorn production_server:app --host 0.0.0.0 --port $PORT --worker-class uvicorn.workers.UvicornWorker`
6. Click "Create Web Service"

---

## 🔧 Configuration Files Created

### `production_server.py`
- Production-optimized FastAPI server
- Global CORS enabled
- Environment variable support
- Robust error handling
- Health monitoring

### `requirements_production.txt`
- Production dependencies
- Optimized for cloud deployment
- Fallback support for missing libraries

### `Procfile`
- Heroku/Railway deployment configuration
- Gunicorn WSGI server setup

### `railway.toml`
- Railway-specific configuration
- Build and deployment settings

---

## 🌍 After Deployment

### Your API will be available globally at:
- **Railway**: `https://your-app-name.railway.app`
- **Render**: `https://your-app-name.onrender.com`
- **Google Cloud Run**: `https://emosense-api-[hash]-uc.a.run.app`
- **Heroku**: `https://your-app-name.herokuapp.com`

### Test Your Global API:
```bash
# Health check
curl https://your-deployed-url.com/health

# Text analysis
curl -X POST "https://your-deployed-url.com/api/v1/analyze/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so excited about this global deployment!"}'
```

---

## 📱 Update Flutter App for Global Access

### Update the base URL in your Flutter app:
```dart
// lib/services/emosense_api.dart
class EmoSenseAPI {
  // Replace with your deployed URL
  static const String _baseUrl = 'https://your-app-name.railway.app';
  
  // Rest of your code remains the same...
}
```

---

## 🔒 Production Security (Optional)

### Add API Key Authentication:
```python
# In production_server.py, add:
from fastapi import Header

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("API_KEY", "your-secret-key"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# Then add to endpoints:
@app.post("/api/v1/analyze/text")
async def analyze_text_emotion(
    request: TextAnalysisRequest,
    api_key: str = Depends(verify_api_key)
):
    # Your existing code...
```

---

## 📊 Monitoring Your Global API

### Built-in Endpoints:
- **Health**: `GET /health` - Check API status
- **Models**: `GET /api/v1/models` - Check model availability
- **Docs**: `GET /docs` - API documentation (dev only)

### Platform Monitoring:
- **Railway**: Built-in monitoring dashboard
- **Render**: Automatic health checks and metrics
- **Google Cloud Run**: Cloud Monitoring integration

---

## 🎉 Success!

Once deployed, your EmoSense emotion analysis API will be:

✅ **Globally Accessible** - Available worldwide  
✅ **Production Ready** - Optimized for cloud deployment  
✅ **Scalable** - Auto-scaling based on demand  
✅ **Reliable** - Built-in health monitoring  
✅ **Fast** - Edge deployment for low latency  

Your Flutter app can now connect to your global API from anywhere in the world!

---

## 🚀 Next Steps

1. **Choose a deployment platform** (Railway recommended)
2. **Follow the deployment steps** above
3. **Update your Flutter app** with the new global URL
4. **Test the global connection**
5. **Share your app** with users worldwide!

Your emotion analysis API is now ready for global production use! 🌍✨
