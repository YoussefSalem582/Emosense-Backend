# EmoSense Backend API

A comprehensive FastAPI backend for emotion analysis providing text, audio, and video emotion detection with global deployment capability.

## 🚀 Current Status: Production Ready

✅ **Railway Deployment Ready** - Optimized for global free deployment  
✅ **Localhost Testing Ready** - Flutter integration testing available  
✅ **Clean Codebase** - Unused files removed, focused structure  
✅ **ML Models Ready** - Real emotion models + lightweight alternatives  

## 🎯 Quick Start

### 🚂 **Deploy to Railway (Recommended)**
```bash
# Use the ready-to-deploy package
cd deploy_railway/
# Follow instructions in deploy_railway/README.md
```

### 🏠 **Test Locally with Flutter**  
```bash
# Start localhost server for Flutter testing
python localhost_test_server.py
# Server runs on http://localhost:8000
```

### 🔧 **Development Server**
```bash
# Full development server
python production_server.py
```

## � Project Structure (Cleaned)

```
emosense_backend/
├── 🚂 deploy_railway/          # Production deployment (Railway)
│   ├── Dockerfile             # Optimized container
│   ├── requirements_railway.txt
│   ├── production_server_railway.py
│   ├── api/emotion_models_railway.py
│   └── README.md              # Deployment guide
├── 🔧 api/                     # Core emotion analysis
│   ├── emotion_models.py      # Full emotion models
│   ├── emotion_models_lite.py # Lightweight models
│   └── index.py               # API endpoints
├── 📱 app/                     # FastAPI application
│   ├── main.py                # Application entry
│   ├── api/                   # API routes
│   ├── models/                # Database models  
│   └── services/              # Business logic
├── 🧪 tests/                   # Test suite
├── 🤖 real_models/             # Trained ML models
├── 🏠 localhost_test_server.py # Flutter testing server
├── 🚀 production_server.py     # Development server
└── 📋 requirements.txt         # Dependencies
```
│   ├── core/                  # Core functionality
│   │   ├── __init__.py
│   │   ├── security.py        # Security utilities
│   │   └── exceptions.py      # Custom exceptions
│   ├── models/                # Database models
│   │   ├── __init__.py
│   │   ├── user.py           # User model
│   │   └── emotion.py        # Emotion analysis models
│   ├── schemas/               # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── user.py           # User schemas
│   │   └── emotion.py        # Emotion analysis schemas
│   ├── services/              # Business logic
│   │   ├── __init__.py
│   │   ├── health.py         # Health check service
│   │   └── emotion/          # Emotion analysis services
│   └── utils/                 # Utility functions
├── tests/                     # Test files
├── .github/                   # GitHub configuration
│   └── copilot-instructions.md # Copilot instructions
├── requirements.txt           # Python dependencies
├── docker-compose.yml         # Docker setup
├── Dockerfile                 # Docker image
├── .env.example              # Environment variables template
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL 12+
- Redis 6+
- FFmpeg (for audio/video processing)

### Installation

1. **Clone the repository and set up virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Set up database:**
   ```bash
   # Create PostgreSQL database
   createdb emosense_db
   
   # Run migrations (when implemented)
   alembic upgrade head
   ```

5. **Run the application:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Using Docker (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

## 📚 API Documentation

Once the server is running, visit:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## 🔗 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Refresh access token
- `POST /api/v1/auth/logout` - User logout

### Emotion Analysis
- `POST /api/v1/emotion/text` - Analyze text emotions
- `POST /api/v1/emotion/video` - Analyze video emotions
- `POST /api/v1/emotion/audio` - Analyze audio emotions
- `POST /api/v1/emotion/batch` - Batch analysis
- `GET /api/v1/emotion/{analysis_id}` - Get analysis results
- `GET /api/v1/emotion/` - List user analyses

### Analytics
- `GET /api/v1/analytics/dashboard` - Analytics dashboard data
- `GET /api/v1/analytics/reports` - Generate reports
- `GET /api/v1/analytics/stats` - Usage statistics

### System
- `GET /health` - Health check
- `GET /metrics` - System metrics
- `GET /api/v1/system/info` - System information

## ⚙️ Configuration

Key environment variables (see `.env.example` for complete list):

```bash
# Application
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=your-super-secret-key

# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/emosense_db

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
```

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-mock

# Run tests
pytest tests/

# Run with coverage
pytest --cov=app tests/
```

## 🚀 Deployment

### Production Deployment

1. **Environment Setup:**
   ```bash
   # Set production environment variables
   export ENVIRONMENT=production
   export DEBUG=false
   export SECRET_KEY=your-secure-production-key
   ```

2. **Database Migration:**
   ```bash
   alembic upgrade head
   ```

3. **Start with Gunicorn:**
   ```bash
   gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

### Docker Production

```bash
# Build production image
docker build -t emosense-backend .

# Run with docker-compose (production profile)
docker-compose --profile production up -d
```

## 📊 Monitoring

- **Health Check**: `GET /health`
- **Metrics**: `GET /metrics` (Prometheus format)
- **Flower** (Celery monitoring): [http://localhost:5555](http://localhost:5555)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📝 Development Guidelines

- Follow PEP 8 style guidelines
- Use type hints throughout the codebase
- Write comprehensive tests
- Document all functions and classes
- Use async/await for I/O operations
- Implement proper error handling

## 🔒 Security

- JWT-based authentication
- Password hashing with bcrypt
- Input validation and sanitization
- Rate limiting
- CORS configuration
- Secure file upload handling

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

For support, please contact the development team or create an issue in the repository.

---

Built with ❤️ using FastAPI and modern Python technologies.
