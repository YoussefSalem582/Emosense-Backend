# Railway-optimized Dockerfile for EmoSense Backend Free Tier
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000

# Install minimal system dependencies for Railway free tier
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements_railway.txt requirements.txt

# Install Python dependencies (minimal for free tier)
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (lightweight)
RUN python -c "import nltk; nltk.download('vader_lexicon', quiet=True)" || echo "NLTK download skipped"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /tmp/uploads

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Expose port (Railway will set PORT automatically)
EXPOSE $PORT

# Run the application
CMD ["python", "production_server_railway.py"]
