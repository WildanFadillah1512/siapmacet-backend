FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements FIRST to leverage cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Environment variables (overriden by docker-compose)
ENV PORT=8000
ENV LOG_LEVEL=info

# Run with Gunicorn using Uvicorn workers
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT
