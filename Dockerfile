# ============================================================
# RAG Q&A System — Dockerfile
# Multi-stage approach: single image running both services
# For production, split into separate backend/frontend images
# ============================================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data faiss_index logs

# Expose ports
EXPOSE 8000 8501

# Default command: run the FastAPI backend
# Override in docker-compose for Streamlit
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
