# Use a slim Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Set workdir
WORKDIR /app

# Install system dependencies (Tesseract + libs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libtesseract-dev \
        libzbar0 \
        libzbar-dev \
        libmagic1 \
        file \
        libgl1 \
        libglib2.0-0 \
        build-essential \
        poppler-utils \
        && rm -rf /var/lib/apt/lists/*



# Copy requirements first (for better build caching)
COPY requirements.txt /app/requirements.txt

# If you have a separate ML requirements file, copy that as well
# (uncomment the next line if you actually have this file)
# COPY requirements-ml.txt /app/requirements-ml.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
    # If you have ml requirements, also:
    # pip install -r requirements-ml.txt

# Copy the rest of the project
COPY . /app

# Expose port (FastAPI default we'll use)
EXPOSE 8000

# Default command: run uvicorn
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
