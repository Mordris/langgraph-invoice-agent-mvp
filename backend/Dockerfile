# Use an official Python runtime as a parent image
# 3.11-slim-bookworm is stable, small, and highly compatible with ML libs
FROM python:3.11-slim-bookworm

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc
# PYTHONUNBUFFERED: Ensures python output is sent straight to terminal (logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
# libpq-dev and gcc are often needed for building database drivers if wheels fail
# curl is useful for healthchecks
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port
EXPOSE 8000

# Run the application with uvicorn
# --host 0.0.0.0 allows external access (from other containers)
# --reload is ignored in production usually, but useful here if mapped
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]