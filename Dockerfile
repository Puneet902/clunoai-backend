FROM python:3.10-slim

WORKDIR /app

# Install system audio and build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libportaudio2 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

# Start Uvicorn backend server
CMD ["bash", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
