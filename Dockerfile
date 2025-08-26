# Dockerfile - CPU build (Python 3.10 to keep rembg happy)
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps for OpenCV, ffmpeg for image handling, git/wget used for optional model download
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget ca-certificates ffmpeg \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Copy and install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU-only PyTorch (explicit package index for CPU wheels)
# This uses the PyTorch wheel index so pip picks CPU-only builds.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio

# Copy app source
COPY . .

# Create model + uploads dirs (persist these via Coolify volumes)
RUN mkdir -p /app/models /app/uploads

# Optional helper script for model downloads (executable)
RUN chmod +x /app/scripts/download_models.sh || true

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
