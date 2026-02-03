FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_DISABLE_CUDA=1 \
    CUDA_VISIBLE_DEVICES=""

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app /app

# Install PyTorch CPU wheels explicitly
RUN pip install --upgrade pip && \
    pip install torch==2.10.0 torchaudio==2.10.0 \
      --index-url https://download.pytorch.org/whl/cpu

# Copy Pocket-TTS source
# Install Pocket-TTS from local path
RUN pip install -e /app/pocket-tts




RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code


EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "8", "--timeout-keep-alive", "3000"]
