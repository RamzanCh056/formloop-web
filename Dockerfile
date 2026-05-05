# syntax=docker/dockerfile:1.4
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app
ARG CACHE_BUST=20260505_1935

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    ultralytics \
    requests \
    opencv-python-headless \
    pillow \
    imageio \
    firebase-admin \
    runpod \
    huggingface_hub \
    scipy \
    einops \
    kornia \
    timm && \
    python3 -c "print('cache bust:', '${CACHE_BUST}')"

COPY process_video_pro.py .
COPY handler.py .

# Keep build lightweight/stable for RunPod builder.
# Models download on first worker run and are cached per worker instance.

CMD ["python", "-u", "handler.py"]
