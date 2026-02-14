# ============================================================
# FLUX.2 [dev] — RunPod Serverless Worker
# ============================================================
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Set HF token for gated model access at runtime
ARG HF_TOKEN=""
ENV HF_TOKEN=$HF_TOKEN

# Copy handler script
COPY handler.py /app/handler.py

# Model will be downloaded on first startup (cached in container disk)
# This avoids build failures and keeps the image small (~8GB vs ~30GB)

CMD ["python", "-u", "/app/handler.py"]
