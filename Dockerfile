# ============================================================
# FLUX.2 [dev] FP8 — RunPod Serverless Worker
# ============================================================
# Base image: RunPod PyTorch with CUDA support
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV MODEL_ID=black-forest-labs/FLUX.2-dev

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download model weights during build
# This bakes the model into the Docker image for instant cold starts
# NOTE: You need to set HF_TOKEN as a build arg if the model is gated
ARG HF_TOKEN=""
RUN if [ -n "$HF_TOKEN" ]; then \
        huggingface-cli login --token $HF_TOKEN; \
    fi

# Download model weights
RUN python -c "\
from diffusers import FluxPipeline; \
import torch; \
pipe = FluxPipeline.from_pretrained( \
    'black-forest-labs/FLUX.2-dev', \
    torch_dtype=torch.float16, \
    use_safetensors=True \
); \
print('Model downloaded successfully')" || echo "WARNING: Model download failed. Model will be downloaded at runtime."

# Copy handler script
COPY handler.py /app/handler.py

# Start the RunPod handler
CMD ["python", "-u", "/app/handler.py"]
