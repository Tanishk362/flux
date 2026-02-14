# ============================================================
# FLUX.2 [dev] — RunPod Serverless Worker
# ============================================================
# Base image: RunPod PyTorch with CUDA support
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV MODEL_ID=diffusers/FLUX.2-dev-bnb-4bit

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
ARG HF_TOKEN=""
RUN if [ -n "$HF_TOKEN" ]; then \
    huggingface-cli login --token $HF_TOKEN; \
    fi

# Download the quantized FLUX.2-dev model weights (4-bit, ~10GB)
RUN python -c "\
    from huggingface_hub import snapshot_download; \
    snapshot_download('diffusers/FLUX.2-dev-bnb-4bit'); \
    print('FLUX.2-dev-bnb-4bit model downloaded successfully')"

# Also download the base model config for Flux2Pipeline compatibility
RUN python -c "\
    from huggingface_hub import snapshot_download; \
    snapshot_download('black-forest-labs/FLUX.2-dev', allow_patterns=['*.json', '*.txt', 'tokenizer*']); \
    print('FLUX.2-dev config files downloaded successfully')"

# Copy handler script
COPY handler.py /app/handler.py

# Start the RunPod handler
CMD ["python", "-u", "/app/handler.py"]
