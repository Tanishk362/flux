"""
Configuration for FLUX.2 [dev] FP8 RunPod Client
==================================================
Loads credentials from .env file automatically.
"""

import os

# Load .env file manually (no extra dependency needed)
def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

_load_env()

# ============================================================
# RunPod API Configuration
# ============================================================

# Loaded from .env file (runpod_api_key=your_key)
RUNPOD_API_KEY = os.environ.get("runpod_api_key", "YOUR_RUNPOD_API_KEY_HERE")

# Your serverless endpoint ID (created after deploying the Docker image)
# Found at: https://www.runpod.io/console/serverless → your endpoint
ENDPOINT_ID = "YOUR_ENDPOINT_ID_HERE"

# ============================================================
# Output Settings
# ============================================================

# Local folder where generated images will be saved
OUTPUT_DIR = "./output"

# ============================================================
# Default Image Generation Settings
# ============================================================

# Image dimensions (must be multiples of 8)
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024

# Number of inference steps (more = better quality, slower)
# Recommended: 20 for good quality, 28-30 for best quality
DEFAULT_STEPS = 20

# Guidance scale (how closely to follow the prompt)
# Recommended: 3.5 for FLUX.2
DEFAULT_GUIDANCE_SCALE = 3.5

# ============================================================
# Client Settings
# ============================================================

# How many images to send concurrently (RunPod processes them in parallel)
# Set to 0 to send all at once
BATCH_CONCURRENCY = 10

# Seconds between polling for results
POLL_INTERVAL = 3

# Maximum time to wait for a single image (seconds)
JOB_TIMEOUT = 300

# Number of retry attempts for failed jobs
MAX_RETRIES = 2
