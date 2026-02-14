# FLUX.2 [dev] FP8 — RunPod Serverless Image Generator

Generate high-quality AI images using FLUX.2 [dev] with FP8 quantization on RunPod Serverless GPU. **Pay only for active GPU time** — GPU auto-shuts down when idle.

## ⚡ Quick Start

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed
- [RunPod account](https://www.runpod.io/) with credits
- [Docker Hub account](https://hub.docker.com/) (free)
- Python 3.10+ on your local PC
- [Hugging Face account](https://huggingface.co/) with access to FLUX.2 [dev]

### Local dependencies
```bash
pip install requests
```

---

## 🚀 Step-by-Step Deployment

### Step 1: Get Hugging Face Access Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with **Read** access
3. Go to [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) and accept the license
4. Save your token — you'll need it for the Docker build

### Step 2: Build the Docker Image

```bash
# Navigate to this project folder
cd "c:\Users\hanstanisjk18\Desktop\Flux 2.0"

# Build the Docker image (pass your HF token to download model)
docker build --build-arg HF_TOKEN=hf_YOUR_TOKEN_HERE -t flux2-dev-fp8-runpod .

# This will take 15-30 minutes (downloads ~20GB model)
```

### Step 3: Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Tag your image (replace YOUR_DOCKERHUB_USERNAME)
docker tag flux2-dev-fp8-runpod YOUR_DOCKERHUB_USERNAME/flux2-dev-fp8-runpod:latest

# Push to Docker Hub
docker push YOUR_DOCKERHUB_USERNAME/flux2-dev-fp8-runpod:latest

# This will take 10-20 minutes depending on upload speed
```

### Step 4: Create RunPod Serverless Endpoint

1. Go to [RunPod Console → Serverless](https://www.runpod.io/console/serverless)
2. Click **"+ New Endpoint"**
3. Configure:
   - **Endpoint Name**: `flux2-dev-fp8`
   - **Docker Image**: `YOUR_DOCKERHUB_USERNAME/flux2-dev-fp8-runpod:latest`
   - **GPU**: Select **RTX 4090 (24 GB)** for best value
   - **Min Workers**: `0` (zero cost when idle)
   - **Max Workers**: `1` (or more for parallel generation)
   - **Idle Timeout**: `5 minutes` (auto-shutdown after 5 min of no requests)
   - **Execution Timeout**: `300 seconds`
4. Click **Create**
5. Copy the **Endpoint ID** from the dashboard

### Step 5: Get RunPod API Key

1. Go to [RunPod Settings](https://www.runpod.io/console/user/settings)
2. Under **API Keys**, create a new key
3. Copy the API key

### Step 6: Configure the Client

Open `config.py` and fill in your credentials:

```python
RUNPOD_API_KEY = "rp_xxxxxxxxxxxxx"     # Your RunPod API key
ENDPOINT_ID = "abc123def456"             # Your endpoint ID
```

### Step 7: Create Your Prompts

Edit `prompts.json` with your image prompts:

```json
[
    {
        "prompt": "A stunning mountain landscape at golden hour, photorealistic, 8K",
        "width": 1024,
        "height": 1024,
        "num_steps": 20,
        "guidance_scale": 3.5
    },
    {
        "prompt": "A modern luxury villa with infinity pool overlooking the ocean",
        "width": 1024,
        "height": 1024,
        "num_steps": 20,
        "guidance_scale": 3.5
    }
]
```

**Simple format also works** (uses default settings):
```json
[
    "A beautiful sunset over mountains",
    "A futuristic cyberpunk city at night",
    "A cozy cabin in the woods during winter"
]
```

### Step 8: Generate Images!

```bash
# Test first (no API calls)
python client.py --dry-run

# Generate for real
python client.py

# Use a custom prompts file
python client.py my_custom_prompts.json
```

---

## 📁 Project Structure

```
Flux 2.0/
├── handler.py          # RunPod serverless handler (runs on GPU)
├── Dockerfile          # Docker image configuration
├── requirements.txt    # Python dependencies for GPU worker
├── client.py           # Local batch client (runs on your PC)
├── config.py           # Your API credentials and settings
├── prompts.json        # Your image prompts (batch input)
├── README.md           # This file
└── output/             # Downloaded images (auto-created)
    ├── image_0001_seed12345.png
    ├── image_0002_seed67890.png
    └── ...
```

## 💰 Cost Breakdown

| GPU | Cost/hr | 50 Images | 100 Images |
|---|---|---|---|
| RTX 4090 | $0.44 | ~$0.10 | ~$0.19 |
| A100 80GB | $1.64 | ~$0.37 | ~$0.73 |

**Zero cost when idle** — Min Workers = 0 means no charges between batches.

## 🔧 Configuration Options

| Setting | Default | Description |
|---|---|---|
| `DEFAULT_WIDTH` | 1024 | Image width (multiple of 8) |
| `DEFAULT_HEIGHT` | 1024 | Image height (multiple of 8) |
| `DEFAULT_STEPS` | 20 | Inference steps (20=fast, 30=quality) |
| `DEFAULT_GUIDANCE_SCALE` | 3.5 | Prompt adherence (3.5 recommended for FLUX) |
| `BATCH_CONCURRENCY` | 10 | Jobs sent per batch wave |
| `POLL_INTERVAL` | 3 | Seconds between status polls |
| `JOB_TIMEOUT` | 300 | Max seconds per image |
| `MAX_RETRIES` | 2 | Retry attempts for failed jobs |

## ❓ Troubleshooting

| Issue | Solution |
|---|---|
| Cold start takes 30-60s | Normal for first request. Model needs to load into GPU. Set Min Workers = 1 to keep warm (costs money) |
| Image generation timeout | Increase `JOB_TIMEOUT` in config.py |
| Out of VRAM | Ensure you selected 24GB+ GPU for the endpoint |
| Model download fails in Docker | Verify your HF token has access to FLUX.2 [dev] |
| API returns 401 | Check your `RUNPOD_API_KEY` in config.py |
