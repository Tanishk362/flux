"""
RunPod Serverless Handler for FLUX.2 [dev]
============================================
This handler loads the FLUX.2 [dev] model with 4-bit quantization
and generates images from text prompts via RunPod's serverless API.
"""

import runpod
import torch
import base64
import io
import time
import os
import requests as http_requests
from PIL import Image


# ============================================================
# Global model reference — loaded once, reused across requests
# ============================================================
PIPE = None
MODEL_ID = os.environ.get("MODEL_ID", "diffusers/FLUX.2-dev-bnb-4bit")
DEVICE = "cuda:0"
TORCH_DTYPE = torch.bfloat16


def remote_text_encoder(prompts):
    """
    Use the HuggingFace remote text encoder for FLUX.2.
    This avoids loading the large T5 XXL model locally, saving VRAM.
    """
    from huggingface_hub import get_token
    
    if isinstance(prompts, str):
        prompts = [prompts]
    
    response = http_requests.post(
        "https://remote-text-encoder-flux-2.huggingface.co/predict",
        json={"prompt": prompts},
        headers={
            "Authorization": f"Bearer {get_token()}",
            "Content-Type": "application/json"
        }
    )
    response.raise_for_status()
    prompt_embeds = torch.load(io.BytesIO(response.content))
    return prompt_embeds.to(DEVICE)


def load_model():
    """
    Load the FLUX.2 [dev] model with 4-bit quantization.
    This runs once when the worker starts and stays in GPU memory.
    """
    global PIPE
    
    print("=" * 60)
    print("Loading FLUX.2 [dev] 4-bit quantized model...")
    print(f"Model: {MODEL_ID}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    print("=" * 60)
    
    start_time = time.time()
    
    from diffusers import Flux2Pipeline
    
    # Load the pipeline with 4-bit quantization (no text encoder — use remote)
    PIPE = Flux2Pipeline.from_pretrained(
        MODEL_ID,
        text_encoder=None,
        torch_dtype=TORCH_DTYPE,
    ).to(DEVICE)
    
    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.1f} seconds")
    print("=" * 60)


def generate_image(prompt, width=1024, height=1024, num_steps=28, 
                   guidance_scale=4.0, seed=None):
    """
    Generate a single image from a text prompt.
    
    Args:
        prompt: Text description of the image to generate
        width: Image width in pixels (default: 1024)
        height: Image height in pixels (default: 1024)
        num_steps: Number of inference steps (default: 28)
        guidance_scale: Guidance scale for generation (default: 4.0)
        seed: Random seed for reproducibility (default: None = random)
    
    Returns:
        dict with base64 image, seed used, and generation time
    """
    global PIPE
    
    if PIPE is None:
        load_model()
    
    # Set up generator with seed
    generator = None
    if seed is not None:
        generator = torch.Generator(DEVICE).manual_seed(int(seed))
    else:
        seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(DEVICE).manual_seed(seed)
    
    print(f"Generating: '{prompt[:80]}...' | {width}x{height} | {num_steps} steps | seed: {seed}")
    
    start_time = time.time()
    
    # Get remote text embeddings
    prompt_embeds = remote_text_encoder(prompt)
    
    # Generate the image
    with torch.inference_mode():
        result = PIPE(
            prompt_embeds=prompt_embeds,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
    
    image = result.images[0]
    gen_time = time.time() - start_time
    
    # Convert to base64 PNG
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    print(f"✓ Generated in {gen_time:.1f}s | Size: {len(img_base64) // 1024} KB")
    
    return {
        "image_base64": img_base64,
        "seed": seed,
        "generation_time": round(gen_time, 2),
        "width": width,
        "height": height,
        "steps": num_steps,
    }


# ============================================================
# RunPod Handler
# ============================================================

def handler(job):
    """
    RunPod serverless handler function.
    
    Expected input format:
    {
        "input": {
            "prompt": "A beautiful sunset over mountains",
            "width": 1024,        # optional, default 1024
            "height": 1024,       # optional, default 1024
            "num_steps": 28,      # optional, default 28
            "guidance_scale": 4.0, # optional, default 4.0
            "seed": 42            # optional, default random
        }
    }
    """
    try:
        job_input = job["input"]
        
        # Validate required fields
        prompt = job_input.get("prompt")
        if not prompt:
            return {"error": "Missing required field: 'prompt'"}
        
        # Extract optional parameters with defaults
        width = int(job_input.get("width", 1024))
        height = int(job_input.get("height", 1024))
        num_steps = int(job_input.get("num_steps", 28))
        guidance_scale = float(job_input.get("guidance_scale", 4.0))
        seed = job_input.get("seed")
        
        # Validate dimensions (must be multiples of 8)
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        # Clamp values
        width = max(256, min(width, 2048))
        height = max(256, min(height, 2048))
        num_steps = max(1, min(num_steps, 50))
        guidance_scale = max(0.0, min(guidance_scale, 20.0))
        
        # Generate the image
        result = generate_image(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        
        return result
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    print("Starting FLUX.2 [dev] RunPod Worker...")
    
    # Pre-load model on startup for faster first request
    load_model()
    
    # Start the RunPod serverless worker
    runpod.serverless.start({"handler": handler})
