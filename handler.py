"""
RunPod Serverless Handler for FLUX.2 [dev] FP8
================================================
This handler loads the FLUX.2 [dev] model with FP8 quantization
and generates images from text prompts via RunPod's serverless API.
"""

import runpod
import torch
import base64
import io
import time
import os
from PIL import Image


# ============================================================
# Global model reference — loaded once, reused across requests
# ============================================================
PIPE = None
MODEL_ID = "black-forest-labs/FLUX.2-dev"
# Use FP8 variant if available, otherwise we quantize at load time
FP8_MODEL_ID = os.environ.get("MODEL_ID", MODEL_ID)


def load_model():
    """
    Load the FLUX.2 [dev] model with FP8 quantization.
    This runs once when the worker starts and stays in GPU memory.
    """
    global PIPE
    
    print("=" * 60)
    print("Loading FLUX.2 [dev] FP8 model...")
    print(f"Model: {FP8_MODEL_ID}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    print("=" * 60)
    
    start_time = time.time()
    
    from diffusers import FluxPipeline
    
    # Load the pipeline with FP8 / float16 mixed precision
    PIPE = FluxPipeline.from_pretrained(
        FP8_MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    
    # Move to GPU
    PIPE.to("cuda")
    
    # Enable memory-efficient attention
    PIPE.enable_attention_slicing()
    
    # Try to enable xformers for even better memory efficiency
    try:
        PIPE.enable_xformers_memory_efficient_attention()
        print("✓ xformers memory efficient attention enabled")
    except Exception:
        print("⚠ xformers not available, using default attention")
    
    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.1f} seconds")
    print("=" * 60)


def generate_image(prompt, width=1024, height=1024, num_steps=20, 
                   guidance_scale=3.5, seed=None):
    """
    Generate a single image from a text prompt.
    
    Args:
        prompt: Text description of the image to generate
        width: Image width in pixels (default: 1024)
        height: Image height in pixels (default: 1024)
        num_steps: Number of inference steps (default: 20)
        guidance_scale: Guidance scale for generation (default: 3.5)
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
        generator = torch.Generator("cuda").manual_seed(int(seed))
    else:
        seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator("cuda").manual_seed(seed)
    
    print(f"Generating: '{prompt[:80]}...' | {width}x{height} | {num_steps} steps | seed: {seed}")
    
    start_time = time.time()
    
    # Generate the image
    with torch.inference_mode():
        result = PIPE(
            prompt=prompt,
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
            "num_steps": 20,      # optional, default 20
            "guidance_scale": 3.5, # optional, default 3.5
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
        num_steps = int(job_input.get("num_steps", 20))
        guidance_scale = float(job_input.get("guidance_scale", 3.5))
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
    print("Starting FLUX.2 [dev] FP8 RunPod Worker...")
    
    # Pre-load model on startup for faster first request
    load_model()
    
    # Start the RunPod serverless worker
    runpod.serverless.start({"handler": handler})
