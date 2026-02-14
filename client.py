"""
FLUX.2 [dev] FP8 — RunPod Batch Image Generator Client
========================================================
Sends batch prompts to RunPod serverless, tracks progress,
downloads images locally, and GPU auto-shuts down when done.

Usage:
    python client.py                    # Run with prompts.json
    python client.py my_prompts.json    # Run with custom prompts file
    python client.py --dry-run          # Test without sending to API
"""

import os
import sys
import json
import time
import base64
import requests
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path

# Import configuration
try:
    from config import (
        RUNPOD_API_KEY,
        ENDPOINT_ID,
        OUTPUT_DIR,
        DEFAULT_WIDTH,
        DEFAULT_HEIGHT,
        DEFAULT_STEPS,
        DEFAULT_GUIDANCE_SCALE,
        BATCH_CONCURRENCY,
        POLL_INTERVAL,
        JOB_TIMEOUT,
        MAX_RETRIES,
    )
except ImportError:
    print("ERROR: config.py not found. Please create it from config.py and fill in your credentials.")
    sys.exit(1)


# ============================================================
# RunPod API Client
# ============================================================

class RunPodClient:
    """Client for interacting with RunPod Serverless API."""
    
    def __init__(self, api_key, endpoint_id):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    
    def submit_job(self, prompt_data):
        """Submit a single image generation job. Returns job ID."""
        payload = {"input": prompt_data}
        
        response = requests.post(
            f"{self.base_url}/run",
            headers=self.headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("id")
    
    def check_status(self, job_id):
        """Check the status of a job. Returns full response."""
        response = requests.get(
            f"{self.base_url}/status/{job_id}",
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    
    def cancel_job(self, job_id):
        """Cancel a running job."""
        try:
            response = requests.post(
                f"{self.base_url}/cancel/{job_id}",
                headers=self.headers,
                timeout=10,
            )
            return response.status_code == 200
        except Exception:
            return False


# ============================================================
# Batch Image Generator
# ============================================================

class BatchGenerator:
    """Manages batch image generation with progress tracking."""
    
    def __init__(self, client, output_dir):
        self.client = client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.jobs = {}       # job_id -> {index, prompt, status, result, retries}
        self.completed = 0
        self.failed = 0
        self.total = 0
        self.start_time = None
    
    def load_prompts(self, prompts_file):
        """Load prompts from a JSON file."""
        with open(prompts_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        
        if not isinstance(prompts, list):
            raise ValueError("prompts.json must contain a JSON array of prompt objects")
        
        # Ensure each prompt has required fields with defaults
        for i, p in enumerate(prompts):
            if isinstance(p, str):
                # Simple string prompt — wrap it
                prompts[i] = {
                    "prompt": p,
                    "width": DEFAULT_WIDTH,
                    "height": DEFAULT_HEIGHT,
                    "num_steps": DEFAULT_STEPS,
                    "guidance_scale": DEFAULT_GUIDANCE_SCALE,
                }
            else:
                # Fill in defaults for missing optional fields
                prompts[i].setdefault("width", DEFAULT_WIDTH)
                prompts[i].setdefault("height", DEFAULT_HEIGHT)
                prompts[i].setdefault("num_steps", DEFAULT_STEPS)
                prompts[i].setdefault("guidance_scale", DEFAULT_GUIDANCE_SCALE)
        
        return prompts
    
    def submit_all(self, prompts, concurrency=0):
        """Submit all prompts to RunPod. Returns list of (index, job_id) pairs."""
        self.total = len(prompts)
        self.start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"  FLUX.2 [dev] FP8 — Batch Image Generator")
        print(f"  Submitting {self.total} images to RunPod...")
        print(f"{'='*60}\n")
        
        submitted = 0
        
        for i, prompt_data in enumerate(prompts):
            try:
                job_id = self.client.submit_job(prompt_data)
                self.jobs[job_id] = {
                    "index": i,
                    "prompt": prompt_data.get("prompt", "")[:80],
                    "full_prompt": prompt_data,
                    "status": "IN_QUEUE",
                    "result": None,
                    "retries": 0,
                    "submitted_at": time.time(),
                }
                submitted += 1
                print(f"  ✓ [{submitted}/{self.total}] Submitted: {prompt_data.get('prompt', '')[:60]}...")
                
                # Throttle submissions if concurrency is set
                if concurrency > 0 and submitted % concurrency == 0 and submitted < self.total:
                    print(f"    ⏳ Batch of {concurrency} submitted, brief pause...")
                    time.sleep(1)
                    
            except Exception as e:
                print(f"  ✗ [{i+1}/{self.total}] Failed to submit: {str(e)}")
                self.failed += 1
        
        print(f"\n  ✓ {submitted} jobs submitted successfully")
        if self.failed > 0:
            print(f"  ✗ {self.failed} jobs failed to submit")
        print()
        
        return submitted
    
    def poll_and_download(self):
        """Poll all jobs until complete, download images as they finish."""
        
        pending_jobs = {jid: info for jid, info in self.jobs.items()}
        
        print(f"  Polling for results (every {POLL_INTERVAL}s)...\n")
        
        while pending_jobs:
            completed_this_round = []
            
            for job_id, info in pending_jobs.items():
                try:
                    status_response = self.client.check_status(job_id)
                    status = status_response.get("status", "UNKNOWN")
                    
                    if status == "COMPLETED":
                        # Download the image
                        output = status_response.get("output", {})
                        if isinstance(output, dict) and "error" in output:
                            print(f"  ✗ Image {info['index']+1}: Server error — {output['error']}")
                            self.failed += 1
                        else:
                            self._save_image(info["index"], output, info["prompt"])
                            self.completed += 1
                        
                        completed_this_round.append(job_id)
                        
                    elif status == "FAILED":
                        error_msg = status_response.get("error", "Unknown error")
                        
                        # Retry logic
                        if info["retries"] < MAX_RETRIES:
                            info["retries"] += 1
                            print(f"  ⟳ Image {info['index']+1}: Retrying ({info['retries']}/{MAX_RETRIES})...")
                            try:
                                new_job_id = self.client.submit_job(info["full_prompt"])
                                self.jobs[new_job_id] = info.copy()
                                self.jobs[new_job_id]["status"] = "IN_QUEUE"
                                pending_jobs[new_job_id] = self.jobs[new_job_id]
                            except Exception:
                                pass
                        else:
                            print(f"  ✗ Image {info['index']+1}: FAILED — {error_msg}")
                            self.failed += 1
                        
                        completed_this_round.append(job_id)
                        
                    elif status in ("IN_QUEUE", "IN_PROGRESS"):
                        # Check for timeout
                        elapsed = time.time() - info["submitted_at"]
                        if elapsed > JOB_TIMEOUT:
                            print(f"  ✗ Image {info['index']+1}: TIMEOUT after {JOB_TIMEOUT}s")
                            self.client.cancel_job(job_id)
                            self.failed += 1
                            completed_this_round.append(job_id)
                    
                except requests.exceptions.RequestException as e:
                    # Network error — will retry on next poll
                    pass
            
            # Remove completed jobs from pending
            for job_id in completed_this_round:
                pending_jobs.pop(job_id, None)
            
            if pending_jobs:
                # Print progress
                total_done = self.completed + self.failed
                elapsed = time.time() - self.start_time
                eta = ""
                if self.completed > 0:
                    avg_time = elapsed / self.completed
                    remaining = (self.total - total_done) * avg_time
                    eta = f" | ETA: {timedelta(seconds=int(remaining))}"
                
                self._print_progress(total_done, elapsed, eta)
                time.sleep(POLL_INTERVAL)
        
        return self.completed, self.failed
    
    def _save_image(self, index, output, prompt_preview):
        """Save a base64 image to the output directory."""
        try:
            # Handle different output formats
            if isinstance(output, dict):
                img_base64 = output.get("image_base64", "")
                seed = output.get("seed", "unknown")
                gen_time = output.get("generation_time", 0)
            else:
                img_base64 = str(output)
                seed = "unknown"
                gen_time = 0
            
            if not img_base64:
                print(f"  ✗ Image {index+1}: Empty image data")
                self.failed += 1
                self.completed -= 1
                return
            
            # Decode and save
            img_data = base64.b64decode(img_base64)
            filename = f"image_{index+1:04d}_seed{seed}.png"
            filepath = self.output_dir / filename
            
            with open(filepath, "wb") as f:
                f.write(img_data)
            
            size_kb = len(img_data) / 1024
            print(f"  ✓ Image {index+1}/{self.total}: Saved → {filename} ({size_kb:.0f} KB, {gen_time}s)")
            
        except Exception as e:
            print(f"  ✗ Image {index+1}: Save error — {str(e)}")
            self.failed += 1
            self.completed -= 1
    
    def _print_progress(self, total_done, elapsed, eta):
        """Print a progress bar."""
        pct = (total_done / self.total) * 100 if self.total > 0 else 0
        bar_len = 30
        filled = int(bar_len * total_done / self.total) if self.total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        print(f"  [{bar}] {pct:.0f}% | {total_done}/{self.total} | "
              f"✓{self.completed} ✗{self.failed} | {elapsed_str}{eta}")
    
    def print_summary(self):
        """Print final summary of the batch run."""
        elapsed = time.time() - self.start_time
        
        print(f"\n{'='*60}")
        print(f"  BATCH COMPLETE")
        print(f"{'='*60}")
        print(f"  Total images:     {self.total}")
        print(f"  ✓ Successful:     {self.completed}")
        print(f"  ✗ Failed:         {self.failed}")
        print(f"  Total time:       {timedelta(seconds=int(elapsed))}")
        
        if self.completed > 0:
            avg_time = elapsed / self.completed
            print(f"  Avg per image:    {avg_time:.1f} seconds")
        
        # Cost estimate (RTX 4090 @ $0.44/hr)
        gpu_hours = elapsed / 3600
        cost_4090 = gpu_hours * 0.44
        cost_a100 = gpu_hours * 1.64
        print(f"\n  💰 Estimated cost:")
        print(f"     RTX 4090:  ${cost_4090:.4f}")
        print(f"     A100 80GB: ${cost_a100:.4f}")
        
        print(f"\n  📁 Images saved to: {self.output_dir.resolve()}")
        print(f"  🔌 GPU will auto-shutdown after idle timeout")
        print(f"{'='*60}\n")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    # Parse arguments
    dry_run = "--dry-run" in sys.argv
    prompts_file = "prompts.json"
    
    for arg in sys.argv[1:]:
        if arg.endswith(".json"):
            prompts_file = arg
        elif arg == "--dry-run":
            continue
        elif arg in ("--help", "-h"):
            print(__doc__)
            sys.exit(0)
    
    # Validate config
    if not dry_run:
        if RUNPOD_API_KEY == "YOUR_RUNPOD_API_KEY_HERE":
            print("ERROR: Please set your RUNPOD_API_KEY in config.py")
            print("  Get your key at: https://www.runpod.io/console/user/settings")
            sys.exit(1)
        
        if ENDPOINT_ID == "YOUR_ENDPOINT_ID_HERE":
            print("ERROR: Please set your ENDPOINT_ID in config.py")
            print("  Create an endpoint at: https://www.runpod.io/console/serverless")
            sys.exit(1)
    
    # Initialize
    client = RunPodClient(RUNPOD_API_KEY, ENDPOINT_ID)
    generator = BatchGenerator(client, OUTPUT_DIR)
    
    # Load prompts
    try:
        prompts = generator.load_prompts(prompts_file)
        print(f"\n  📋 Loaded {len(prompts)} prompts from {prompts_file}")
    except FileNotFoundError:
        print(f"ERROR: Prompts file not found: {prompts_file}")
        print("  Create a prompts.json file or specify a different file:")
        print("  python client.py my_prompts.json")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {prompts_file}: {e}")
        sys.exit(1)
    
    # Dry run mode
    if dry_run:
        print(f"\n{'='*60}")
        print(f"  DRY RUN — No API calls will be made")
        print(f"{'='*60}")
        print(f"\n  Would submit {len(prompts)} images to:")
        print(f"    Endpoint: {ENDPOINT_ID}")
        print(f"    Output:   {Path(OUTPUT_DIR).resolve()}")
        print(f"\n  Prompts preview:")
        for i, p in enumerate(prompts[:5]):
            print(f"    {i+1}. {p.get('prompt', '')[:70]}...")
        if len(prompts) > 5:
            print(f"    ... and {len(prompts) - 5} more")
        
        # Time estimate
        est_time = len(prompts) * 15  # ~15s per image
        est_cost = (est_time / 3600) * 0.44
        print(f"\n  ⏱ Estimated time: {timedelta(seconds=est_time)}")
        print(f"  💰 Estimated cost (RTX 4090): ${est_cost:.4f}")
        print(f"\n  Run without --dry-run to start generation.")
        return
    
    # Confirm with user
    est_time = len(prompts) * 15
    est_cost = (est_time / 3600) * 0.44
    print(f"\n  ⏱ Estimated time: ~{timedelta(seconds=est_time)}")
    print(f"  💰 Estimated cost (RTX 4090): ~${est_cost:.4f}")
    
    confirm = input(f"\n  Start generating {len(prompts)} images? [Y/n]: ").strip().lower()
    if confirm in ("n", "no"):
        print("  Cancelled.")
        return
    
    # Submit all jobs
    submitted = generator.submit_all(prompts, BATCH_CONCURRENCY)
    
    if submitted == 0:
        print("  No jobs were submitted. Check your configuration.")
        return
    
    # Poll and download
    try:
        completed, failed = generator.poll_and_download()
    except KeyboardInterrupt:
        print("\n\n  ⚠ Interrupted! Already downloaded images are saved.")
        print("  Note: Running jobs on RunPod may still complete and incur charges.")
    
    # Print summary
    generator.print_summary()


if __name__ == "__main__":
    main()
