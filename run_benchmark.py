#!/usr/bin/env python3
"""Entry point for the lukepiette/ai-bench container.

This script is designed to be executed inside a RunPod pod.  It reads the
BENCHMARK_TYPE, AI_MODEL_REPO, and either LLM_PARAMS_JSON or
IMAGE_PARAMS_JSON environment variables (exactly one of the two JSON envs must
be present).  Based on the benchmark type it then runs the appropriate
benchmark routine.

For now this file contains *placeholder* benchmark functions that simply sleep
for a short period and return synthetic numbers.  They establish the
end-to-end wiring so we can iterate quickly.

Later we will replace these placeholders with real calls into the existing
benchmark code under `Nvidia/` (LLM) and `image-gen/` (image-gen).
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import subprocess
import tempfile
from pathlib import Path
from datetime import datetime # For created_at

from dotenv import load_dotenv
from supabase import create_client, Client
import requests

# --------------------------------------------------------------------------------------
# Helper: structured error with clear message for easier debugging inside the container
# --------------------------------------------------------------------------------------

def fatal(message: str) -> None:  # noqa: D401 – short helper
    """Print *message* to stderr and exit with non-zero status."""
    print(f"[benchmarkly] ERROR: {message}", file=sys.stderr, flush=True)
    sys.exit(1)


# --------------------------------------------------------------------------------------
# LLM benchmark – calls NVIDIA vLLM throughput script via subprocess
# --------------------------------------------------------------------------------------

def get_gpu_model() -> str:
    """Attempt to get the GPU model using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        # If multiple GPUs, this might return multiple lines. For now, take the first.
        return result.stdout.strip().split('\n')[0] or "UNKNOWN_GPU"
    except Exception: # noqa E722
        # print(f"[benchmarkly] Could not get GPU model: {e}", file=sys.stderr) # Optional: log if needed
        return "UNKNOWN_GPU"

def push_to_supabase(output_data: Dict[str, Any], supabase_client: Client, original_llm_params_for_batch_sizes: Dict[str, Any] | None = None) -> None:
    print("[benchmarkly] Attempting to push results to Supabase...", flush=True)
    try:
        benchmark_type = output_data.get("benchmark_type")
        model_repo = output_data.get("model_repo")
        runpod_pod_id = os.getenv("RUNPOD_POD_ID")

        current_gpu_model = get_gpu_model()
        gpu_models_array = [current_gpu_model]

        benchmark_insert_data = {
            "status": "pending",
            "gpu_models": gpu_models_array,
            "benchmark_type": benchmark_type,
            "ai_model_repo": model_repo,
            "runpod_pod_id": runpod_pod_id,
            # created_at is handled by Supabase default
        }
        print(f"[benchmarkly] DEBUG: Inserting into benchmarks: {benchmark_insert_data}", flush=True)
        benchmark_res = supabase_client.table("benchmarks").insert(benchmark_insert_data).execute()
        
        if not benchmark_res.data or not benchmark_res.data[0].get("id"):
            print("[benchmarkly] ERROR: Failed to insert into benchmarks table or no ID returned.", file=sys.stderr, flush=True)
            error_detail = getattr(benchmark_res, 'error', None)
            if error_detail:
                print(f"[benchmarkly] Supabase error: {error_detail.message}", file=sys.stderr, flush=True)
            return

        benchmark_id = benchmark_res.data[0]["id"]
        print(f"[benchmarkly] DEBUG: Created benchmark entry with ID: {benchmark_id}", flush=True)

        param_id = None # To be used by results table

        if benchmark_type == "LLM":
            if original_llm_params_for_batch_sizes is None:
                 print("[benchmarkly] ERROR: Original LLM parameters (for batchSizes) not provided for Supabase push.", file=sys.stderr, flush=True)
                 supabase_client.table("benchmarks").update({"status": "failed", "error_message": "Missing LLM params for Supabase push"}).eq("id", benchmark_id).execute()
                 return

            llm_params_data = {
                "benchmark_id": benchmark_id,
                "tokens_in": output_data.get("tokens_in"),
                "tokens_out": output_data.get("tokens_out"),
                "batch_sizes": original_llm_params_for_batch_sizes.get("batchSizes", []),
            }
            print(f"[benchmarkly] DEBUG: Inserting into llm_benchmark_params: {llm_params_data}", flush=True)
            param_res = supabase_client.table("llm_benchmark_params").insert(llm_params_data).execute()

            if not param_res.data or not param_res.data[0].get("id"):
                print("[benchmarkly] ERROR: Failed to insert into llm_benchmark_params table or no ID returned.", file=sys.stderr, flush=True)
                error_detail = getattr(param_res, 'error', None)
                if error_detail:
                    print(f"[benchmarkly] Supabase error: {error_detail.message}", file=sys.stderr, flush=True)
                supabase_client.table("benchmarks").update({"status": "failed", "error_message": "Failed to insert LLM params"}).eq("id", benchmark_id).execute()
                return
            param_id = param_res.data[0]["id"]
            print(f"[benchmarkly] DEBUG: Created LLM params entry with ID: {param_id}", flush=True)

            llm_results_to_insert = []
            for result_item in output_data.get("results", []):
                llm_results_to_insert.append({
                    "param_id": param_id,
                    "gpu_model": current_gpu_model,
                    "batch_size": result_item.get("batch_size"),
                    "output_throughput": result_item.get("output_throughput"),
                    "tokens_per_second": result_item.get("tokens_per_second"),
                })
            
            if llm_results_to_insert:
                print(f"[benchmarkly] DEBUG: Inserting {len(llm_results_to_insert)} rows into llm_benchmark_results", flush=True)
                results_res = supabase_client.table("llm_benchmark_results").insert(llm_results_to_insert).execute()
                if not results_res.data:
                    print("[benchmarkly] ERROR: Failed to insert into llm_benchmark_results table.", file=sys.stderr, flush=True)
                    error_detail = getattr(results_res, 'error', None)
                    if error_detail:
                        print(f"[benchmarkly] Supabase error: {error_detail.message}", file=sys.stderr, flush=True)
                    supabase_client.table("benchmarks").update({"status": "failed", "error_message": "Failed to insert LLM results"}).eq("id", benchmark_id).execute()
                    return
            print(f"[benchmarkly] DEBUG: Successfully inserted LLM results.", flush=True)

        elif benchmark_type == "Image":
            image_params_data = {
                "benchmark_id": benchmark_id,
                "step_count": output_data.get("step_count"),
                "resolution_width": output_data.get("resolution_width"),
                "resolution_height": output_data.get("resolution_height"),
            }
            print(f"[benchmarkly] DEBUG: Inserting into image_benchmark_params: {image_params_data}", flush=True)
            param_res = supabase_client.table("image_benchmark_params").insert(image_params_data).execute()

            if not param_res.data or not param_res.data[0].get("id"):
                print("[benchmarkly] ERROR: Failed to insert into image_benchmark_params table.", file=sys.stderr, flush=True)
                error_detail = getattr(param_res, 'error', None)
                if error_detail:
                    print(f"[benchmarkly] Supabase error: {error_detail.message}", file=sys.stderr, flush=True)
                supabase_client.table("benchmarks").update({"status": "failed", "error_message": "Failed to insert Image params"}).eq("id", benchmark_id).execute()
                return
            param_id = param_res.data[0]["id"]
            print(f"[benchmarkly] DEBUG: Created Image params entry with ID: {param_id}", flush=True)

            image_result_item = output_data.get("results", {})
            if image_result_item:
                image_result_to_insert = {
                    "param_id": param_id,
                    "gpu_model": current_gpu_model,
                    "images_per_second": image_result_item.get("images_per_second"),
                }
                print(f"[benchmarkly] DEBUG: Inserting into image_benchmark_results: {image_result_to_insert}", flush=True)
                results_res = supabase_client.table("image_benchmark_results").insert(image_result_to_insert).execute()
                if not results_res.data:
                    print("[benchmarkly] ERROR: Failed to insert into image_benchmark_results table.", file=sys.stderr, flush=True)
                    error_detail = getattr(results_res, 'error', None)
                    if error_detail:
                        print(f"[benchmarkly] Supabase error: {error_detail.message}", file=sys.stderr, flush=True)
                    supabase_client.table("benchmarks").update({"status": "failed", "error_message": "Failed to insert Image results"}).eq("id", benchmark_id).execute()
                    return
            print(f"[benchmarkly] DEBUG: Successfully inserted Image results.", flush=True)

        supabase_client.table("benchmarks").update({"status": "completed"}).eq("id", benchmark_id).execute()
        print("[benchmarkly] Successfully pushed results to Supabase and marked benchmark as completed.", flush=True)

    except Exception as e:
        print(f"[benchmarkly] ERROR: Overall exception in push_to_supabase: {e}", file=sys.stderr, flush=True)
        if 'benchmark_id' in locals() and benchmark_id and supabase_client: # Ensure benchmark_id and client exist
            try:
                supabase_client.table("benchmarks").update({"status": "failed", "error_message": f"Push failed: {str(e)}"}).eq("id", benchmark_id).execute()
            except Exception as e_update:
                print(f"[benchmarkly] ERROR: Could not update benchmark status to failed after another error: {e_update}", file=sys.stderr, flush=True)

def run_llm_benchmark(
    model_repo: str, tokens_in: int, tokens_out: int, batch_size: int
) -> Dict[str, float]:
    """Run the real NVIDIA vLLM throughput benchmark for a single *batch_size*.

    This function invokes the `Nvidia.inference.benchmark_throughput` module in a
    subprocess, passing it the model repo, prompt/response lengths, and
    `--num-prompts` equal to *batch_size*.  The benchmark script writes a JSON
    file containing metrics, which we parse and return.
    """

    # Create a temporary directory to hold the results JSON
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        output_path = tmpdir / "results.json"

        cmd = [
            sys.executable,
            "-u",
            "./benchmark_throughput.py", # Script is now in the root
            "--backend",
            "vllm",
            "--model",
            model_repo,
            "--input-len",
            str(tokens_in),
            "--output-len",
            str(tokens_out),
            "--num-prompts",
            str(batch_size), # This corresponds to the number of requests vLLM will run
            "--output-json",
            str(output_path),
            "--trust-remote-code", # Often needed for Hugging Face models
            # Add other relevant vLLM parameters as needed, e.g.:
            # "--tensor-parallel-size", "1",
            # "--dtype", "auto",
        ]

        # Surface stdout/stderr live so the user can see model loading progress etc.
        # Use Popen for live output, then communicate or wait.
        print(f"[benchmarkly] Running LLM benchmark with command: {' '.join(cmd)}", flush=True)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(f"[vLLM_script] {line.strip()}", flush=True)
        process.wait()

        if process.returncode != 0:
            fatal(
                f"Benchmark script exited with code {process.returncode} for batch_size={batch_size}."
            )

        if not output_path.exists():
            fatal(f"Benchmark script did not produce expected results.json file at {output_path}")

        try:
            data = json.loads(output_path.read_text())
        except Exception as exc:  # noqa: BLE001
            fatal(f"Failed to parse benchmark results JSON: {exc}")

        # The NVIDIA script uses "requests_per_second" for throughput and "tokens_per_second"
        return {
            "batch_size": batch_size,
            "tokens_per_second": data.get("tokens_per_second"),
            "output_throughput": data.get("requests_per_second"),
            "elapsed_seconds": data.get("elapsed_time"),
        }


def run_image_benchmark(
    model_repo: str, step_count: int, res_w: int, res_h: int
) -> Dict[str, float]:
    """Run a real image-generation benchmark using Hugging Face Diffusers."""
    # Load pipeline
    from diffusers import DiffusionPipeline
    import torch

    # Use HF token if provided
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[benchmarkly] Loading image pipeline {model_repo} on {device}", flush=True)
    pipe = DiffusionPipeline.from_pretrained(
        model_repo,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_auth_token=hf_token,
    )
    pipe = pipe.to(device)

    # Run a single inference and time it
    start_time = time.perf_counter()
    output = pipe(
        prompt="",  # empty prompt by default
        height=res_h,
        width=res_w,
        num_inference_steps=step_count,
    )
    elapsed = time.perf_counter() - start_time
    images_per_second = 1.0 / elapsed
    print(f"[benchmarkly] Image generated in {elapsed:.3f}s ({images_per_second:.2f} images/s)", flush=True)
    return {
        "images_per_second": images_per_second,
        "elapsed_seconds": elapsed,
    }


# --------------------------------------------------------------------------------------
# Main dispatcher logic
# --------------------------------------------------------------------------------------

def main() -> None:
    # Load Supabase credentials from supabase_credentials.env
    env_path = Path(__file__).resolve().parent / "supabase_credentials.env"
    if env_path.exists():
        print(f"[benchmarkly] Loading environment variables from: {env_path}", flush=True)
        load_dotenv(dotenv_path=env_path)
    else:
        fatal(f"Supabase credentials file not found at {env_path}")

    # Initialize Supabase client
    supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    supabase_key = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    if not supabase_url or not supabase_key:
        fatal("NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY are required in environment")
    supabase_client = create_client(supabase_url, supabase_key)
    print("[benchmarkly] Supabase client initialized.", flush=True)

    # Get the benchmark ID to process
    benchmark_id = os.getenv("BENCHMARK_ID")
    if not benchmark_id:
        fatal("BENCHMARK_ID environment variable is required")

    # Fetch the main benchmark record
    try:
        bench_res = supabase_client.table("benchmarks").select("*").eq("id", benchmark_id).execute()
        if not bench_res.data:
            fatal(f"Could not fetch benchmark record for ID={benchmark_id} or no data returned (bench_res.data is empty or None).")
    except Exception as e:
        # It's good practice to log the actual exception for debugging
        print(f"[benchmarkly] Exception details when fetching benchmark: {type(e).__name__} - {str(e)}", file=sys.stderr, flush=True)
        fatal(f"Error fetching benchmark record for ID={benchmark_id}. See exception details above.")
    
    bench_record = bench_res.data[0]
    benchmark_type = bench_record["benchmark_type"].lower()
    model_repo = bench_record["ai_model_repo"]

    # Update status to 'running'
    try:
        supabase_client.table("benchmarks").update({"status": "running"}).eq("id", benchmark_id).execute()
    except Exception as e:
        # Log this error but proceed if possible, as status update is not critical for the benchmark itself
        print(f"[benchmarkly] WARN: Could not update benchmark status to 'running' for ID={benchmark_id}: {e}", file=sys.stderr, flush=True)

    # Fetch benchmark parameters
    params_table_name = "llm_benchmark_params" if benchmark_type == "llm" else "image_benchmark_params"
    try:
        params_res = supabase_client.table(params_table_name).select("*").eq("benchmark_id", benchmark_id).execute()
        if not params_res.data:
            fatal(f"Could not fetch parameters from {params_table_name} for benchmark ID={benchmark_id} or no data returned (params_res.data is empty or None).")
    except Exception as e:
        print(f"[benchmarkly] Exception details when fetching params: {type(e).__name__} - {str(e)}", file=sys.stderr, flush=True)
        fatal(f"Error fetching parameters from {params_table_name} for benchmark ID={benchmark_id}. See exception details above.")
        
    params_record = params_res.data[0]
    param_id = params_record["id"]

    # Run the benchmark and insert result rows
    results_output = [] # type: List[Dict[str, Any]] | Dict[str, Any]
    current_gpu = get_gpu_model() # Get GPU model once

    if benchmark_type == "llm":
        tokens_in = params_record["tokens_in"]
        tokens_out = params_record["tokens_out"]
        batch_sizes = params_record["batch_sizes"]
        for bs in batch_sizes:
            metrics = run_llm_benchmark(model_repo, tokens_in, tokens_out, bs)
            results_output.append(metrics)
            try:
                supabase_client.table("llm_benchmark_results").insert({
                    "param_id": param_id,
                    "gpu_model": current_gpu,
                    "batch_size": metrics["batch_size"],
                    "output_throughput": metrics["output_throughput"],
                    "tokens_per_second": metrics["tokens_per_second"],
                }).execute()
            except Exception as e:
                print(f"[benchmarkly] WARN: Failed to insert LLM result for batch_size={bs}: {e}", file=sys.stderr, flush=True)
    else: # Image benchmark
        step_count = params_record["step_count"]
        res_w = params_record["resolution_width"]
        res_h = params_record["resolution_height"]
        metrics = run_image_benchmark(model_repo, step_count, res_w, res_h)
        results_output = metrics # For image, results_output is a single dict
        try:
            supabase_client.table("image_benchmark_results").insert({
                "param_id": param_id,
                "gpu_model": current_gpu,
                "images_per_second": metrics["images_per_second"],
            }).execute()
        except Exception as e:
            print(f"[benchmarkly] WARN: Failed to insert Image result: {e}", file=sys.stderr, flush=True)

    # Mark benchmark as completed
    try:
        supabase_client.table("benchmarks").update({"status": "completed"}).eq("id", benchmark_id).execute()
        print("[benchmarkly] Benchmark completed and results pushed (or attempted).", flush=True)
    except Exception as e:
        print(f"[benchmarkly] WARN: Could not update benchmark status to 'completed' for ID={benchmark_id}: {e}", file=sys.stderr, flush=True)

    # Print summary JSON to stdout
    output = {
        "benchmark_id": benchmark_id,
        "benchmark_type": benchmark_type.upper(),
        "model_repo": model_repo,
        "results": results_output,
    }
    print(json.dumps(output, indent=2), flush=True)


# ----------------------------------------------------------------------------
def terminate_runpod_pod():
    """Terminate the RunPod pod by calling the RunPod API."""
    pod_id = os.getenv("RUNPOD_POD_ID")
    api_key = os.getenv("RUNPOD_API_KEY")
    if pod_id and api_key:
        headers = {"Authorization": f"Bearer {api_key}"}
        url = f"https://rest.runpod.io/v1/pods/{pod_id}"
        try:
            resp = requests.delete(url, headers=headers)
            if resp.status_code in (200, 204):
                print(f"[benchmarkly] RunPod pod {pod_id} termination requested (status {resp.status_code}).", flush=True)
            else:
                print(f"[benchmarkly] WARN: Failed to terminate RunPod pod {pod_id}: {resp.status_code} {resp.text}", flush=True)
        except Exception as e:
            print(f"[benchmarkly] WARN: Exception during RunPod pod termination: {e}", file=sys.stderr, flush=True)
    else:
        print("[benchmarkly] No RUNPOD_POD_ID or RUNPOD_API_KEY provided, skipping pod termination.", flush=True)

if __name__ == "__main__":
    try:
        main()
    finally:
        terminate_runpod_pod() 