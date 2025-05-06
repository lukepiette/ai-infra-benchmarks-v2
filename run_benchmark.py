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
    """Fake image benchmark producing images/s metric."""
    start_time = time.perf_counter()
    time.sleep(0.5)  # Simulate some work for the image benchmark
    elapsed = time.perf_counter() - start_time

    images_per_second = 1.0 / elapsed  # one image generated in the sleep()
    return {
        "images_per_second": images_per_second,
        "elapsed_seconds": elapsed,
    }


# --------------------------------------------------------------------------------------
# Main dispatcher logic
# --------------------------------------------------------------------------------------

def main() -> None:  # noqa: D401 – simple main
    benchmark_type_raw = os.getenv("BENCHMARK_TYPE")
    model_repo = os.getenv("AI_MODEL_REPO")

    if not benchmark_type_raw:
        fatal("BENCHMARK_TYPE environment variable is required (\"LLM\" or \"Image\")")
    if not model_repo:
        fatal("AI_MODEL_REPO environment variable is required (HuggingFace repo id)")

    benchmark_type = benchmark_type_raw.strip().lower()

    # ------------------------------------------------------------------ LLM benchmark --
    if benchmark_type == "llm":
        params_json = os.getenv("LLM_PARAMS_JSON")
        if not params_json:
            fatal("LLM_PARAMS_JSON environment variable is required for LLM benchmarks")
        try:
            params = json.loads(params_json)
        except json.JSONDecodeError as exc:
            fatal(f"Invalid JSON in LLM_PARAMS_JSON: {exc}")

        # Basic validation
        try:
            tokens_in: int = int(params["tokensIn"])
            tokens_out: int = int(params["tokensOut"])
            batch_sizes: List[int] = list(map(int, params["batchSizes"]))
        except (KeyError, ValueError, TypeError) as exc:
            fatal(f"LLM_PARAMS_JSON missing or malformed fields: {exc}")
        if tokens_in <= 0 or tokens_out <= 0:
            fatal("tokensIn and tokensOut must be positive integers")
        if not batch_sizes:
            fatal("batchSizes must contain at least one item")

        results: List[Dict[str, Any]] = []
        for bs in batch_sizes:
            metrics = run_llm_benchmark(model_repo, tokens_in, tokens_out, bs)
            results.append(metrics)

        output: Dict[str, Any] = {
            "benchmark_type": "LLM",
            "model_repo": model_repo,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "results": results,
        }

    # --------------------------------------------------------------- Image benchmark --
    elif benchmark_type == "image":
        params_json = os.getenv("IMAGE_PARAMS_JSON")
        if not params_json:
            fatal("IMAGE_PARAMS_JSON environment variable is required for Image benchmarks")
        try:
            params = json.loads(params_json)
        except json.JSONDecodeError as exc:
            fatal(f"Invalid JSON in IMAGE_PARAMS_JSON: {exc}")

        try:
            step_count: int = int(params["stepCount"])
            res_w: int = int(params["resolutionWidth"])
            res_h: int = int(params["resolutionHeight"])
        except (KeyError, ValueError, TypeError) as exc:
            fatal(f"IMAGE_PARAMS_JSON missing or malformed fields: {exc}")

        metrics = run_image_benchmark(model_repo, step_count, res_w, res_h)
        output = {
            "benchmark_type": "Image",
            "model_repo": model_repo,
            "step_count": step_count,
            "resolution_width": res_w,
            "resolution_height": res_h,
            "results": metrics,
        }

    else:
        fatal("BENCHMARK_TYPE must be either 'LLM' or 'Image'")

    # Print JSON results to stdout so the RunPod caller can capture them
    print(json.dumps(output, indent=2), flush=True)


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    main() 