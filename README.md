# Benchmarking with `vllm`

## Overview

This document provides instructions on how to run benchmarks using the `vllm`. You can run benchmarks through Python scripts or shell scripts provided in this repository.

## Installation

1. **Install `vllm`:**

   To install the `vllm` library, use pip:

   ```bash
   pip install vllm
2. **Running Benchmarks**
Using Python Script  Once vllm is installed, you can execute the Python script to run a benchmark. Here’s an example command:

   ```bash
   python benchmark_throughput.py --backend vllm --model "meta-llama/Meta-Llama-3-8B-Instruct" --input-len=128 --output-len=128 --gpu-memory-        
   utilization=0.98 --num-prompts=1


## Using Shell Scripts

Alternatively, you can use the provided shell scripts to run benchmarks. Follow these steps:

### Grant Execute Permissions:

Before running the shell scripts, make sure to grant execute permissions using chmod:

```bash
chmod +x benchmark_throughput.sh
chmod +x benchmark_latency.sh
chmod +x benchmark_all.sh
```

### Run the Benchmarks:
```bash
For throughput benchmarking: ./benchmark_throughput.sh 
For latency benchmarking: ./benchmark_latency.sh
To run both throughput and latency benchmarks: ./benchmark_all.sh
```

### Additional Information

Make sure to replace the example parameters with your own if needed.
The shell scripts and Python script are designed to be run in a Unix-like environment.
For any further customization or parameters, refer to the script documentation or source code.
