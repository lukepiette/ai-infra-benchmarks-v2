#!/bin/bash

# Directory to store results
RESULTS_DIR="benchmark_throughput_results"
mkdir -p $RESULTS_DIR

# Model and other parameters
MODEL="openchat/openchat-3.5-0106"
INPUT_LEN=128
OUTPUT_LEN=128
# GPU_MEMORY_UTILIZATION=0.98

# Array of num-prompts values
NUM_PROMPTS=(2 4 8 16 32 64)

# Run the benchmark for each num-prompts value and store results in different files
for num_prompts in "${NUM_PROMPTS[@]}"; do
    OUTPUT_FILE="${RESULTS_DIR}/results_${num_prompts}.json"
    python benchmark_throughput.py --backend vllm \
                                   --model $MODEL \
                                   --input-len $INPUT_LEN \
                                   --output-len $OUTPUT_LEN \
                                   --num-prompts $num_prompts \
                                   --output-json $OUTPUT_FILE 
                                   # --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
                                   # --dataset path/to/your/dataset.json
                                   # --tokenizer $MODEL \
                                   # --quantization None \
                                   # --tensor-parallel-size 1 \
                                   # --n 1 \
                                   # --use-beam-search \
                                   # --seed 42 \
                                   # --trust-remote-code \
                                   # --max-model-len None \
                                   # --dtype auto \
                                   # --enforce-eager \
                                   # --kv-cache-dtype auto \
                                   # --quantization-param-path path/to/quantization_params.json \
                                   # --device auto \
                                   # --enable-prefix-caching \
                                   # --enable-chunked-prefill \
                                   # --max-num-batched-tokens 1024 \
                                   # --download-dir path/to/download_dir \
                                   # --distributed-executor-backend mp \
                                   # --load-format auto
done
