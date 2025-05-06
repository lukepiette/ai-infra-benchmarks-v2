#!/bin/bash

# Directory to store results
RESULTS_DIR="benchmark_results"
mkdir -p $RESULTS_DIR

# Model and other parameters
MODEL="openchat/openchat-3.5-0106"
INPUT_LEN=128
OUTPUT_LEN=128
GPU_MEMORY_UTILIZATION=0.98

# Array of num-prompts values
NUM_PROMPTS=(2 4 8 16 32 64)

# Run the benchmark for each num-prompts value and store results in different files
for num_prompts in "${NUM_PROMPTS[@]}"; do
    OUTPUT_FILE="${RESULTS_DIR}/results_${num_prompts}.json"
    python benchmark_throughput.py --backend vllm \
                                   --model $MODEL \
                                   --input-len $INPUT_LEN \
                                   --output-len $OUTPUT_LEN \
                                   --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
                                   --num-prompts $num_prompts \
                                   --output-json $OUTPUT_FILE
                                   # --dataset path/to/your/dataset.json \
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

# Latency benchmark script
LATENCY_RESULTS_DIR="latency_benchmark_results"
mkdir -p $LATENCY_RESULTS_DIR

# Array of batch sizes for latency testing
BATCH_SIZES=(1 2 4 8 16 32)

for batch_size in "${BATCH_SIZES[@]}"; do
    LATENCY_OUTPUT_FILE="${LATENCY_RESULTS_DIR}/latency_results_${batch_size}.json"
    python benchmark_latency.py --model $MODEL \
                                --input-len $INPUT_LEN \
                                --output-len $OUTPUT_LEN \
                                --batch-size $batch_size \
                                --output-json $LATENCY_OUTPUT_FILE
                                # --speculative-model None \
                                # --num-speculative-tokens None \
                                # --speculative-draft-tensor-parallel-size None \
                                # --tokenizer $MODEL \
                                # --quantization None \
                                # --tensor-parallel-size 1 \
                                # --n 1 \
                                # --use-beam-search \
                                # --trust-remote-code \
                                # --max-model-len None \
                                # --dtype auto \
                                # --enforce-eager \
                                # --kv-cache-dtype auto \
                                # --quantization-param-path path/to/quantization_params.json \
                                # --profile \
                                # --profile-result-dir path/to/profile_result_dir \
                                # --device auto \
                                # --block-size 16 \
                                # --enable-chunked-prefill \
                                # --enable-prefix-caching \
                                # --use-v2-block-manager \
                                # --ray-workers-use-nsight \
                                # --download-dir path/to/download_dir \
                                # --load-format auto \
                                # --distributed-executor-backend mp \
                                # --otlp-traces-endpoint None
done
