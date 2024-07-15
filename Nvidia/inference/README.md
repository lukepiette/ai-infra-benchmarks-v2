[WIP- vllm]
** pip install vllm 
** wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
** python benchmark_throughput.py  --model "openchat/openchat-3.5-0106"     --dataset "./ShareGPT_V3_unfiltered_cleaned_split.json"  --num-prompts=100  --tensor-parallel-size=1 --dtype=bfloat16 --output-json results.json --backend vllm  

## shell script
** benchmarks on different batch size. 

[TODO: Update the documentation]