# ImageGen Benchmarking
import time
import torch
from diffusers import FluxPipeline, DiffusionPipeline




# Add Models as Required 
BENCHMARK_MODELS = [
    "black-forest-labs/FLUX.1-dev",
    "stabilityai/stable-diffusion-xl-base-1.0"
]

COMMON_PROMPT = "A astronaut riding a unicorn in amazon rainforest"

class BenchMarkRunner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = BENCHMARK_MODELS
        self.common_prompt = COMMON_PROMPT
        self.resolutions = [
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048)
        ]

    def benchmark_model(self, model_name):
        print(f"Benchmarking model: {model_name}")
        
        results = {
            "model": model_name,  # Fixed key name to match print statement
            "timings": {}
        }
        
        for width, height in self.resolutions:
            print(f"Running benchmark for resolution: {width}x{height}")
            
            try:
                if model_name == "black-forest-labs/FLUX.1-dev":
                    pipe = FluxPipeline.from_pretrained(
                        "black-forest-labs/FLUX.1-dev",
                        torch_dtype=torch.bfloat16
                    )
                    pipe.enable_model_cpu_offload()
                    
                    # Warmup run
                    _ = pipe(self.common_prompt, width=width, height=height)
                    torch.cuda.synchronize()
                    
                    # Timed run
                    start_time = time.time()
                    _ = pipe(self.common_prompt, width=width, height=height)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                elif model_name == "stabilityai/stable-diffusion-xl-base-1.0":
                    pipe = DiffusionPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        use_safetensors=True,
                        variant="fp16"
                    )
                    pipe.to(self.device)
                    
                    # Warmup run
                    _ = pipe(self.common_prompt, width=width, height=height)
                    torch.cuda.synchronize()
                    
                    # Timed run
                    start_time = time.time()
                    _ = pipe(self.common_prompt, width=width, height=height)
                    torch.cuda.synchronize()
                    end_time = time.time()
                
                generation_time = end_time - start_time
                results["timings"][f"{width}x{height}"] = generation_time
                print(f"Generation time: {generation_time:.2f}s")
                
                # Cleanup
                del pipe
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error at resolution {width}x{height}: {str(e)}")
                results["timings"][f"{width}x{height}"] = f"Error: {str(e)}"
        
        return results

    def run_benchmarks(self):
        results = []
        for model in self.models:
            model_results = self.benchmark_model(model)
            results.append(model_results)

        # Print Summary
        print("\n=== Benchmark Summary ===")
        for result in results:
            print(f"\nModel: {result['model']}")
            for res, time in result['timings'].items():
                if isinstance(time, (int, float)):
                    print(f"{res}: {time:.2f}s")
                else:
                    print(f"{res}: {time}")

if __name__ == "__main__":
    runner = BenchMarkRunner()
    runner.run_benchmarks()
