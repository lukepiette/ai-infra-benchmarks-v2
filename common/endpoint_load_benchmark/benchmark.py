import asyncio
import aiohttp
import time
import numpy as np
from tqdm import tqdm
from typing import List, Tuple 
from endpoint_load_benchmark.utils import RequestInput, RequestOutput, async_request_chat_completions, async_request_completions
from tabulate import tabulate

class LoadTester:
    def __init__(self, url, qps, model_id):
        self.url = url
        self.qps = qps
        self.latencies = []
        self.errors = 0
        self.model_id = model_id
    # the function will yield one request every 1/qps seconds
    async def get_request(self, input_requests, qps):
        for request in input_requests:
            yield request
            await asyncio.sleep(1 / qps)

    async def benchmark(
        self,
        backend: str,
        api_url: str,
        model_id: str,
        input_requests: List[Tuple[str, int, int]],
        qps: float,
        disable_tqdm: bool,
    ):
        if backend == "endpoint":
            if "v1/completions" in api_url:
                request_func = async_request_completions
            else:
                request_func = async_request_chat_completions
        else:
            raise ValueError(f"Unknown backend: {backend}")

        print("Starting a test run...")
        test_prompt, test_prompt_len, test_output_len = input_requests[0]
        test_input = RequestInput(
            model=model_id,
            prompt=test_prompt,
            api_url=api_url,
            prompt_len=test_prompt_len,
            output_len=test_output_len,
        )
        test_output = await request_func(test_input)
        if not test_output.success:
            raise ValueError(
                "Initial test run failed - Please make sure benchmark arguments "
                f"are correctly specified. Error: {test_output.error}")
        else:
            print("Initial test run completed. Starting main benchmark run...")
        print(f"Query Per Second: {qps}")

        pbar = None if disable_tqdm else tqdm(total=len(input_requests))

        benchmark_start_time = time.perf_counter()
        tasks: List[asyncio.Task] = []
        async for request in self.get_request(input_requests, qps):
            prompt, prompt_len, output_len = request
            request_func_input = RequestInput(
                model=self.model_id,
                prompt=prompt,
                api_url=api_url,
                prompt_len=prompt_len,
                output_len=output_len,
            )
            tasks.append(
                asyncio.create_task(
                    request_func(request_func_input=request_func_input,
                                 pbar=pbar)))
        outputs: List[RequestOutput] = await asyncio.gather(*tasks)

        if pbar is not None:
            pbar.close()

        benchmark_duration = time.perf_counter() - benchmark_start_time

        metrics = self.calculate_metrics(input_requests, outputs, benchmark_duration)

        data = [
        ["Successful requests:", metrics['completed']],
        ["Benchmark duration (s):", "{:.2f}".format(benchmark_duration)],
        ["Request throughput (req/s):", "{:.2f}".format(metrics['request_throughput'])],
        ["Input token throughput (tok/s):", "{:.2f}".format(metrics['input_throughput'])],
        ["Output token throughput (tok/s):", "{:.2f}".format(metrics['output_throughput'])],
        ["Errors:", "{:.2f}".format(metrics['errors'])],
        ["Latency (s):", "{:.2f}".format(outputs[0].latency)],
        ]
        
        print(tabulate(data, tablefmt="grid"))
        result = {
            "duration": benchmark_duration,
            "completed": metrics['completed'],
            "request_throughput": metrics['request_throughput'],
            "input_throughput": metrics['input_throughput'],
            "output_throughput": metrics['output_throughput'],
            "errors": [output.error for output in outputs],
        }
        return result

    def calculate_metrics(self, input_requests, outputs, duration):
        completed = sum(1 for output in outputs if output.success)
        total_input = sum(req[1] for req in input_requests)
        total_output = sum(len(output.generated_text) for output in outputs if output.success)
        request_throughput = len(outputs) / duration
        input_throughput = total_input / duration
        output_throughput = total_output / duration
        #calculate errors
        errors = sum(1 for output in outputs if not output.success)
        
        metrics = {
            'completed': completed,
            'total_input': total_input,
            'total_output': total_output,
            'request_throughput': request_throughput,
            'input_throughput': input_throughput,
            'output_throughput': output_throughput,
            'errors': errors,
        }
        return metrics

    def run(self):
        loop = asyncio.get_event_loop()
        input_requests = [("This is a test prompt", 5, 50)] * self.qps
        qps = self.qps
        disable_tqdm = False
        outputs = loop.run_until_complete(self.benchmark(
            backend="endpoint",
            api_url=self.url,
            model_id=self.model_id,
            input_requests=input_requests,
            qps=qps,
            disable_tqdm=disable_tqdm,
        ))
        return outputs
