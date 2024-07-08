import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Union

import aiohttp
from tqdm.asyncio import tqdm

@dataclass
class RequestInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str

@dataclass
class RequestOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0 
    prompt_len: int = 0
    error: str = ""


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)



async def async_request_chat_completions(
    request_func_input: RequestInput,
    pbar: Optional[tqdm] = None,
) -> RequestOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("v1/chat/completions"), "Chat Completions API URL must end with 'v1/chat/completions'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "messages": [
                {
                    "role": "user",
                    "content": request_func_input.prompt,
                },
            ],
            "temperature": 0.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('API_KEY')}",
        }

        output = RequestOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        st = time.perf_counter()
        latency = 0.0
        # TODO: Latency calculation is faulty. Fix it.
        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = str.removeprefix(chunk_bytes.decode("utf-8"), "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            delta = data["choices"][0]["delta"]
                            if delta.get("content", None):
                                generated_text += delta["content"]

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
    
    if pbar:
        pbar.update(1)

    return output
async def async_request_completions(
    request_func_input: RequestInput,
    pbar: Optional[tqdm] = None,
) -> RequestOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "v1/completions"
    ), "Completions API URL must end with 'v1/completions'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {os.environ.get('API_KEY')}"
        }

        output = RequestOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        st = time.perf_counter()
        # TODO: Latency calculation is faulty. Fix it.
        latency = 0.0
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = str.removeprefix(chunk_bytes.decode("utf-8"),
                                              "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            data = json.loads(chunk)

                            if data["choices"][0]["text"]:
                                timestamp = time.perf_counter()
                                generated_text += data["choices"][0]["text"]

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output
