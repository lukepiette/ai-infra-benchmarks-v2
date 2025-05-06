# syntax=docker/dockerfile:1

# Use an NVIDIA CUDA base image that includes CUDA 12.1 and cuDNN 8
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3 (system default), pip, and build-essential
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3 \
       python3-pip \
       git \
       build-essential \
       kmod \
    && rm -rf /var/lib/apt/lists/*

# Ensure `python` and `pip` refer to Python3 and pip3
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Create a non-root user (optional but recommended)
# Using a high UID/GID to avoid conflicts with host system users if volumes are mounted
RUN groupadd --gid 1001 runner && \
    useradd --uid 1001 --gid 1001 -ms /bin/bash runner

# Set working directory
WORKDIR /app

# Copy just the requirements first to leverage Docker layer caching
COPY common/requirements.txt ./requirements.txt

# Extra OS build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-dev ninja-build cmake && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and pre-install everything vLLM needs at build-time
RUN pip install --no-cache-dir --upgrade pip && \
    # Core build stack (new setuptools first!)
    pip install --no-cache-dir \
        setuptools>=70 setuptools_scm pybind11 ninja cmake packaging wheel && \
    # Heavy deps needed at runtime and for vllm's setup.py
    pip install --no-cache-dir \
        numpy \
        torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 && \
    # Finally, install the project requirements (no build isolation)
    pip install --no-cache-dir --break-system-packages --no-build-isolation -r requirements.txt

# Copy the rest of the source tree
COPY . .

# Give ownership to the non-root user and switch to it
RUN chown -R runner:runner /app
USER runner

# Default entrypoint – the benchmark orchestrator script
ENTRYPOINT ["python3", "-u", "/app/run_benchmark.py"] 