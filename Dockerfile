# syntax=docker/dockerfile:1

# Use an NVIDIA CUDA base image that includes CUDA 12.1 and cuDNN 8
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11, pip, git, and build-essential
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.11 \
       python3-pip \
       git \
       build-essential \
       kmod \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python and pip3 the default pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create a non-root user (optional but recommended)
# Using a high UID/GID to avoid conflicts with host system users if volumes are mounted
RUN groupadd --gid 1001 runner && \
    useradd --uid 1001 --gid 1001 -ms /bin/bash runner

# Set working directory
WORKDIR /app

# Copy just the requirements first to leverage Docker layer caching
COPY common/requirements.txt ./requirements.txt

# Upgrade pip, install numpy first so build-time dependencies are available, then install all requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy && \
    pip install --no-cache-dir --break-system-packages --no-build-isolation -r requirements.txt

# Copy the rest of the source tree
COPY . .

# Give ownership to the non-root user and switch to it
RUN chown -R runner:runner /app
USER runner

# Default entrypoint – the benchmark orchestrator script
ENTRYPOINT ["python", "-u", "/app/run_benchmark.py"] 