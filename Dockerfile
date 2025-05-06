# syntax=docker/dockerfile:1

# Lightweight base image with Python 3.11 (good default for many ML libs)
FROM python:3.11-slim AS base

# Ensure we have bash & some build essentials for potential future installs
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (optional but recommended)
RUN useradd -ms /bin/bash runner

# Set working directory
WORKDIR /app

# Copy just the requirements first to leverage Docker layer caching if/when we add more deps
COPY common/requirements.txt ./requirements.txt

# Install Python dependencies – for now we only need what run_benchmark.py
# requires (all stdlib).  We still install the shared requirements in case future
# benchmark code relies on them.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source tree
COPY . .

# Give ownership to the non-root user and switch to it
RUN chown -R runner:runner /app
USER runner

# Default entrypoint – the benchmark orchestrator script
ENTRYPOINT ["python", "-u", "/app/run_benchmark.py"] 