# Load Benchmark

## Overview

HTTP load-testing and benchmarking tool designed to evaluate the performance and reliability of LLM Inference endpoints. Supports OpenAI compatibility APIs, with /v1/completions, v1/chat/completions API.

## Install Requirements 
pip install -r requirements.txt


## How to Run Benchmarks

You can run benchmarks in two ways: by cloning the repository or using Docker.

### Method 1: Clone the Repository

1. **Clone the Repository:**
    ```bash
    git clone <repo>
    ```

2. **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Navigate to the Repository:**
    ```bash
    cd <REPO>
    ```

4. **Set Environment Variable:**
    ```bash
    export API_KEY=<API_KEY>
    ```

5. **Run the Benchmark:**
    ```bash
    python3 -m endpoint_load_benchmark.main <URL> --qps <QPS_rate> --model_id <MODEL_ID>
    ```

### Method 2: Run Using Docker

1. **Clone the Repository:**
    ```bash
    git clone <repo>
    ```
2. **Navigate to the Repository:**
    ```bash
    cd <Repo>
    ```
3. **Build Docker Container:**
    ```bash
    docker build -t load_benchmark .
    ```
4. **Set Environment Variable and Run the Container:**

    ```bash
    docker run -e API_KEY=<API_KEY> load_benchmark <url> --qps <qps_num> --model_id <Model_id>
    ```

