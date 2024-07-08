# Use an official Python runtime as a parent image
FROM python:3.9-slim

RUN apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /opt

# Set up virtual environment
RUN python3.11 -m venv venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .
ENTRYPOINT ["python3", "-m", "load_benchmark.main"]

