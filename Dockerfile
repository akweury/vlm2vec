FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv python3.10-dev \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

COPY requirements.txt .

RUN pip3 install --upgrade pip && \
    pip3 install packaging torch numpy

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
