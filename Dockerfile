FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install packaging first to avoid build errors
RUN pip install --upgrade pip && \
    pip install packaging

RUN pip install --no-cache-dir -r requirements.txt

COPY . .