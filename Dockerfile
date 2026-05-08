FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir fastapi \
    uvicorn \
    httpx \
    pydantic \
    PyYAML \
    apscheduler \
    qdrant-client