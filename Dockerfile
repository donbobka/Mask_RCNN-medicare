FROM nvidia/cuda:9.0-cudnn7-devel as base

RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
