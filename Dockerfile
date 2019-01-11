FROM nvidia/cuda:9.2-cudnn7-devel as base

RUN apt-get update && apt-get install -y \
    python3 python3-pip

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
