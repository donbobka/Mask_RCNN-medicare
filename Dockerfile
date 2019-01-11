FROM nvidia/cuda:9.2-cudnn7-devel as base

RUN apt-get update && apt-get install -y \
    python3 python3-pip

RUN pip3 install -r requirements.txt
