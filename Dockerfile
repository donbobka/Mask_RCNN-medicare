FROM nvidia/cuda:9.2-cudnn7-devel as base

RUN pip3 install -r requirements.txt
