#!/bin/bash

if [ $1 == all ] 
then 
    docker run --gpus all -it --rm -p 9000:8888 -p 0.0.0.0:9006:6000 --shm-size=1g --ulimit memlock=-1 -v $(pwd)/..:/workspace/myspace -w /workspace/myspace  nvcr.io/nvidia/tensorflow:21.05-tf2-py3
else 
    docker run --gpus '"device=$1"' -it --rm -p 9000:8888 -p 0.0.0.0:9006:6000 --shm-size=1g --ulimit memlock=-1 -v $(pwd)/..:/workspace/myspace -w /workspace/myspace nvcr.io/nvidia/tensorflow:21.05-tf2-py3
fi 

# docker run --gpus '"device=$1"' -it --rm -p 9000:8888 -p 0.0.0.0:9006:6000 --shm-size=1g --ulimit memlock=-1 -v $(pwd)/..:/workspace/myspace -w /workspace/myspace  nvcr.io/nvidia/tensorflow:21.05-tf2-py3
