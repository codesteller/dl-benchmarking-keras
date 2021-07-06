#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters." 
    echo "Usage - ./startDocker.sh <GPU_ID> <Workspace_Path>"
    echo "Example: ./startDocker.sh 7 $(pwd)/.."
    exit 1
fi

gpu_id=$1
gpu_device="device=${gpu_id}"
echo $gpu_device
wspath=$2
echo $wspath

if [ $1 != all ]; then 
    docker run --gpus device=$1 -it --rm -p 9000:8888 -p 0.0.0.0:9006:6006 \
    --shm-size=1g --ulimit memlock=-1 -v $2:/workspace/myspace \
    -v /home/nvidia/Developer/03_Customers/02_Crisp/02_Flipkart/01_CVbenchmark/dataset/dogs_cats/train_data:/dataset/dogs_cats/train_data \
    -w /workspace/myspace  \
    codesteller/tf2-py3-ffmpeg-coco:v1.1 # nvcr.io/nvidia/tensorflow:21.03-tf2-py3
    

else 
    docker run --gpus all -it --rm -p 9000:8888 -p 0.0.0.0:9006:6006 \
    --shm-size=1g --ulimit memlock=-1 \
    -v $2:/workspace/myspace \
    -v /home/nvidia/Developer/03_Customers/02_Crisp/02_Flipkart/01_CVbenchmark/dataset/dogs_cats/train_data:/dataset/dogs_cats/train_data \
    -w /workspace/myspace \
    codesteller/tf2-py3-ffmpeg-coco:v1.1 # nvcr.io/nvidia/tensorflow:21.03-tf2-py3
fi 
