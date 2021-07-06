#!/bin/bash
apt-get update && apt-get upgrade -y
apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx ffmpeg

pip install cython #cupy
pip install numpy "opencv-python<4.3" matplotlib pillow ffmpeg-python 

# Pycocotools
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# For Jupyter Notebook
# pip install jupyterlab-nvdashboard
# jupyter labextension install jupyterlab-nvdashboard
# jupyter-lab --ip=0.0.0.0 --allow-root --NotebookApp.token=''