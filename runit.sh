#!/usr/bin/env bash
echo 'create myenv'
conda create --name myenv
conda create -n myenv python=3.6
source activate myenv
pip install opencv-contrib-python==3.4.2.17
pip install scipy numpy autograd matplotlib jupyter sklearn
