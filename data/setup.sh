#!/bin/bash

# ROS Installation
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-get -y install curl # if you haven't already installed curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt-get update;  sudo apt-get -y install ros-noetic-desktop
source /opt/ros/noetic/setup.bash

# GCC Compiler
sudo apt-get -y install gcc c++; gcc --version;  g++ --version

# CUDA Toolkit (-L/usr/lib/wsl/lib/)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4

# Solectrix
sudo apt-get -y install libsdl2-ttf-dev libgl-dev libglm-dev libavcodec-dev libavformat-dev libavutil-dev libjpeg-turbo8-dev libpng-dev libswscale-dev libopencv-dev libsdl2-dev
PATH=/home/ubuntu/scania-raw-diff/data/solectrix/src/release/:$PATH

# SoftISP
sudo apt-get install ffmpeg
sudo dpkg -i /home/ubuntu/scania-raw-diff/data/soft-isp/231024_scania-man_SoftISP_3.8.2_ubuntu_20.04.deb
sudo apt-get -f install  # Dependencies installation
SoftISP --verbose=1 \
        --input rosbag \
        --file /home/ubuntu/campipe-road-data/imx490/221103/proframe_dev0_port16_2022-11-03_10-16-48.bag \
        --config /home/ubuntu/scania-raw-diff/data/soft-isp/profiles/imx490.json