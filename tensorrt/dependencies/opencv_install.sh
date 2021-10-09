#!/bin/bash

apt-get update
apt-get install -y unzip wget
apt-get install -y build-essential cmake
apt-get install -y pkg-config

apt-get install -y libjpeg-dev libtiff5-dev libpng-dev
apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev
apt-get install -y libv4l-dev v4l-utils
apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
apt-get install -y libgtk2.0-dev
apt-get install -y mesa-utils libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-dev
apt-get install -y libatlas-base-dev gfortran libeigen3-dev

cd /
mkdir opencv
cd opencv

wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
unzip opencv.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip
unzip opencv_contrib.zip

cd opencv-4.2.0
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=OFF -D WITH_IPP=OFF -D WITH_1394=OFF -D BUILD_WITH_DEBUG_INFO=OFF -D BUILD_DOCS=OFF -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D WITH_QT=OFF -D WITH_GTK=ON -D WITH_OPENGL=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.2.0/modules -D WITH_V4L=ON  -D WITH_FFMPEG=ON -D WITH_XINE=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D OPENCV_GENERATE_PKGCONFIG=ON ../

time make -j64

make install

ldconfig

pkg-config --modversion opencv4
