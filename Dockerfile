FROM ubuntu:16.04
MAINTAINER gitmirgut
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libjpeg8-dev \
    libtiff5-dev \
    libjasper-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libgtk2.0-dev \
    libatlas-base-dev \
    gfortran \
    python3 \
    python3.5-dev \
    python-dev \
    python-numpy \
    libtbb2 \
    libtbb-dev \
    ffmpeg \
    wget

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py

RUN pip3 install \
    numpy \
    Cython \
    git+https://github.com/scikit-image/scikit-image.git

RUN cd ~ && \
    git clone https://github.com/Itseez/opencv.git && \
    cd opencv && \
    pwd && \
    git checkout 3.1.0 && \
    cd ~ && \
    git clone https://github.com/Itseez/opencv_contrib.git && \
    cd opencv_contrib && \
    git checkout 3.1.0

RUN cd ~/opencv && \
    mkdir build && \
    cd build && \
    cmake  -D CMAKE_BUILD_TYPE=RELEASE \
	    -D CMAKE_INSTALL_PREFIX=/usr/local \
	    -D INSTALL_C_EXAMPLES=OFF \
	    -D INSTALL_PYTHON_EXAMPLES=OFF \
	    -D OPENCV_EXTRA_MODULES_PATH=/root/opencv_contrib/modules \
	    -D BUILD_EXAMPLES=OFF \
        -D PYTHON_EXECUTABLE=/usr/bin/python3 .. && \
    make -j4 && \
    make install && \
    ldconfig

RUN pip3 install \
    scipy \
    matplotlib \
git+https://github.com/gitmirgut/bb_stitcher.git \