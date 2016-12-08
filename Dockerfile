FROM ubuntu:16.04

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        wget \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python3.5-dev \
        python3-pip \
        rsync \
        cmake \
        pkg-config \
        libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
        libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
        libxvidcore-dev libx264-dev \
        libgtk-3-dev \
        libatlas-base-dev gfortran \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.1.0.zip \
  && unzip opencv.zip

RUN wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.1.0.zip \
  && unzip opencv_contrib.zip

RUN apt-get update && apt-get install -y python-pip
RUN pip install ipykernel
RUN pip3 install --upgrade pip
RUN pip3 install setuptools
RUN pip3 --no-cache-dir install \
        matplotlib \
        numpy \
        scipy \
        pyyaml \
        keras \
        sklearn \
        scikit-image \
        Pillow \
        imutils \
        && \
    python -m ipykernel.kernelspec

RUN cd opencv-3.1.0/  \
    && mkdir build \
    && cd build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib-3.1.0/modules \
    -D PYTHON_EXECUTABLE=/usr/bin/python .. \
    && make -j4 \
    && make install \
    && ldconfig \

RUN rm -rf opencv*

RUN pip3 --no-cache-dir install \
    http://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc0-cp35-cp35m-linux_x86_64.whl