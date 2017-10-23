FROM ubuntu:16.04

MAINTAINER Michal Olejniczak <kaomanster@gmail.com>

# Run, run so far away!
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config  \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
        libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
        libxvidcore-dev libx264-dev \
        libatlas-base-dev gfortran \
        unzip wget \
        python3.5-dev python3-pip python3-setuptools python3-tk \
        && apt-get clean && rm -rf /var/lib/apt/lists/* \
        && wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.1.0.zip && unzip opencv.zip && rm opencv.zip \
        && wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.1.0.zip && unzip opencv_contrib.zip && rm opencv_contrib.zip \
        && pip3 --no-cache-dir install Pillow matplotlib numpy scipy keras sklearn scikit-image imutils opencv-python h5py ipykernel jupyter \
        https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.1-cp35-cp35m-linux_x86_64.whl \
        && python3 -m ipykernel.kernelspec

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/
COPY notebooks /notebooks
# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY run_jupyter.sh /


EXPOSE 8888

WORKDIR "/notebooks"

CMD ["/run_jupyter.sh", "--allow-root"]
