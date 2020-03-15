#ARG CUDA="10.1"

#FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04
#FROM nvidia/cudagl:${CUDA}-devel-ubuntu18.04
FROM tensorflow/tensorflow:2.1.0-gpu-py3-jupyter

#########################################
# SECTION 0: Install cudnn              #
#########################################
#ENV CUDNN_VERSION 7.6.5.32
#LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
#
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    libcudnn7=$CUDNN_VERSION-1+cuda10.1 \
#    libcudnn7-dev=$CUDNN_VERSION-1+cuda10.1 \
#  && \
#    apt-mark hold libcudnn7 \
#  && \
#    apt-get install -y --no-install-recommends \
#    libnvinfer6=6.0.1-1+cuda10.1 \
#    libnvinfer-dev=6.0.1-1+cuda10.1 \
#    libnvinfer-plugin6=6.0.1-1+cuda10.1 \
#  && \
#    rm -rf /var/lib/apt/lists/*

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

#########################################
# SECTION 1: Essentials                 #
#########################################

#Update apt-get and upgrade
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils
RUN apt-get -y upgrade

#########################################
# SECTION 2: Common tools               #
#########################################

RUN apt-get update && apt-get install -y --no-install-recommends \
    vim git curl wget yasm cmake unzip pkg-config \
    checkinstall build-essential ca-certificates \
    software-properties-common

#########################################
# SECTION 3: Setup Libraries            #
#########################################

RUN apt-get update && apt-get install -y --no-install-recommends \
    zlib1g-dev libjpeg-dev xvfb ffmpeg xorg-dev \
    xorg-dev libboost-all-dev libsdl2-dev swig \
    libopenblas-base libatlas-base-dev graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python
#RUN apt-get update && apt-get install -y \
#    python3 \
#    python3-pip
#
# Upgrade pip
RUN python3 -m pip install --upgrade pip

#########################################
# SECTION 4: Install Python Libraries   #
#########################################

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

RUN pip3 install jupyter_contrib_nbextensions \
  && jupyter contrib nbextension install \
  && pip3 install jupyter_nbextensions_configurator \
  && jupyter nbextensions_configurator enable

WORKDIR /

######################################
# SECTION 5: Add running instruction #
######################################
COPY bashrc /tmp/
RUN cat /tmp/bashrc >> /etc/bash.bashrc
