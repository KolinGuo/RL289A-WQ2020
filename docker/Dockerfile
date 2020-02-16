ARG CUDA="10.2"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04

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
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

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

WORKDIR /root

######################################
# SECTION 5: Add running instruction #
######################################
RUN echo "echo -e \"\n\"" >> /root/.bashrc
RUN echo "echo -e \"################################################################################\n\"" >> /root/.bashrc
RUN echo "echo -e \"\tCommand to enter repository:\n\t\tcd /root/RL289A-WQ2020 && xvfb-run -s "-screen 0 1280x720x24" jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --port=9000 &\n\"" >> /root/.bashrc
RUN echo "echo -e \"################################################################################\n\"" >> /root/.bashrc

