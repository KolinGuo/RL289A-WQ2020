FROM tensorflow/tensorflow:2.1.0-gpu-py3-jupyter
#FROM tensorflow/tensorflow:2.2.0rc0-gpu-py3-jupyter

#########################################
# SECTION 0: Install OpenGL             #
#########################################
RUN dpkg --add-architecture i386 && \
    apt-get update && apt-get install -y --no-install-recommends \
        libxau6 libxau6:i386 \
        libxdmcp6 libxdmcp6:i386 \
        libxcb1 libxcb1:i386 \
        libxext6 libxext6:i386 \
        libx11-6 libx11-6:i386 && \
    rm -rf /var/lib/apt/lists/*

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
        ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
        ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

# Required for non-glvnd setups.
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install glvnd runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglvnd0 libglvnd0:i386 \
      libgl1 libgl1:i386 \
      libglx0 libglx0:i386 \
      libegl1 libegl1:i386 \
      libgles2 libgles2:i386 && \
    rm -rf /var/lib/apt/lists/*

# Install glvnd devel
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
      libglvnd-dev libglvnd-dev:i386 \
      libgl1-mesa-dev libgl1-mesa-dev:i386 \
      libegl1-mesa-dev libegl1-mesa-dev:i386 \
      libgles2-mesa-dev libgles2-mesa-dev:i386 && \
    rm -rf /var/lib/apt/lists/*

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

# Upgrade pip
RUN python3 -m pip install --upgrade pip

#########################################
# SECTION 4: Install Python Libraries   #
#########################################

COPY install/requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

RUN pip3 install jupyter_contrib_nbextensions \
  && jupyter contrib nbextension install \
  && pip3 install jupyter_nbextensions_configurator \
  && jupyter nbextensions_configurator enable

COPY gym-sokoban /gym-sokoban
RUN pip3 install -e /gym-sokoban

RUN pip3 uninstall -y tensorflow tensorboard
RUN pip3 uninstall -y tensorflow tensorboard
RUN pip3 install -U tf-nightly-gpu tb-nightly tensorboard_plugin_profile

RUN pip3 install virtualenv

WORKDIR /

######################################
# SECTION 5: Add running instruction #
######################################
COPY bashrc /tmp/
RUN cat /tmp/bashrc >> /etc/bash.bashrc
