# RL289A-WQ2020
Final project for EEC 289A Reinforcement Learning Course

## Group Members 
  * Kolin Guo
  * Daniel Vallejo
  * Fengqiao Yang
  
## Prerequisites
  * Ubuntu 18.04
  * NVIDIA GPU with CUDA version >= 10.2
  * [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) version >= 19.03, API >= 1.40
  * [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#ubuntu-16041804-debian-jessiestretchbuster) (previously known as nvidia-docker)  
  
Command to test if all prerequisites are met:  
  `sudo docker run -it --rm --gpus all ubuntu nvidia-smi`
  
## Setup Instructions
  `bash ./setup.sh`  
If you need `sudo` permission to run `docker`, run `sudo -s` before running *setup.sh*.  
You should be greeted by the Docker container **openaigym** when this script finishes. The working directory is */root* and the repo is mounted at */root/RL289A-WQ2020*.  
