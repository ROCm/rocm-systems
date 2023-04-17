FROM --platform=linux/amd64 ubuntu:20.04
WORKDIR /home

USER root

ENV DEBIAN_FRONTEND noninteractive
ENV TZ "US/Chicago"

RUN apt-get update
RUN apt-get install -y wget
RUN wget https://repo.radeon.com/amdgpu-install/5.3/ubuntu/focal/amdgpu-install_5.3.50300-1_all.deb 
RUN apt-get install -y ./amdgpu-install_5.3.50300-1_all.deb
RUN amdgpu-install --usecase=rocm -y --accept-eula

RUN apt-get install -y build-essential