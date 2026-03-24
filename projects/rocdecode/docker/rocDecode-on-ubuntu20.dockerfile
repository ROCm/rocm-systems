FROM ubuntu:20.04

# install base dependencies
RUN apt-get update -y
#RUN apt-get dist-upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install gcc g++ cmake pkg-config git apt-utils sudo vainfo dialog

# install ROCm
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install initramfs-tools libnuma-dev wget keyboard-configuration && \
        wget https://repo.radeon.com/amdgpu-install/6.1/ubuntu/focal/amdgpu-install_6.1.60100-1_all.deb && \
        sudo apt-get install -y ./amdgpu-install_6.1.60100-1_all.deb && \
        sudo amdgpu-install -y --usecase=rocm

WORKDIR /workspace

# install rocDecode package
RUN DEBIAN_FRONTEND=noninteractive sudo apt install -y rocdecode rocdecode-dev rocdecode-test rocdecode-host