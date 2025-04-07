# Use rocm/dev-ubuntu-22.04 as the base image
FROM rocm/dev-ubuntu-22.04

# Set environment variables for build directories and package patterns
ENV BUILD_FOLDER=/home/amdsmi/build
ENV DEB_BUILD="amd-smi-lib*99999-local_amd64.deb"
ENV DEB_BUILD_TEST="amd-smi-lib-tests*99999-local_amd64.deb"

# Set the working directory to /home
WORKDIR /home

# Install necessary system packages
RUN apt update && apt-get install -y git build-essential rpm pkg-config g++ python3 python3-pip python3-wheel python3-setuptools

# Upgrade pip and install cmake and virtualenv using pip
RUN python3 -m pip install --upgrade pip setuptools && \
    python3 -m pip install cmake virtualenv

# Clone the AMD SMI repository from GitHub
RUN git clone -b amd-mainline https://github.com/ROCm/amdsmi.git

# Navigate to the amdsmi directory
WORKDIR /home/amdsmi

# Build and Install AMDSMI
RUN rm -rf ${BUILD_FOLDER} && \
    mkdir -p ${BUILD_FOLDER} && \
    cd ${BUILD_FOLDER} && \
    cmake .. -DBUILD_TESTS=ON -DENABLE_ESMI_LIB=ON && \
    make -j $(nproc) VERBOSE=1 && \
    make package && \
    sudo apt install -y --allow-downgrades ${BUILD_FOLDER}/${DEB_BUILD} && \
    sudo ln -s /opt/rocm/bin/amd-smi /usr/local/bin

# Verify the installation of Python packages related to AMD SMI
RUN python3 -m pip list | grep -E "amd|pip|setuptools"

# Set the entrypoint to bash for interactive use
ENTRYPOINT ["/bin/bash"]
