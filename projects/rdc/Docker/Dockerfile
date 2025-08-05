# Use amdsmi base image
FROM amdsmi-image

# Sync ROCm repositories
RUN apt-get update && apt-get install -y wget gnupg2 && \
    wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - && \
    echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ jammy main' | tee /etc/apt/sources.list.d/rocm.list

# Install ROCm runtime and dependencies
RUN apt-get update && \
    apt-get install -y \
    rocm-smi-lib \
    cmake \
    make \
    g++ \
    doxygen \
    texlive-latex-base \
    automake \
    unzip \
    build-essential \
    autoconf \
    libtool \
    pkg-config \
    libgflags-dev \
    libgtest-dev \
    clang \
    libc++-dev \
    curl \
    libcap-dev \
    python3-argcomplete \
    python3-pip \
    bash && \
    rm -rf /var/lib/apt/lists/*

# Update setuptools
RUN python3 -m pip install --upgrade setuptools==59.6.0

# Check for modprobe
RUN command -v modprobe || echo "modprobe not found"

# Set environment variables
ENV GRPC_ROOT=/opt/grpc
ENV RDC_LIB_DIR=/opt/rocm/lib/rdc
ENV CMAKE_ROOT=/usr/bin/cmake

# Install gRPC
RUN git clone -b v1.61.0 https://github.com/grpc/grpc --depth=1 --shallow-submodules --recurse-submodules && \
    cd grpc && \
    cmake -B build \
        -DgRPC_INSTALL=ON \
        -DgRPC_BUILD_TESTS=OFF \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_PREFIX="$GRPC_ROOT" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_BUILD_TYPE=Release && \
    make -C build -j $(nproc) && \
    make -C build install && \
    echo "$GRPC_ROOT" | tee /etc/ld.so.conf.d/grpc.conf

# Build and install RDC
RUN git clone https://github.com/ROCm/rdc && \
    cd rdc && \
    cmake -B build -DGRPC_ROOT="$GRPC_ROOT" \
        -DSMIDIR="$SMI_DIR" \
        -DBUILD_TESTS=OFF \
        -DBUILD_PROFILER=OFF \
        -DBUILD_RUNTIME=OFF \
        -DBUILD_RVS=OFF && \
    make -C build -j $(nproc) && \
    make -C build install

# Update system library path
RUN export RDC_LIB_DIR=/opt/rocm/lib/rdc && \
    export GRPC_LIB_DIR="/opt/grpc/lib" && \
    echo "${RDC_LIB_DIR}" | tee /etc/ld.so.conf.d/x86_64-librdc_client.conf && \
    echo "${GRPC_LIB_DIR}" | tee -a /etc/ld.so.conf.d/x86_64-librdc_client.conf && \
    ldconfig

# Expose the port for Prometheus
EXPOSE 50051

# Expose the prometheus plugin port
EXPOSE 5000

# Set working directory to /opt/rocm/libexec/rdc/python_binding
WORKDIR /opt/rocm/libexec/rdc/python_binding

# Install Prometheus
RUN wget https://github.com/prometheus/prometheus/releases/download/v2.41.0/prometheus-2.41.0.linux-amd64.tar.gz && \
    tar xvfz prometheus-2.41.0.linux-amd64.tar.gz && \
    mv prometheus-2.41.0.linux-amd64/prometheus /usr/local/bin/ && \
    rm -rf prometheus-2.41.0.linux-amd64*

# Install Grafana
RUN wget https://dl.grafana.com/oss/release/grafana-9.3.2.linux-amd64.tar.gz && \
    tar -zxvf grafana-9.3.2.linux-amd64.tar.gz && \
    mv grafana-9.3.2 /usr/local/grafana && \
    rm grafana-9.3.2.linux-amd64.tar.gz

# Install Prometheus client for Python
RUN python3 -m pip install prometheus_client

# Ensure sudo can run without a password for the rdcd command
RUN echo "ALL ALL=(ALL) NOPASSWD: /opt/rocm/bin/rdcd" >> /etc/sudoers

# Set the entry point to run the rdcd command
ENTRYPOINT ["sudo", "/opt/rocm/bin/rdcd", "-u"]
