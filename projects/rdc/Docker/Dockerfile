# Use amdsmi base image
FROM amdsmi-image

# Install ROCm runtime and dependencies
RUN apt-get update && \
    apt-get install -y \
    cmake \
    make \
    g++ \
    doxygen \
    texlive-latex-base \
    automake \
    unzip \
    build-essential \
    ccache \
    cmake \
    curl \
    git \
    gzip \
    jq \
    libcap-dev \
    tar \
    unzip \
    wget \
    zip \
    zstd && \
    rm -rf /var/lib/apt/lists/*

# Update setuptools
RUN python3 -m pip install --upgrade setuptools==59.6.0

# Check for modprobe
RUN command -v modprobe || echo "modprobe not found"

# Set environment variables
ENV GRPC_ROOT=/opt/grpc
ENV RDC_LIB_DIR=/opt/rocm/lib/rdc
ENV CMAKE_ROOT=/usr/bin/cmake

# Install a newer version of CMake
RUN apt-get remove -y cmake && \
    apt-get update && apt-get install -y wget && \
    wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh && \
    chmod +x cmake-3.26.4-linux-x86_64.sh && \
    ./cmake-3.26.4-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.26.4-linux-x86_64.sh

# Install gRPC
RUN git clone -b v1.61.0 https://github.com/grpc/grpc --depth=1 --shallow-submodules --recurse-submodules && \
    cd grpc && \
    cmake -B build \
        -DgRPC_INSTALL=ON \
        -DgRPC_BUILD_TESTS=OFF \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_PREFIX="/opt/grpc" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_BUILD_TYPE=Release && \
    make -C build -j $(nproc) && \
    make -C build install && \
    echo "/opt/grpc" | tee /etc/ld.so.conf.d/grpc.conf && \
    ldconfig

# Build and install RDC
RUN git clone https://github.com/ROCm/rdc && \
    cd rdc && \
    cmake \
        -B build \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_BUILD_TYPE=Release \
        -DGRPC_DESIRED_VERSION=1.61.0 \
        -DGRPC_ROOT="/opt/grpc" \
        -DBUILD_RUNTIME=ON \
        -DGPU_TARGETS=gfx942 \
        -DBUILD_PROFILER=OFF \
        -DBUILD_RVS=OFF \
        -DBUILD_TESTS=ON \
        -DCPACK_GENERATOR="DEB" \
        -DCMAKE_INSTALL_PREFIX="/opt/rocm" && \
    make -C build -j $(nproc) && \
    make -C build -j $(nproc) package && \
    make -C build -j $(nproc) install

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
