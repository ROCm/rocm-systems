ARG BASE_IMAGE=rocm/rocm-terminal
FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive

ARG BRANCH=amd-staging
# Unprivileged account of the base image (rocm/rocm-terminal runs as rocm-user)
ARG BASE_USER=rocm-user

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends libdw-dev libsqlite3-dev && \
    rm -rf /var/lib/apt/lists/*

USER ${BASE_USER}
RUN git clone -b ${BRANCH} https://github.com/ROCm/rocprofiler-sdk.git rocprofiler-sdk-source && \
    python3 -m pip install -r rocprofiler-sdk-source/requirements.txt && \
    cmake -B rocprofiler-sdk-build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DROCPROFILER_BUILD_{SAMPLES,TESTS,BENCHMARK}=ON -DPython3_EXECUTABLE=$(which python3) -DCMAKE_INSTALL_PREFIX=$(realpath /opt/rocm) rocprofiler-sdk-source && \
    cmake --build rocprofiler-sdk-build --target all --parallel 16

USER root
RUN cmake --build rocprofiler-sdk-build --target install && \
    rm -rf rocprofiler-sdk-source rocprofiler-sdk-build

USER ${BASE_USER}
