ARG BASE_IMAGE=rocm/rocm-terminal
FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive

ARG BRANCH=amd-staging

RUN git clone -b ${BRANCH} https://github.com/ROCm/rocprofiler-sdk.git rocprofiler-sdk-source && \
    python3 -m pip install -r rocprofiler-sdk-source/requirements.txt && \
    sudo apt update && \
    sudo apt install -y libdw-dev && \
    cmake -B rocprofiler-sdk-build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DROCPROFILER_BUILD_{SAMPLES,TESTS,BENCHMARK}=ON -DPython3_EXECUTABLE=$(which python3) -DCMAKE_INSTALL_PREFIX=$(realpath /opt/rocm) rocprofiler-sdk-source && \
    cmake --build rocprofiler-sdk-build --target all --parallel 16 && \
    sudo cmake --build rocprofiler-sdk-build --target install && \
    sudo rm -rf rocprofiler-sdk-source rocprofiler-sdk-build
