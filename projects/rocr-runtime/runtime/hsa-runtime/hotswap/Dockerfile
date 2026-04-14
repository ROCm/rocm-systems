FROM rocm/dev-ubuntu-22.04:7.2.1

RUN apt-get update && \
    apt-get install -y --no-install-recommends cmake ninja-build python3 rocm-llvm-dev libzstd-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/projects/rocr-runtime/runtime/hsa-runtime/hotswap

CMD ["sleep", "infinity"]
