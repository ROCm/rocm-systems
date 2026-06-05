#! /usr/bin/env bash

set -euxo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <np> <msg_size> [<build-flag> [<gpu-arch>]]"
  exit 1
fi

NP=${1}
MSG_SIZE=${2}
BUILD_FLAG=${3:-false}
# TARGET_GPU_ARCH=${4:-gfx942}
TARGET_GPU_ARCH=${4:-gfx950}

DOCKERFILE="Dockerfile-rccl-gin-anvil"
DOCKER_IMAGE="gin-anvil:latest"

# Derived sizes / shared docker+mpirun settings (expand on host, not inside container).
MAX_BYTES=$((${NP} * ${MSG_SIZE}))
# No -it: script is often run over non-interactive SSH.
DOCKER_GPU="--rm --shm-size 64G --network host --device /dev/dri --device /dev/kfd --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged"
RCCL_LD_PATH="/workspace/rocshmem/lib:/workspace/rccl/lib:/opt/ucx/lib:/opt/ompi/lib:/opt/rocm/lib:/opt/rocm/core/lib/rocm_sysdeps/lib"
MPIRUN_BASE="-n ${NP} --allow-run-as-root -mca pml ob1 -mca btl ^openib"

# --- build
if ${BUILD_FLAG}; then
  N=1
  docker build -f ${DOCKERFILE} -t ${DOCKER_IMAGE} \
    --no-cache \
    --build-arg GPU_TARGETS=${TARGET_GPU_ARCH} \
    --build-arg USE_LOCAL_SRC=1 \
    --build-arg ROCSHMEM_CACHE_BUST=$((N++)) .
  docker image inspect "${DOCKER_IMAGE}" >/dev/null
fi

# --- sanity: image layout
docker run --rm ${DOCKER_IMAGE} bash -lc "
  echo '=== workspace ==='
  pwd
  ls -la /workspace
  ls -la /workspace/rocshmem/bin 2>/dev/null || true
  ls -la /workspace/rccl/lib 2>/dev/null || true
  ls -la /workspace/rccl-tests/alltoall_perf 2>/dev/null || true
"

if [ 1 -eq 1 ]; then
#####
# rocSHMEM IPC alltoall (reference)
echo "=== rocSHMEM IPC alltoall np=${NP} max_bytes=${MAX_BYTES} ==="
docker run ${DOCKER_GPU} ${DOCKER_IMAGE} \
  mpirun ${MPIRUN_BASE} \
  -x ROCSHMEM_TEST_UUID=1 \
  -x ROCSHMEM_BACKEND=ipc \
  -x ROCSHMEM_SDMA_ENABLED=1 \
  -x ROCSHMEM_DEBUG_LEVEL=info:noversion \
  /workspace/rocshmem/bin/rocshmem_functional_tests \
  -a 19 -w 1 -z 256 -v ${MAX_BYTES} -n 100 -noverif
fi

if [ 1 -eq 1 ]; then
#####
# RCCL AlltoAll: GIN proxy (NCCL_GIN_TYPE=2, inter-node capable)
echo "=== RCCL AlltoAll: GIN proxy (NCCL_GIN_TYPE=2, inter-node capable) np=${NP} max_bytes=${MAX_BYTES} ==="
docker run ${DOCKER_GPU} ${DOCKER_IMAGE} \
  mpirun ${MPIRUN_BASE} \
  -x RCCL_ROCSHMEM_ENABLE=0 \
  -x NCCL_DEBUG=VERSION \
  -x NCCL_GIN_ENABLE=1 \
  -x NCCL_GIN_TYPE=2 \
  -x NCCL_DEBUG_SUBSYS=INIT,NET \
  -x NCCL_CUMEM_ENABLE=1 \
  -x RCCL_ENABLE_INTRANET=1 \
  -x NCCL_DMABUF_ENABLE=1 \
  -x NCCL_MSCCL_ENABLE=0 \
  -x HSA_NO_SCRATCH_RECLAIM=1 \
  -x LD_LIBRARY_PATH=${RCCL_LD_PATH} \
  /workspace/rccl-tests/alltoall_perf \
  -b 128 -e ${MAX_BYTES} -f 2 -g 1 -R 0 -D 0 -A 1
fi

if [ 1 -eq 1 ]; then
#####
# RCCL AlltoAll: GIN proxy (NCCL_GIN_TYPE=2, intra-node only)
echo "=== RCCL AlltoAll: GIN proxy (NCCL_GIN_TYPE=2, intra-node only) np=${NP} max_bytes=${MAX_BYTES} ==="
docker run ${DOCKER_GPU} ${DOCKER_IMAGE} \
  mpirun ${MPIRUN_BASE} \
  -x RCCL_ROCSHMEM_ENABLE=0 \
  -x NCCL_DEBUG=VERSION \
  -x NCCL_GIN_ENABLE=1 \
  -x NCCL_GIN_TYPE=2 \
  -x NCCL_DEBUG_SUBSYS=INIT,NET \
  -x NCCL_CUMEM_ENABLE=1 \
  -x RCCL_ENABLE_INTRANET=1 \
  -x NCCL_DMABUF_ENABLE=1 \
  -x NCCL_MSCCL_ENABLE=0 \
  -x HSA_NO_SCRATCH_RECLAIM=1 \
  -x LD_LIBRARY_PATH=${RCCL_LD_PATH} \
  /workspace/rccl-tests/alltoall_perf \
  -b 128 -e ${MAX_BYTES} -f 2 -g 1 -R 2 -D 1 -A 1
fi

if [ 1 -eq 1 ]; then
#####
# RCCL AlltoAll: GIN_ROCSHMEM (NCCL_GIN_TYPE=4) + rocSHMEM SDMA path
echo "=== RCCL AlltoAll: GIN_ROCSHMEM (NCCL_GIN_TYPE=4) + rocSHMEM SDMA np=${NP} max_bytes=${MAX_BYTES} ==="
docker run ${DOCKER_GPU} ${DOCKER_IMAGE} \
  mpirun ${MPIRUN_BASE} \
  -x RCCL_ROCSHMEM_ENABLE=1 \
  -x ROCSHMEM_BACKEND=ipc \
  -x ROCSHMEM_HEAP_SIZE=1073741824 \
  -x ROCSHMEM_SDMA_ENABLED=1 \
  -x NCCL_GIN_ENABLE=1 \
  -x NCCL_GIN_TYPE=4 \
  -x NCCL_CUMEM_ENABLE=1 \
  -x RCCL_ENABLE_INTRANET=1 \
  -x NCCL_DMABUF_ENABLE=1 \
  -x NCCL_MSCCL_ENABLE=0 \
  -x HSA_NO_SCRATCH_RECLAIM=1 \
  -x LD_LIBRARY_PATH=${RCCL_LD_PATH} \
  /workspace/rccl-tests/alltoall_perf \
  -b 128 -e ${MAX_BYTES} -f 2 -g 1 -R 2 -D 3 -A 1
fi

if [ 1 -eq 1 ]; then
# --- RCCL AlltoAll with GIN_ANVIL (NCCL_GIN_TYPE=5, intra-node MI300 xGMI SDMA)
# Matches Dockerfile-rccl-gin-anvil example; single-node only (no IB device required).
echo "=== RCCL AlltoAll: GIN_ANVIL (NCCL_GIN_TYPE=5) np=${NP} max_bytes=${MAX_BYTES} ==="
docker run ${DOCKER_GPU} ${DOCKER_IMAGE} \
  mpirun ${MPIRUN_BASE} \
  -x RCCL_TEST_SKIP_ROCSHMEM_PREINIT=1 \
  -x RCCL_ROCSHMEM_ENABLE=0 \
  -x NCCL_GIN_ENABLE=1 \
  -x NCCL_GIN_TYPE=5 \
  -x NCCL_DEBUG=VERSION \
  -x NCCL_DEBUG_SUBSYS=INIT,NET \
  -x NCCL_CUMEM_ENABLE=1 \
  -x RCCL_ENABLE_INTRANET=1 \
  -x NCCL_DMABUF_ENABLE=1 \
  -x NCCL_MSCCL_ENABLE=0 \
  -x HSA_NO_SCRATCH_RECLAIM=1 \
  -x LD_LIBRARY_PATH=${RCCL_LD_PATH} \
  /workspace/rccl-tests/alltoall_perf \
  -b 128 -e ${MAX_BYTES} -f 2 -g 1 -R 2 -D 3 -A 1
fi

set +x
