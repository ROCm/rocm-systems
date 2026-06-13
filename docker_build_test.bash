#! /usr/bin/env bash

set -euxo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <nnodes> <ppn> <msg_size> [<build-flag> [<gpu-arch>]]"
  exit 1
fi

NNODES=${1}
PPN=${2}
NP=$(( ${NNODES} * ${PPN} ))
MSG_SIZE=${3}
BUILD_FLAG=${4:-false}
TARGET_GPU_ARCH=${5:-gfx950}
# TARGET_GPU_ARCH=${5:-gfx942}

DOCKERFILE="Dockerfile-rccl-gin-anvil"
DOCKER_IMAGE="gin-anvil:latest"

# Derived sizes / shared docker+mpirun settings (expand on host, not inside container).
MAX_BYTES=$((${NP} * ${MSG_SIZE}))
# No -it: script is often run over non-interactive SSH.
# --init: PID 1 reaps children so ranks exit more cleanly (reduces NCCL IPC/socket teardown WARNs).
DOCKER_GPU="--rm --init --shm-size 64G --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged"
RCCL_LD_PATH="/workspace/rocshmem/lib:/workspace/rccl/lib:/opt/ucx/lib:/opt/ompi/lib:/opt/rocm/lib:/opt/rocm/core/lib/rocm_sysdeps/lib"
HFILE="my_hostfile"
MPIRUN_BASE="-n ${NP} --allow-run-as-root -mca pml ob1 -mca btl ^openib"
MPIRUN_BASE_HFILE="-n ${NP} --hostfile /workspace/${HFILE} --allow-run-as-root -mca pml ob1 -mca btl ^openib"
# Quiets RCCL init.cc when built without HIP_UNCACHED_MEMORY. NCCL_DEBUG=VERSION avoids printing
# NCCL_LOG_WARN teardown lines (e.g. socket/IPC deregister) that appear when NCCL_DEBUG=WARN is set.
# Avoid a duplicate -D 0 run mixing NCCL_GIN_ENABLE=1 with -D 0 and NCCL_DEBUG=WARN (see ddai-a2a-1gb-perf-try2.log).
RCCL_ENV_COMMON="-x HSA_FORCE_FINE_GRAIN_PCIE=1 -x NCCL_DEBUG=VERSION"

# rccl-tests alltoall_perf: -R is local_register (0=off, 1=local, 2=symmetric ncclCommWindowRegister).
# common.cu requires -R 2 whenever -D>0 (device/GIN kernels use ncclWindow_t from symmetric collective windows).
# GIN_ANVIL (NCCL_GIN_TYPE=5, -D 5) relies on that path; use the same -R for host -D 0 baselines so large-message
# numbers are comparable to symmetric-buffer runs (e.g. tuned -D 0 with -R 2), not dominated by unregistered buffers.

if [ -x scontrol ]; then
    scontrol show hostnames "$SLURM_JOB_NODELIST" | awk '{print $1 " slots='${PPN}'"}' > ${HFILE}
else
    echo "$(hostname) slots=${PPN}" > ${HFILE}
fi

# Mount hostfile so mpirun sees current nodes without rebuilding the image.
DOCKER_GPU="${DOCKER_GPU} -v $(pwd)/${HFILE}:/workspace/${HFILE}:ro"

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
  ls -la /workspace 2>/dev/null || true
  cat /workspace/my_hostfile 2>/dev/null || true
  ls -la /workspace/rocshmem/bin 2>/dev/null || true
  ls -la /workspace/rccl/lib 2>/dev/null || true
  ls -la /workspace/rccl-tests/alltoall_perf 2>/dev/null || true
  # echo '=== alltoall_perf anvil symbol export ==='
  # nm -D /workspace/rccl-tests/alltoall_perf 2>/dev/null | grep anvil || echo 'MISSING anvil symbols'
"

if [ 1 -eq 1 ]; then
#####
# rocSHMEM IPC alltoall (reference)
echo "=== rocSHMEM IPC alltoall np=${NP} max_bytes=${MAX_BYTES} ==="
set -x
docker run ${DOCKER_GPU} ${DOCKER_IMAGE} \
  mpirun ${MPIRUN_BASE} \
  -x ROCSHMEM_TEST_UUID=1 \
  -x ROCSHMEM_BACKEND=ipc \
  -x ROCSHMEM_SDMA_ENABLED=1 \
  -x ROCSHMEM_DEBUG_LEVEL=info:noversion \
  /workspace/rocshmem/bin/rocshmem_functional_tests \
  -a 19 -w 1 -z 256 -v ${MAX_BYTES} -n 100 -noverif
set +x
fi

if [ 1 -eq 1 ]; then
#####
# RCCL AlltoAll: -D 0, (host-initiated, inter-node capable)
echo "=== RCCL AlltoAll: -D 0, non-GIN (inter-node capable) np=${NP} max_bytes=${MAX_BYTES} ==="
set -x
docker run ${DOCKER_GPU} ${DOCKER_IMAGE} \
  mpirun ${MPIRUN_BASE} \
  ${RCCL_ENV_COMMON} \
  -x RCCL_ROCSHMEM_ENABLE=0 \
  -x NCCL_GIN_ENABLE=0 \
  -x NCCL_GIN_TYPE=0 \
  -x NCCL_DEBUG_SUBSYS=INIT,NET \
  -x NCCL_CUMEM_ENABLE=1 \
  -x RCCL_ENABLE_INTRANET=1 \
  -x NCCL_DMABUF_ENABLE=1 \
  -x NCCL_MSCCL_ENABLE=0 \
  -x HSA_NO_SCRATCH_RECLAIM=1 \
  -x LD_LIBRARY_PATH=${RCCL_LD_PATH} \
  /workspace/rccl-tests/alltoall_perf \
  -b 128 -e ${MAX_BYTES} -f 2 -g 1 -R 2 -D 0 -A 1
set +x
fi

if [ 0 -eq 1 ]; then
#####
# RCCL AlltoAll: -D 2, GIN host proxy (NCCL_GIN_TYPE=2, intra-node only)
echo "=== RCCL AlltoAll: -D 2, GIN host proxy (NCCL_GIN_TYPE=2, intra-node only) np=${NP} max_bytes=${MAX_BYTES} ==="
set -x
docker run ${DOCKER_GPU} ${DOCKER_IMAGE} \
  mpirun ${MPIRUN_BASE} \
  ${RCCL_ENV_COMMON} \
  -x NCCL_DEBUG=VERSION \
  -x RCCL_ROCSHMEM_ENABLE=0 \
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
set +x
fi

# Optional: NCCL_GIN_TYPE=2 host-proxy + -D 3 needs a working external/IB GIN plugin; gin-anvil
# typically validates TYPE 4/5. 
if [ 1 -eq 1 ]; then
#####
# RCCL AlltoAll: -D 3, GIN host proxy (NCCL_GIN_TYPE=2, intra-node only)
echo "=== RCCL AlltoAll: -D 3, GIN host proxy (NCCL_GIN_TYPE=2, intra-node only) np=${NP} max_bytes=${MAX_BYTES} ==="
set -x
docker run ${DOCKER_GPU} ${DOCKER_IMAGE} \
  mpirun ${MPIRUN_BASE} \
  ${RCCL_ENV_COMMON} \
  -x NCCL_DEBUG=VERSION \
  -x RCCL_ROCSHMEM_ENABLE=0 \
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
  -b 128 -e ${MAX_BYTES} -f 2 -g 1 -R 2 -D 3 -A 1
set +x
fi


if [ 0 -eq 1 ]; then
#####
# RCCL AlltoAll: -D 3, GIN_ROCSHMEM (NCCL_GIN_TYPE=4) + rocSHMEM SDMA path
echo "=== RCCL AlltoAll: -D 3, GIN_ROCSHMEM (NCCL_GIN_TYPE=4) + rocSHMEM SDMA np=${NP} max_bytes=${MAX_BYTES} ==="
set -x
docker run ${DOCKER_GPU} ${DOCKER_IMAGE} \
  mpirun ${MPIRUN_BASE} \
  -x RCCL_ROCSHMEM_ENABLE=0 \
  -x NCCL_GIN_ENABLE=1 \
  -x ROCSHMEM_BACKEND=ipc \
  -x ROCSHMEM_HEAP_SIZE=1073741824 \
  -x ROCSHMEM_SDMA_ENABLED=1 \
  -x NCCL_GIN_TYPE=4 \
  -x NCCL_DEBUG=VERSION \
  -x NCCL_DEBUG_SUBSYS=INIT \
  -x NCCL_CUMEM_ENABLE=1 \
  -x RCCL_ENABLE_INTRANET=1 \
  -x NCCL_DMABUF_ENABLE=1 \
  -x NCCL_MSCCL_ENABLE=0 \
  -x HSA_NO_SCRATCH_RECLAIM=1 \
  -x LD_LIBRARY_PATH=${RCCL_LD_PATH} \
  /workspace/rccl-tests/alltoall_perf \
  -b 128 -e ${MAX_BYTES} -f 2 -g 1 -R 2 -D 3 -A 1
set +x
fi

if [ 1 -eq 1 ]; then
#####
# RCCL AlltoAll: -D 4, GIN_ROCSHMEM (NCCL_GIN_TYPE=4) + rocSHMEM SDMA path
echo "=== RCCL AlltoAll: -D 4, GIN_ROCSHMEM (NCCL_GIN_TYPE=4) + rocSHMEM SDMA np=${NP} max_bytes=${MAX_BYTES} ==="
set -x
docker run ${DOCKER_GPU} ${DOCKER_IMAGE} \
  mpirun ${MPIRUN_BASE} \
  ${RCCL_ENV_COMMON} \
  -x RCCL_ROCSHMEM_ENABLE=0 \
  -x NCCL_GIN_ENABLE=1 \
  -x ROCSHMEM_BACKEND=ipc \
  -x ROCSHMEM_HEAP_SIZE=1073741824 \
  -x ROCSHMEM_SDMA_ENABLED=1 \
  -x NCCL_GIN_TYPE=4 \
  -x NCCL_DEBUG_SUBSYS=INIT \
  -x NCCL_CUMEM_ENABLE=1 \
  -x RCCL_ENABLE_INTRANET=1 \
  -x NCCL_DMABUF_ENABLE=1 \
  -x NCCL_MSCCL_ENABLE=0 \
  -x HSA_NO_SCRATCH_RECLAIM=1 \
  -x LD_LIBRARY_PATH=${RCCL_LD_PATH} \
  /workspace/rccl-tests/alltoall_perf \
  -b 128 -e ${MAX_BYTES} -f 2 -g 1 -R 2 -D 4 -A 1
set +x
fi

if [ 0 -eq 1 ]; then
# --- RCCL AlltoAll with GIN_ANVIL (NCCL_GIN_TYPE=5, intra-node MI300 xGMI SDMA)
# Matches Dockerfile-rccl-gin-anvil example; single-node only (no IB device required).
echo "=== RCCL AlltoAll: -D 3, GIN_ANVIL (NCCL_GIN_TYPE=5) np=${NP} max_bytes=${MAX_BYTES} ==="
set -x
docker run ${DOCKER_GPU} ${DOCKER_IMAGE} \
  mpirun ${MPIRUN_BASE} \
  -x RCCL_ROCSHMEM_ENABLE=0 \
  -x NCCL_GIN_ENABLE=1 \
  -x NCCL_GIN_TYPE=5 \
  -x NCCL_DEBUG=VERSION \
  -x NCCL_DEBUG_SUBSYS=INIT \
  -x NCCL_CUMEM_ENABLE=1 \
  -x RCCL_ENABLE_INTRANET=1 \
  -x NCCL_DMABUF_ENABLE=1 \
  -x NCCL_MSCCL_ENABLE=0 \
  -x HSA_NO_SCRATCH_RECLAIM=1 \
  -x LD_LIBRARY_PATH=${RCCL_LD_PATH} \
  /workspace/rccl-tests/alltoall_perf \
  -b 128 -e ${MAX_BYTES} -f 2 -g 1 -R 2 -D 3 -A 1
set +x
fi

if [ 0 -eq 1 ]; then
# --- RCCL AlltoAll with GIN_ANVIL (NCCL_GIN_TYPE=5, intra-node MI300 xGMI SDMA)
# Matches Dockerfile-rccl-gin-anvil example; single-node only (no IB device required).
echo "=== RCCL AlltoAll: -D 4, GIN_ANVIL (NCCL_GIN_TYPE=5) np=${NP} max_bytes=${MAX_BYTES} ==="
set -x
docker run ${DOCKER_GPU} ${DOCKER_IMAGE} \
  mpirun ${MPIRUN_BASE} \
  -x RCCL_ROCSHMEM_ENABLE=0 \
  -x NCCL_GIN_ENABLE=1 \
  -x NCCL_GIN_TYPE=5 \
  -x NCCL_DEBUG=VERSION \
  -x NCCL_DEBUG_SUBSYS=INIT \
  -x NCCL_CUMEM_ENABLE=1 \
  -x RCCL_ENABLE_INTRANET=1 \
  -x NCCL_DMABUF_ENABLE=1 \
  -x NCCL_MSCCL_ENABLE=0 \
  -x HSA_NO_SCRATCH_RECLAIM=1 \
  -x LD_LIBRARY_PATH=${RCCL_LD_PATH} \
  /workspace/rccl-tests/alltoall_perf \
  -b 128 -e ${MAX_BYTES} -f 2 -g 1 -R 2 -D 4 -A 1
set +x
fi

if [ 1 -eq 1 ]; then
# --- RCCL AlltoAll with GIN_ANVIL (NCCL_GIN_TYPE=5, intra-node MI300 xGMI SDMA)
# Symmetric collective windows (-R 2) are required for -D 5 and feed ncclGinAnvilRegister LSA resolution.
# Single-node -D 5 uses cooperative AlltoAllLsaCopy for large slices; do not raise -V beyond defaults
# without matching devComm barrier counts — oversubscribing CTAs regresses badly on MI355-class nodes.
# Matches Dockerfile-rccl-gin-anvil example; single-node only (no IB device required).
echo "=== RCCL AlltoAll: -D 5, GIN_ANVIL (NCCL_GIN_TYPE=5) np=${NP} max_bytes=${MAX_BYTES} ==="
set -x
docker run ${DOCKER_GPU} ${DOCKER_IMAGE} \
  mpirun ${MPIRUN_BASE} \
  ${RCCL_ENV_COMMON} \
  -x RCCL_ROCSHMEM_ENABLE=0 \
  -x NCCL_GIN_ENABLE=1 \
  -x NCCL_GIN_TYPE=5 \
  -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=4 \
  -x NCCL_GIN_ANVIL_SDMA_CHUNK_MB=16 \
  -x NCCL_DEBUG_SUBSYS=INIT \
  -x NCCL_CUMEM_ENABLE=1 \
  -x RCCL_ENABLE_INTRANET=1 \
  -x NCCL_DMABUF_ENABLE=1 \
  -x NCCL_MSCCL_ENABLE=0 \
  -x HSA_NO_SCRATCH_RECLAIM=1 \
  -x LD_LIBRARY_PATH=${RCCL_LD_PATH} \
  /workspace/rccl-tests/alltoall_perf \
  -b 128 -e ${MAX_BYTES} -f 2 -g 1 -R 2 -D 5 -A 1
set +x
fi

if [ 0 -eq 1 ]; then
echo "=== RCCL AlltoAll: -D 5, GIN_ANVIL (NCCL_GIN_TYPE=5, NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=2) np=${NP} max_bytes=${MAX_BYTES} ==="
set -x
docker run ${DOCKER_GPU} ${DOCKER_IMAGE} \
  mpirun ${MPIRUN_BASE} \
  -x RCCL_ROCSHMEM_ENABLE=0 \
  -x NCCL_GIN_ENABLE=1 \
  -x NCCL_GIN_TYPE=5 \
  -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=2 \
  -x NCCL_DEBUG=VERSION \
  -x NCCL_DEBUG_SUBSYS=INIT \
  -x NCCL_CUMEM_ENABLE=1 \
  -x RCCL_ENABLE_INTRANET=1 \
  -x NCCL_DMABUF_ENABLE=1 \
  -x HSA_NO_SCRATCH_RECLAIM=1 \
  -x LD_LIBRARY_PATH=${RCCL_LD_PATH} \
  /workspace/rccl-tests/alltoall_perf \
  -b 128 -e ${MAX_BYTES} -f 2 -g 1 -R 2 -D 5 -A 1
set +x
fi

#
if [ 1 -eq 1 ]; then
# Example: 2 nodes × 8 GPUs = 16 ranks (adjust -n and --hostfile)
#
# GIN_ANVIL (NCCL_GIN_TYPE=5) is single-node only (gin_host_anvil.cc). With nnodes>1, RCCL
# emits WARN + ncclInvalidUsage. Multi-node: run standard alltoall (-D 0) instead.
if [[ "${NNODES}" -gt 1 ]]; then
echo "=== RCCL AlltoAll: -D 0 (hostfile, nnodes=${NNODES} PPN=${PPN}; GIN_ANVIL skipped, single-node only) max_bytes=${MAX_BYTES} ==="
set -x
docker run ${DOCKER_GPU} ${DOCKER_IMAGE} \
  mpirun ${MPIRUN_BASE_HFILE} \
  ${RCCL_ENV_COMMON} \
  -x RCCL_ROCSHMEM_ENABLE=0 \
  -x NCCL_GIN_ENABLE=0 \
  -x NCCL_GIN_TYPE=0 \
  -x NCCL_DEBUG_SUBSYS=INIT,NET \
  -x NCCL_CUMEM_ENABLE=1 \
  -x RCCL_ENABLE_INTRANET=1 \
  -x NCCL_DMABUF_ENABLE=1 \
  -x NCCL_MSCCL_ENABLE=0 \
  -x HSA_NO_SCRATCH_RECLAIM=1 \
  -x LD_LIBRARY_PATH=${RCCL_LD_PATH} \
  /workspace/rccl-tests/alltoall_perf \
  -b 128 -e ${MAX_BYTES} -f 2 -g 1 -R 2 -D 0 -A 1
set +x
else
echo "=== RCCL AlltoAll: -D 5, GIN_ANVIL (NCCL_GIN_TYPE=5) nnodes=${NNODES} PPN=${PPN} max_bytes=${MAX_BYTES} ==="
set -x
docker run ${DOCKER_GPU} ${DOCKER_IMAGE} \
  mpirun ${MPIRUN_BASE_HFILE} \
  ${RCCL_ENV_COMMON} \
  -x RCCL_ROCSHMEM_ENABLE=0 \
  -x NCCL_GIN_ENABLE=1 \
  -x NCCL_GIN_TYPE=5 \
  -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=4 \
  -x NCCL_GIN_ANVIL_SDMA_CHUNK_MB=16 \
  -x NCCL_DEBUG_SUBSYS=INIT \
  -x NCCL_CUMEM_ENABLE=1 \
  -x RCCL_ENABLE_INTRANET=1 \
  -x NCCL_DMABUF_ENABLE=1 \
  -x NCCL_MSCCL_ENABLE=0 \
  -x HSA_NO_SCRATCH_RECLAIM=1 \
  -x LD_LIBRARY_PATH=${RCCL_LD_PATH} \
  /workspace/rccl-tests/alltoall_perf \
  -b 128 -e ${MAX_BYTES} -f 2 -g 1 -R 2 -D 5 -A 1
set +x
fi
fi

