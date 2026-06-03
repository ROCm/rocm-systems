#! /usr/bin/env bash

set -x

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <np> <msg_size> [<build-flag> [<gpu-arch>]]]"
  exit 1
fi

NP=${1}
MSG_SIZE=${2}
BUILD_FLAG=${3:-false}
# TARGET_GPU_ARCH=${4:-gfx942}
TARGET_GPU_ARCH=${4:-gfx950}

DOCKERFILE="dockerfile-gin-fast-path"
DOCKER_IMAGE="gin-fast-path:latest"


# --- build
if ${BUILD_FLAG}; then
  # docker build -f dockerfile-gin-sdma -t gin-sdma:latest .
  # docker build -f Dockerfile-rccl-gin-gda -t rccl-gingda --build-arg GPU_TARGETS=gfx950  .
  docker rmi --force ${DOCKER_IMAGE}
  N=1
  docker build -f ${DOCKERFILE} -t ${DOCKER_IMAGE} --build-arg GPU_TARGETS=${TARGET_GPU_ARCH} --no-cache --build-arg USE_LOCAL_SRC=1 --build-arg ROCSHMEM_CACHE_BUST=$((N++)) .
fi

# --- run test
# Use double quotes for bash -lc so ${NP} and message size expand on the host.
# (Inside single quotes, ${NP} is passed literally and mpirun fails.)
#
docker run ${DOCKER_IMAGE} bash -lc "
    pwd 
    ls /workspace
    ls /workspace/rocshmem
  "
if [ 0 -eq 1 ]; then
#####
docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video --security-opt seccomp=unconfined \
  ${DOCKER_IMAGE} bash -lc "
    export LD_LIBRARY_PATH=/workspace/rocshmem/lib:/opt/ucx/lib:/opt/ompi/lib:/opt/rocm/lib:/opt/rocm/core/lib/rocm_sysdeps/lib
    export ROCSHMEM_BACKEND=ipc ROCSHMEM_SDMA_ENABLED=0
    export ROCSHMEM_DEBUG_LEVEL=NONE
    mpirun -np ${NP} --allow-run-as-root --timeout 200 \
      /workspace/rocshmem/bin/rocshmem_functional_tests \
      -a 19 -w 1 -z 8 -localbuftype heap -s ${MSG_SIZE} -n 100 -noverif \
      2>&1 |grep -v '\[ipc_resolve\]'
  "

#####
docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video --security-opt seccomp=unconfined \
  ${DOCKER_IMAGE} bash -lc "
    export LD_LIBRARY_PATH=/workspace/rocshmem/lib:/opt/ucx/lib:/opt/ompi/lib:/opt/rocm/lib:/opt/rocm/core/lib/rocm_sysdeps/lib
    export ROCSHMEM_BACKEND=ipc ROCSHMEM_SDMA_ENABLED=1
    export ROCSHMEM_DEBUG_LEVEL=NONE
    mpirun -np ${NP} --allow-run-as-root --timeout 200 \
      /workspace/rocshmem/bin/rocshmem_functional_tests \
      -a 19 -w 1 -z 8 -localbuftype heap -s ${MSG_SIZE} -n 100 -noverif \
      2>&1 |grep -v '\[ipc_resolve\]'
  "

#####
docker run -it --rm --shm-size 64G   --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband --ipc host   --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged  ${DOCKER_IMAGE} mpirun -n ${NP} -x ROCSHMEM_TEST_UUID=1 -x ROCSHMEM_BACKEND=ipc -x ROCSHMEM_SDMA_ENABLED=1 -x ROCSHMEM_DEBUG_LEVEL=info:noversion /workspace/rocshmem/bin/rocshmem_functional_tests -a 19 -w 1 -z 64 -v $((${NP}*${MSG_SIZE})) -n 100 -noverif 

fi


if [ 0 -eq 1 ]; then
#####
# rccl a2a gin host-proxy
docker run -it --rm --shm-size 64G   --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband --ipc host   --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged  ${DOCKER_IMAGE} mpirun -n ${NP} -mca pml ob1 -mca btl ^openib -x RCCL_ROCSHMEM_ENABLE=0 -x RCCL_ROCSHMEM_THRESHOLD=$((${NP}*${MSG_SIZE})) -x NCCL_DEBUG=VERSION -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=2 -x NCCL_DEBUG_SUBSYS=INIT,NET -x NONCCL_CUMEM_ENABLE=1 -x NORCCL_ENABLE_INTRANET=1  -x NCCL_DMABUF_ENABLE=1 -x NCCL_MSCCL_ENABLE=0 -x HSA_NO_SCRATCH_RECLAIM=1 /workspace/rccl-tests/alltoall_perf -b 128 -e 128M -f 2 -g 1 -R 2 -D 0 -A 1 

# rccl a2a gin host-proxy + rocshmem-sdma (slow)
docker run -it --rm --shm-size 64G   --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband --ipc host   --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged  ${DOCKER_IMAGE} mpirun -n ${NP} -mca pml ob1 -mca btl ^openib -x RCCL_ROCSHMEM_ENABLE=1 -x RCCL_ROCSHMEM_THRESHOLD=$((${NP}*${MSG_SIZE})) -x NCCL_DEBUG=VERSION -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=2 -x NCCL_DEBUG_SUBSYS=INIT,NET -x NONCCL_CUMEM_ENABLE=1 -x NORCCL_ENABLE_INTRANET=1  -x NCCL_DMABUF_ENABLE=1 -x NCCL_MSCCL_ENABLE=0 -x HSA_NO_SCRATCH_RECLAIM=1 /workspace/rccl-tests/alltoall_perf -b 128 -e 128M -f 2 -g 1 -R 2 -D 0 -A 1 
fi

# rocshmem-a2a-sdma (for reference, fast)
docker run -it --rm --shm-size 64G   --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband --ipc host   --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged  ${DOCKER_IMAGE} mpirun -n ${NP} -mca pml ob1 -mca btl ^openib /workspace/rocshmem/bin/rocshmem_functional_tests -a 19 -w 1 -z 256 -v $((${NP}*${MSG_SIZE})) -n 100 -noverif 

# rccl-a2a gin host-proxy on a single node
docker run -it --rm --shm-size 64G --network host \
  --device /dev/dri --device /dev/kfd --device /dev/infiniband \
  --ipc host --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  ${DOCKER_IMAGE} \
  mpirun -n ${NP} -mca pml ob1 -mca btl ^openib \
  -x NCCL_CUMEM_ENABLE=1 \
  -x RCCL_ENABLE_INTRANET=1 \
  -x NCCL_DMABUF_ENABLE=1 \
  -x HSA_NO_SCRATCH_RECLAIM=1 \
  -x LD_LIBRARY_PATH=/workspace/rocshmem/lib:/workspace/rccl/lib:/opt/ucx/lib:/opt/ompi/lib:/opt/rocm/lib:/opt/rocm/core/lib/rocm_sysdeps/lib \
  /workspace/rccl-tests/alltoall_perf \
  -b 128 -e $((${NP}*${MSG_SIZE})) -f 2 -g 1 -R 2 -D 1 -A 1 

# --- Crash: rccl-a2a gin rocshmem-sdma on a single node
docker run -it --rm --shm-size 64G --network host \
  --device /dev/dri --device /dev/kfd --device /dev/infiniband \
  --ipc host --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  gin-fast-path:latest \
  mpirun -n ${NP} -mca pml ob1 -mca btl ^openib \
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
  -x LD_LIBRARY_PATH=/workspace/rocshmem/lib:/workspace/rccl/lib:/opt/ucx/lib:/opt/ompi/lib:/opt/rocm/lib:/opt/rocm/core/lib/rocm_sysdeps/lib \
  /workspace/rccl-tests/alltoall_perf \
  -b 128 -e $((${NP}*${MSG_SIZE})) -f 2 -g 1 -R 2 -D 3 -A 1 

if [ 0 -eq 1 ]; then
# rccl-a2a on a single node
docker run -it --rm --shm-size 64G --network host \
  --device /dev/dri --device /dev/kfd --device /dev/infiniband \
  --ipc host --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  gin-fast-path:latest \
  mpirun -n ${NP} -mca pml ob1 -mca btl ^openib \
  -x RCCL_ROCSHMEM_ENABLE=1 \
  -x ROCSHMEM_BACKEND=ipc \
  -x ROCSHMEM_HEAP_SIZE=1073741824 \
  -x NCCL_GIN_ENABLE=1 \
  -x NCCL_GIN_TYPE=4 \
  -x NCCL_CUMEM_ENABLE=1 \
  -x RCCL_ENABLE_INTRANET=1 \
  -x NCCL_DMABUF_ENABLE=1 \
  -x NCCL_MSCCL_ENABLE=0 \
  -x HSA_NO_SCRATCH_RECLAIM=1 \
  -x NCCL_DEBUG=WARN \
  -x LD_LIBRARY_PATH=/workspace/rocshmem/lib:/workspace/rccl/lib:/opt/ucx/lib:/opt/ompi/lib:/opt/rocm/lib:/opt/rocm/core/lib/rocm_sysdeps/lib \
  /workspace/rccl-tests/alltoall_perf \
  -b 128 -e 128M -f 2 -g 1 -R 2 -D 4 -A 1 \
	2>&1 |grep -v '\[ipc_resolve\]'
fi

if [ 0 -eq 1 ]; then
# rccl-a2a-ginproxy (crash due to using wrong rocm version)
docker run -it --rm --shm-size 64G   --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband --ipc host   --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged  ${DOCKER_IMAGE} mpirun -n ${NP} -mca pml ob1 -mca btl ^openib -x RCCL_ROCSHMEM_ENABLE=0 -x RCCL_ROCSHMEM_THRESHOLD=$((${NP}*${MSG_SIZE})) -x NCCL_DEBUG=WARN -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=2 -x NCCL_DEBUG_SUBSYS=INIT,NET -x NCCL_CUMEM_ENABLE=1 -x RCCL_ENABLE_INTRANET=1  -x NCCL_DMABUF_ENABLE=1 -x NCCL_MSCCL_ENABLE=0 -x HSA_NO_SCRATCH_RECLAIM=1 /workspace/rccl-tests/alltoall_perf -b 128 -e 128M -f 2 -g 1 -R 2 -D 3 -A 1 \
	2>&1 |grep -v '\[ipc_resolve\]'
fi

set +x

