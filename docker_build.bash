#! /usr/bin/env bash

set -euxo pipefail

if [[ $# -gt 4 ]]; then
  echo "Usage: $0  [<gpu-arch> [<local-src> [<dockerfile> [<docker-image>]]]]"
  exit 1
fi

TARGET_GPU_ARCH=${1:-gfx950}
# TARGET_GPU_ARCH=${1:-gfx942}
LOC_SRC=${2:-1}
DOCKERFILE=${3:-"Dockerfile-rccl-gin-anvil"}
DOCKER_IMAGE=${4:-"gin-anvil:latest"}

# --- build
N=1

docker build -f ${DOCKERFILE} -t ${DOCKER_IMAGE} \
    --no-cache \
    --build-arg GPU_TARGETS=${TARGET_GPU_ARCH} \
    --build-arg USE_LOCAL_SRC=${LOC_SRCC} \
    --build-arg ROCSHMEM_CACHE_BUST=$((N++)) .

docker image inspect "${DOCKER_IMAGE}" >/dev/null

