#! /usr/bin/env bash

# Test#5 (GIN Anvil-SDMA, NCCL_GIN_TYPE=5) needs rocSHMEM built with USE_SDMA=ON. Default
# ROCSHMEM_USE_SDMA=1 passes --build-arg to the Dockerfile; set ROCSHMEM_USE_SDMA=0 to opt out.
# The Dockerfile upgrades rdma-core/libmlx5 from ${VERSION_CODENAME}-updates when available.
# Optional CI: pass --build-arg RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS=1 to fail the build unless
# libmlx5/libmlx5dv export mlx5dv_reg_dmabuf_mr (MOFED / newer rdma-core). Default 0: stock Ubuntu 24.04
# often lacks those symbols (ddai-gin-build.log); test scripts can skip Test#5 via MLX5 preflight.
# Optional: export RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS=1 for strict MLX5 DMA-BUF symbol check at docker build.
# Optional GinAlltoAll trace (Test#5 debugging; both default off):
#   RCCL_GIN_ALLTOALL_HOST_TRACE=1   — runtime host stderr launch logs (no rebuild; set when running
#                                      docker-gin-gda-sdma-test.bash).
#   RCCL_GIN_ALLTOALL_DEVICE_TRACE=1 — compile GPU phase/signal markers into alltoall_perf; also set
#                                      RCCL_GIN_ALLTOALL_DEVICE_TRACE=1 at test time to poll them.
# ROCSHMEM_USE_SDMA="${ROCSHMEM_USE_SDMA:-1}"

DOCKER_NO_CACHE=${1:-0}
DOCKER_CMD="${DOCKER_CMD:-docker}"
DOCKERFILE="Dockerfile-rccl-gin-gda-sdma"
# Exported so the post-build smoke gate (gin-sdma-a2a-test.bash) runs against THIS
# just-built image rather than the harness's own default name.
export DOCKER_IMAGE="${DOCKER_IMAGE:-rccl-gin-gda-sdma-713}"
TARGET_GPU_ARCH="${GPU_TARGETS:-gfx950}"
USE_LOCAL_SRC=1
ROCSHMEM_USE_SDMA=1
RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS="${RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS:-0}"
# RCCL device-kernel set (ONLY_FUNCS). Includes Broadcast + AllGather + AllReduce so the host
# baselines (broadcast_perf/all_gather_perf/all_reduce_perf -D 0) work across the full size
# range. Set ONLY_FUNCS="" to build every collective (much slower), or override with a custom
# pattern.
# Fast-iteration knob: COLLECTIVE=a2a narrows the generated device-kernel set (ONLY_FUNCS)
# AND the post-build smoke gate to just AllToAll*, so the compile and device link only cover
# what the A2A GIN Anvil-SDMA tests exercise. An explicit ONLY_FUNCS (or RCCL_IMAGE_GIN_SMOKE)
# always wins. Unset (default) => full set + A2A smoke gate.
COLLECTIVE="${COLLECTIVE:-}"
case "${COLLECTIVE}" in
  a2a) : "${ONLY_FUNCS:=SendRecv|AlltoAllPivot|AlltoAllGda|AlltoAllvGda}"
       : "${RCCL_IMAGE_GIN_SMOKE:=1}" ;;
  "")  ;;
  *)   echo "WARN: unknown COLLECTIVE='${COLLECTIVE}' (use a2a); building full set" >&2 ;;
esac
ONLY_FUNCS="${ONLY_FUNCS-SendRecv|AlltoAllPivot|AlltoAllGda|AlltoAllvGda|Broadcast|AllGather|AllReduce|Reduce}"

GIT_CLONE_ROOT="${GIT_CLONE_ROOT:-$PWD}"
DEV_ARTI_DIR="${GIT_CLONE_ROOT}/ddai-artifacts"
DOCKERFILE_PATH="${DEV_ARTI_DIR}/docker/${DOCKERFILE}"

# Dockerfile COPY extra-rdma-debs/ requires the directory in build context (optional .deb install).
mkdir -p extra-rdma-debs

if [ "$DOCKER_NO_CACHE" -eq 1 ]; then
  DOCKER_CACHE_OPT="--no-cache"
  RCCL_CACHE_BUST=1
  ROCSHMEM_CACHE_BUST=1
else
  DOCKER_CACHE_OPT=""
fi

export RCCL_CACHE_BUST=${RCCL_CACHE_BUST:-1}
export ROCSHMEM_CACHE_BUST=${ROCSHMEM_CACHE_BUST:-1}

RCCL_GIN_ALLTOALL_DEVICE_TRACE="${RCCL_GIN_ALLTOALL_DEVICE_TRACE:-0}"

TRACE_BUILD_ARGS=()
if [[ "${RCCL_GIN_ALLTOALL_DEVICE_TRACE}" == 1 ]]; then
  TRACE_BUILD_ARGS+=(--build-arg RCCL_GIN_ALLTOALL_DEVICE_TRACE=1)
  echo "docker build: RCCL_GIN_ALLTOALL_DEVICE_TRACE=1 (GPU kernel phase markers)"
fi

# Optional network mode for RUN steps. On hosts whose docker daemon disables the
# default bridge (daemon.json "bridge":"none", e.g. some shared/slurm nodes), the
# default RUN sandbox fails with "network bridge not found"; set
# DOCKER_BUILD_NETWORK=host to route RUN steps over host networking instead.
DOCKER_NET_OPT=()
[[ -n "${DOCKER_BUILD_NETWORK:-}" ]] && DOCKER_NET_OPT=(--network="${DOCKER_BUILD_NETWORK}")

${DOCKER_CMD} build -f ${DOCKERFILE_PATH} -t ${DOCKER_IMAGE} \
    "${DOCKER_NET_OPT[@]}" \
    ${DOCKER_CACHE_OPT} \
    --build-arg GPU_TARGETS=${TARGET_GPU_ARCH} \
    --build-arg USE_LOCAL_SRC=${USE_LOCAL_SRC} \
    --build-arg ONLY_FUNCS="${ONLY_FUNCS}" \
    --build-arg ROCSHMEM_USE_SDMA=${ROCSHMEM_USE_SDMA} \
    --build-arg RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS=${RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS} \
    "${TRACE_BUILD_ARGS[@]}" \
    --build-arg RCCL_CACHE_BUST=$((RCCL_CACHE_BUST++)) \
    --build-arg ROCSHMEM_CACHE_BUST=$((ROCSHMEM_CACHE_BUST++)) \
    .
${DOCKER_CMD} image inspect "${DOCKER_IMAGE}" >/dev/null

# ---------------------------------------------------------------------------
# Build hardening (runtime): assert real GIN Anvil-SDMA bring-up on GPUs.
# A previous rebuild shipped an image whose communicator came up with
# ginType=NONE ("GIN support is not enabled"); static checks cannot catch that,
# so run a minimal Test#5 (NCCL_GIN_TYPE=5) and fail if GIN does not initialize.
# Skips cleanly on GPU-less builders or when RCCL_IMAGE_GIN_SMOKE=0.
#   RCCL_IMAGE_GIN_SMOKE=0  disable this assert
#   GIN_SMOKE_NP=<n>        GPU/rank count (default 8)
#   GIN_SMOKE_SIZE=<bytes>  max message size, e.g. 1M (default 1M)
# ---------------------------------------------------------------------------
RCCL_IMAGE_GIN_SMOKE="${RCCL_IMAGE_GIN_SMOKE:-1}"
GIN_SMOKE_NP="${GIN_SMOKE_NP:-8}"
GIN_SMOKE_SIZE="${GIN_SMOKE_SIZE:-1M}"
if [ "${RCCL_IMAGE_GIN_SMOKE}" = "1" ]; then
  if [ ! -e /dev/kfd ]; then
    echo "WARN: GIN smoke assert skipped (no /dev/kfd; GPU-less builder). Set RCCL_IMAGE_GIN_SMOKE=0 to silence." >&2
  else
    echo "=== Build hardening: GIN Anvil-SDMA smoke assert (NP=${GIN_SMOKE_NP}, NCCL_GIN_TYPE=5, -e ${GIN_SMOKE_SIZE}) ==="
    GIN_SMOKE_LOG="$(mktemp)"
    RCCL_GIN_RUN_TESTS=5 TEST5_MLX5_PREFLIGHT=0 \
      bash "${DEV_ARTI_DIR}/scripts/gin-sdma-a2a-test.bash" "${GIN_SMOKE_NP}" "${GIN_SMOKE_SIZE}" \
      > "${GIN_SMOKE_LOG}" 2>&1
    if grep -qE "GIN support is not enabled for this communicator|Failed to initialize any GIN plugin|Test failure" "${GIN_SMOKE_LOG}" \
       || ! grep -q "Out of bounds values : 0 OK" "${GIN_SMOKE_LOG}"; then
      echo "ERROR: GIN Anvil-SDMA bring-up FAILED for image '${DOCKER_IMAGE}'." >&2
      echo "       The communicator did not enable GIN (ginType=NONE) or validation failed." >&2
      echo "       This image is NOT safe for GIN tests. Smoke log tail:" >&2
      echo "------------------------------ smoke log tail ------------------------------" >&2
      tail -n 40 "${GIN_SMOKE_LOG}" >&2
      echo "----------------------------------------------------------------------------" >&2
      rm -f "${GIN_SMOKE_LOG}"
      exit 1
    fi
    echo "GIN smoke OK: $(grep 'Avg bus bandwidth' "${GIN_SMOKE_LOG}" | tail -n1 | sed 's/^# *//')"
    rm -f "${GIN_SMOKE_LOG}"
  fi
else
  echo "NOTE: GIN smoke assert disabled (RCCL_IMAGE_GIN_SMOKE=0)."
fi
