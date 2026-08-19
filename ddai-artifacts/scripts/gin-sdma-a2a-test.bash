#! /usr/bin/env bash
# Single-node GIN alltoall perf harness (docker, non-Ruby).
# Usage: ./docker-gin-gda-sdma-test.bash [NP] [MAX_BYTES]
# Options: docs/gin-anvil-sdma-backend-tests.md

NP=${1:-8}
MAX_BYTES="${2:-128M}"
DOCKER_CMD="${DOCKER_CMD:-docker}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rccl-gin-gda-sdma-713}"
RCCL_GIN_RUN_TESTS="${RCCL_GIN_RUN_TESTS:-${RUN_TESTS:-1,5}}"
GDA_HOST_LIB_DIRS="${TEST2_HOST_SO_SEARCH_DIRS:-${TEST2_HOST_SO_SEARCH_DIRS:-/lib64 /usr/lib64 /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu}}"
ROCSHMEM_THRESHOLD=$((128 * 1024 * 1024))

# Shared measurement iteration counts applied to ALL tests (#1-#5) so the host,
# rocSHMEM, and GIN paths are compared over the SAME warmup(skip)/timed(loop)
# counts. WARMUP = discarded warmup iters (-w / device "skip"); ITERS = timed
# iters (-n / device "loop"). For Test#5 device-timing these also feed the
# in-kernel wall_clock64 skip/loop so its measurement window matches the others.
A2A_WARMUP="${A2A_WARMUP:-5}"
A2A_ITERS="${A2A_ITERS:-20}"

_run_test() {
  [[ ",${RCCL_GIN_RUN_TESTS}," == *",$1,"* ]]
}

_trace_on() {
  [[ "${RCCL_GIN_ECHO:-1}" == 1 ]] && set -x
  return 0
}

_trace_off() {
  [[ "${RCCL_GIN_ECHO:-1}" == 1 ]] && set +x
  return 0
}

if [[ "${DOCKER_ULIMIT_MEMLOCK:-1}" != 0 ]]; then
  D_MEMLOCK=(--ulimit memlock=-1:-1)
else
  D_MEMLOCK=()
fi

DOCKER_GPU_COMMON="${D_MEMLOCK[*]} --shm-size 64G --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband --ipc host --group-add video --group-add render --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged ${DOCKER_EXTRA:-}"

GDA_UVERBS_ADDED=0
_rccl_uverbs_seen=""
_rccl_add_uverbs() {
  local p="$1" rp
  [[ -e "${p}" ]] || return 1
  rp=$(readlink -f "${p}" 2>/dev/null) || return 1
  [[ -n "${rp}" ]] || rp="${p}"
  [[ -c "${rp}" ]] || return 1
  [[ " ${_rccl_uverbs_seen} " == *" ${rp} "* ]] && return 0
  DOCKER_GPU_COMMON+=" --device ${rp}"
  _rccl_uverbs_seen+=" ${rp} "
  GDA_UVERBS_ADDED=$((GDA_UVERBS_ADDED + 1))
}
if [[ "${DOCKER_UVERBS:-1}" != 0 ]]; then
  shopt -s nullglob
  for _u in /dev/infiniband/uverbs* /dev/uverbs*; do
    _rccl_add_uverbs "${_u}" || true
  done
  shopt -u nullglob
  if [[ "${GDA_UVERBS_ADDED}" -eq 0 ]]; then
    for _i in $(seq 0 31); do
      _rccl_add_uverbs "/dev/infiniband/uverbs${_i}" || true
      _rccl_add_uverbs "/dev/uverbs${_i}" || true
    done
  fi
fi
unset _rccl_uverbs_seen _u _i

if [[ "${DOCKER_RDMA_GROUP:-1}" != 0 ]] && getent group rdma >/dev/null 2>&1; then
  DOCKER_GPU_COMMON+=" --group-add rdma"
fi

if [[ "${GIN_GDA_DOCKER_IT:-0}" == 1 ]]; then
  DOCKER_GPU="-it --rm --init ${DOCKER_GPU_COMMON}"
else
  DOCKER_GPU="--rm --init ${DOCKER_GPU_COMMON}"
fi

MPI_OPT="--allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none ${MPI_MCA_EXTRA:-}"

GIN_PLUGIN_X=()
[[ "${USE_EXTERNAL_PLUGIN:-0}" != 1 ]] && GIN_PLUGIN_X=(-x NCCL_GIN_PLUGIN=none)

MPI_BASE=(
  -x OMPI_ALLOW_RUN_AS_ROOT=1
  -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
  -x RCCL_ROCSHMEM_ENABLE=0
  -x ROCSHMEM_BACKEND=ipc
  -x ROCSHMEM_DISABLE_MIXED_IPC=1
  -x ROCSHMEM_DEBUG_LEVEL=info:noversion
  -x RCCL_ROCSHMEM_THRESHOLD="${ROCSHMEM_THRESHOLD}"
  -x NCCL_DEBUG="${NCCL_DEBUG:-VERSION}"
  -x NCCL_DEBUG_SUBSYS=INIT,NET
  -x RCCL_ENABLE_INTRANET=1
  -x NCCL_DMABUF_ENABLE=1
  -x NCCL_MSCCL_ENABLE=0
  -x HSA_NO_SCRATCH_RECLAIM=1
)

# [GIN-CONN-CHECK][TEST] Optional fault-injection passthrough to exercise the
# LSA signal connectivity fail-loud abort path deterministically.
[[ -n "${NCCL_GIN_ANVIL_SDMA_CONN_INJECT_FAIL_RANK:-}" ]] && \
  MPI_BASE+=(-x "NCCL_GIN_ANVIL_SDMA_CONN_INJECT_FAIL_RANK=${NCCL_GIN_ANVIL_SDMA_CONN_INJECT_FAIL_RANK}")

# [CUMEM-SKIP-FREE][WORKAROUND] gfx950 hipMemUnmap teardown deadlock: forward
# NCCL_CUMEM_SKIP_FREE=1 to ranks so ncclCuMemFree skips the hanging cuMem unmap
# at ncclCommDestroy (leaks the aperture; OS reclaims at process exit).
[[ -n "${NCCL_CUMEM_SKIP_FREE:-}" ]] && \
  MPI_BASE+=(-x "NCCL_CUMEM_SKIP_FREE=${NCCL_CUMEM_SKIP_FREE}")

DOCKER_TEST2_VOLUMES=""
DOCKER_TEST5_MLX5_VOLUMES=""
_rccl_bind_seen=""

_rccl_bind_add() {
  local src="$1" dst="$2"
  [[ -n "${src}" && -e "${src}" && -n "${dst}" ]] || return 0
  [[ " ${_rccl_bind_seen} " == *" ${dst} "* ]] && return 0
  DOCKER_TEST2_VOLUMES+=" -v ${src}:${dst}:ro"
  _rccl_bind_seen+=" ${dst} "
}

_rccl_resolve_lib() {
  local base="$1" d cand
  for d in ${GDA_HOST_LIB_DIRS}; do
    cand="${d}/${base}"
    [[ -e "${cand}" ]] || continue
    readlink -f "${cand}" 2>/dev/null || echo "${cand}"
    return 0
  done
  return 1
}

_rccl_bind_lib_paths() {
  local base="$1" real d cand
  real=$(_rccl_resolve_lib "${base}") || return 1
  for d in ${GDA_HOST_LIB_DIRS}; do
    cand="${d}/${base}"
    [[ -e "${cand}" ]] || continue
    _rccl_bind_add "${real}" "${cand}"
    [[ "${real}" != "${cand}" ]] && _rccl_bind_add "${real}" "${real}"
  done
}

_rccl_bind_mlx5_adjacent() {
  local vbdir="$1" mlx cand real d
  for mlx in libmlx5.so libmlx5.so.1 libmlx5-infiniband.so.1 libmlx5dv.so libmlx5dv.so.1; do
    cand="${vbdir}/${mlx}"
    [[ -e "${cand}" ]] || continue
    real=$(readlink -f "${cand}" 2>/dev/null || echo "${cand}")
    for d in ${GDA_HOST_LIB_DIRS}; do
      _rccl_bind_add "${real}" "${d}/${mlx}"
    done
  done
}

_rccl_setup_rdma_volumes() {
  local d verbs vbdir plug base nl3 nlrt

  if [[ "${TEST2_BIND_HOST_GNU_DIRS:-${TEST2_BIND_HOST_GNU_DIRS:-0}}" != 0 ]]; then
    echo "warning: TEST2_BIND_HOST_GNU_DIRS=1 can break librccl if host glibc is older than the image." >&2
    for d in ${TEST2_BIND_HOST_LIBDIRS:-${TEST2_BIND_HOST_LIBDIRS:-/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu}}; do
      [[ -d "${d}" ]] && DOCKER_TEST2_VOLUMES+=" -v ${d}:${d}:ro"
    done
  fi

  if [[ "${TEST2_BIND_HOST_RDMA_SO:-${TEST2_BIND_HOST_RDMA_SO:-1}}" != 0 ]]; then
    _rccl_bind_seen=""
    if [[ -n "${TEST2_BIND_HOST_RDMA_BASE+x}" || -n "${TEST2_BIND_HOST_RDMA_BASE+x}" ]]; then
      for base in ${TEST2_BIND_HOST_RDMA_BASE:-${TEST2_BIND_HOST_RDMA_BASE:-}} \
                  ${TEST2_BIND_HOST_RDMA_EXTRA:-${TEST2_BIND_HOST_RDMA_EXTRA:-}} \
                  libnl-3.so.200 libnl-route-3.so.200; do
        [[ -n "${base// }" ]] && _rccl_bind_lib_paths "${base}" || true
      done
    else
      for base in librdmacm.so librdmacm.so.1 libibumad.so.3 \
                  ${TEST2_BIND_HOST_RDMA_EXTRA:-${TEST2_BIND_HOST_RDMA_EXTRA:-}}; do
        [[ -n "${base// }" ]] && _rccl_bind_lib_paths "${base}" || true
      done
      case "${TEST2_BIND_HOST_MLX5_SO:-${TEST2_BIND_HOST_MLX5_SO:-adjacent}}" in
        1)
          for base in libmlx5.so libmlx5.so.1 libmlx5-infiniband.so.1 libmlx5dv.so libmlx5dv.so.1; do
            _rccl_bind_lib_paths "${base}" || true
          done
          ;;
      esac
    fi

    if verbs=$(_rccl_resolve_lib libibverbs.so.1); then
      vbdir=$(dirname "${verbs}")
      for d in ${GDA_HOST_LIB_DIRS}; do
        _rccl_bind_add "${verbs}" "${d}/libibverbs.so"
        _rccl_bind_add "${verbs}" "${d}/libibverbs.so.1"
      done
      _rccl_bind_add "${verbs}" "${verbs}"

      if nl3=$(_rccl_resolve_lib libnl-3.so.200) && nlrt=$(_rccl_resolve_lib libnl-route-3.so.200); then
        for d in ${GDA_HOST_LIB_DIRS}; do
          _rccl_bind_add "${nl3}" "${d}/libnl-3.so.200"
          _rccl_bind_add "${nlrt}" "${d}/libnl-route-3.so.200"
        done
      fi

      plug="${vbdir}/libibverbs"
      if [[ -d "${plug}" ]]; then
        for d in ${GDA_HOST_LIB_DIRS}; do
          _rccl_bind_add "${plug}" "${d}/libibverbs"
        done
      fi

      case "${TEST2_BIND_HOST_MLX5_SO:-${TEST2_BIND_HOST_MLX5_SO:-adjacent}}" in
        0) ;;
        1) ;;
        *) _rccl_bind_mlx5_adjacent "${vbdir}" ;;
      esac
    fi
    unset _rccl_bind_seen
  fi

  if [[ "${TEST2_BIND_HOST_IB_SYSFS:-${TEST2_BIND_HOST_IB_SYSFS:-1}}" != 0 ]]; then
    for d in /sys/class/infiniband /etc/libibverbs.d; do
      [[ -e "${d}" ]] && DOCKER_TEST2_VOLUMES+=" -v ${d}:${d}:ro"
    done
  fi

  case "${TEST2_BIND_HOST_DEV_IFB:-${TEST2_BIND_HOST_DEV_INFINIBAND:-auto}}" in
    on) DOCKER_TEST2_VOLUMES+=" -v /dev/infiniband:/dev/infiniband" ;;
    off) ;;
    auto)
      if [[ "${GDA_UVERBS_ADDED:-0}" -eq 0 ]] && { [[ -d /dev/infiniband ]] || [[ -d /sys/class/infiniband ]]; }; then
        DOCKER_TEST2_VOLUMES+=" -v /dev/infiniband:/dev/infiniband"
      fi
      ;;
    *)
      echo "error: TEST2_BIND_HOST_DEV_IFB must be auto, on, or off" >&2
      exit 1
      ;;
  esac
}

_rccl_ver_ge() {
  local IFS=. a b i ai bi
  read -r -a a <<< "$1"
  read -r -a b <<< "$2"
  for i in 0 1 2 3; do
    ai=${a[i]:-0}
    bi=${b[i]:-0}
    (( 10#${ai} > 10#${bi} )) && return 0
    (( 10#${ai} < 10#${bi} )) && return 1
  done
  return 0
}

_rccl_skip_test4_auto() {
  local nic driver fw any_bnxt=0 min="${MIN_BNXT_FW_FOR_GDA:-233.2.104.0}"
  for nic in /sys/class/net/*; do
    [[ -e "${nic}" ]] || continue
    nic=${nic##*/}
    [[ "${nic}" == lo ]] && continue
    driver=$(readlink -f "/sys/class/net/${nic}/device/driver" 2>/dev/null)
    driver=${driver##*/}
    [[ "${driver}" == bnxt_en ]] || continue
    any_bnxt=1
    command -v ethtool >/dev/null 2>&1 || continue
    fw=$(ethtool -i "${nic}" 2>/dev/null | awk -F': +' '/firmware-version:/{v=$2; gsub(/ .*/,"",v); gsub(/\/.*/,"",v); print v; exit}')
    [[ -n "${fw}" ]] || continue
    if ! _rccl_ver_ge "${fw}" "${min}"; then
      echo "=== RCCL_GIN_GDA: skipping Test#4 (bnxt ${nic} fw ${fw} < ${min}); set TEST4_MODE=run to force ===" >&2
      return 0
    fi
  done
  if [[ "${any_bnxt}" == 0 ]]; then
    echo "=== RCCL_GIN_GDA: skipping Test#4 (no bnxt_en); set TEST4_MODE=run to force ===" >&2
    return 0
  fi
  return 1
}

_should_run_test4() {
  _run_test 4 || return 1
  case "${TEST4_MODE:-${TEST4_MODE:-auto}}" in
    skip) return 1 ;;
    run) return 0 ;;
    auto) ! _rccl_skip_test4_auto ;;
    *)
      echo "error: TEST4_MODE must be auto, run, or skip" >&2
      exit 1
      ;;
  esac
}

_rccl_setup_test5_mlx5_volumes() {
  local dir="$1" cand real base d
  [[ -d "${dir}" ]] || {
    echo "error: TEST5_HOST_MLX5_LIB_DIR is not a directory: ${dir}" >&2
    return 1
  }
  _rccl_bind_seen=""
  shopt -s nullglob
  for cand in "${dir}"/libmlx5*.so* "${dir}"/libmlx5dv*.so*; do
    [[ -f "${cand}" ]] || continue
    real=$(readlink -f "${cand}" 2>/dev/null || echo "${cand}")
    base=$(basename "${cand}")
    for d in ${GDA_HOST_LIB_DIRS}; do
      [[ " ${_rccl_bind_seen} " == *" ${d}/${base} "* ]] && continue
      DOCKER_TEST5_MLX5_VOLUMES+=" -v ${real}:${d}/${base}:ro"
      _rccl_bind_seen+=" ${d}/${base} "
    done
  done
  shopt -u nullglob
  unset _rccl_bind_seen
  [[ -n "${DOCKER_TEST5_MLX5_VOLUMES}" ]] || {
    echo "error: no libmlx5*.so* or libmlx5dv*.so* in ${dir}" >&2
    return 1
  }
}

_rccl_setup_rdma_volumes
if [[ -n "${TEST5_HOST_MLX5_LIB_DIR:-${TEST5_HOST_MLX5_LIB_DIR:-}}" ]]; then
  _rccl_setup_test5_mlx5_volumes "${TEST5_HOST_MLX5_LIB_DIR:-${TEST5_HOST_MLX5_LIB_DIR}}" || exit 1
fi

_rccl_test5_mlx5_ok() {
  [[ -n "${TEST5_HOST_MLX5_LIB_DIR:-${TEST5_HOST_MLX5_LIB_DIR:-}}" ]] && return 0
  ${DOCKER_CMD} run --rm --init ${DOCKER_TEST5_MLX5_VOLUMES} "${DOCKER_IMAGE}" sh -lc \
    'f=/lib/x86_64-linux-gnu/libmlx5.so.1; test -e "$f" || f=/usr/lib/x86_64-linux-gnu/libmlx5.so.1; \
     rf=$(readlink -f "$f"); test -f "$rf" && objdump -T "$rf" | grep -q mlx5dv_reg_dmabuf_mr' \
    >/dev/null 2>&1
}

_should_run_test5() {
  _run_test 5 || return 1
  [[ "${TEST5_MODE:-${TEST5_MODE:-run}}" != skip ]] || return 1
  if [[ "${TEST5_MLX5_PREFLIGHT:-${TEST5_MLX5_PREFLIGHT:-0}}" != 0 ]]; then
    if ! _rccl_test5_mlx5_ok; then
      echo "=== RCCL_GIN_GDA: skipping Test#5 (image libmlx5 lacks mlx5dv_reg_dmabuf_mr); set TEST5_MLX5_PREFLIGHT=0 or TEST5_HOST_MLX5_LIB_DIR ===" >&2
      return 1
    fi
  fi
  return 0
}

# Host-initiated A2A paths (all -D 0), measured 8x MI355X (NCCL_GIN_TYPE=5,
# 2026-07-24), out-of-place busbw:
#   * RING (SM copy): pin the channel count so the tuner does not collapse
#     channels on large messages. Without the pin busbw cliffs hard past 8M
#     (~182 GB/s @4M -> ~30 @8M -> ~45 @128M); pinning MIN=MAX (any 16..64 is
#     equivalent) lifts 8M to ~205 and 128M to ~326 with no small-size loss.
#     Best for <=32M.
#   * CE/SDMA (copy engines): NCCL_CUMEM_ENABLE=1 + symmetric reg (-R 2) +
#     NCCL_CTA_POLICY=ZERO. Slow for small/mid but scales to ~373 @128M.
#     Best for >=64M.
# TEST1_MODE selects: ring (default) | hybrid (RING<=split, CE>split) | ce.
_a2a_host_ring() {  # $1=min $2=max
  ${DOCKER_CMD} run ${DOCKER_GPU} "${DOCKER_IMAGE}" \
    mpirun -n "${NP}" ${MPI_OPT} \
    "${MPI_BASE[@]}" \
    -x NCCL_CUMEM_ENABLE=0 \
    -x ROCSHMEM_SDMA_ENABLED=0 \
    -x NCCL_GIN_ENABLE=0 \
    -x NCCL_GIN_TYPE=0 \
    -x NCCL_MIN_NCHANNELS="${TEST1_NCHANNELS}" \
    -x NCCL_MAX_NCHANNELS="${TEST1_NCHANNELS}" \
    rccl-tests/alltoall_perf -b "$1" -e "$2" -f 2 -g 1 -R 0 -D 0 -A 1 -V 1 -w "${A2A_WARMUP}" -n "${A2A_ITERS}"
}
_a2a_host_ce() {  # $1=min $2=max
  ${DOCKER_CMD} run ${DOCKER_GPU} "${DOCKER_IMAGE}" \
    mpirun -n "${NP}" ${MPI_OPT} \
    "${MPI_BASE[@]}" \
    -x NCCL_CUMEM_ENABLE=1 \
    -x ROCSHMEM_SDMA_ENABLED=0 \
    -x NCCL_GIN_ENABLE=0 \
    -x NCCL_GIN_TYPE=0 \
    -x NCCL_CTA_POLICY=ZERO \
    rccl-tests/alltoall_perf -b "$1" -e "$2" -f 2 -g 1 -R 2 -D 0 -A 1 -V 1 -w "${A2A_WARMUP}" -n "${A2A_ITERS}"
}

# --- UT: GIN-SDMA host policy unit tests (no GPU); optional preflight gate ---
# Validates the shared tier-selection / threshold / chunk logic that the device
# kernels rely on. Fast, GPU-free but OFF by default: the gin_sdma_policy_test
# binary is not built into the current image, so the gate would abort the run.
# Set RUN_POLICY_UT=1 to enable once the binary is present in the image.
POLICY_UT="${POLICY_UT:-rccl-tests/gin_sdma_policy_test}"
if [[ "${RUN_POLICY_UT:-0}" != "0" ]]; then
  echo "=== UT: host policy unit tests (${POLICY_UT}) ==="
  ${DOCKER_CMD} run --rm --init "${DOCKER_IMAGE}" "${POLICY_UT}" \
    || { echo "FATAL: GIN-SDMA policy unit tests failed"; exit 1; }
fi

if _run_test 1; then
  _trace_on
  TEST1_NCHANNELS="${TEST1_NCHANNELS:-32}"
  TEST1_MODE="${TEST1_MODE:-ring}"
  case "${TEST1_MODE}" in
    ring)
      echo "=== Test#1: A2A, ${NP} gpus, Host Initiated RING (channels=${TEST1_NCHANNELS}) ==="
      _a2a_host_ring 128 "${MAX_BYTES}"
      ;;
    ce)
      echo "=== Test#1: A2A, ${NP} gpus, Host Initiated CE/SDMA ==="
      _a2a_host_ce 128 "${MAX_BYTES}"
      ;;
    hybrid)
      # RING for <=split, CE/SDMA for >split. Default split 32M; CE phase
      # starts at the next f2 step (2*split). Override split with
      # TEST1_HYBRID_SPLIT (bytes).
      TEST1_HYBRID_SPLIT="${TEST1_HYBRID_SPLIT:-33554432}"
      _ce_min=$(( TEST1_HYBRID_SPLIT * 2 ))
      echo "=== Test#1a: A2A, ${NP} gpus, Host Initiated HYBRID/RING (channels=${TEST1_NCHANNELS}) 128..${TEST1_HYBRID_SPLIT} ==="
      _a2a_host_ring 128 "${TEST1_HYBRID_SPLIT}"
      echo "=== Test#1b: A2A, ${NP} gpus, Host Initiated HYBRID/CE-SDMA ${_ce_min}..${MAX_BYTES} ==="
      _a2a_host_ce "${_ce_min}" "${MAX_BYTES}"
      ;;
    *)
      echo "error: TEST1_MODE must be ring, hybrid, or ce" >&2
      exit 1
      ;;
  esac
  _trace_off
fi

if _run_test 2; then
  _trace_on
  echo "=== Test#2: A2A, ${NP} gpus, GIN Host Proxy (NCCL_GIN_TYPE=2) ==="
  ${DOCKER_CMD} run ${DOCKER_GPU}${DOCKER_TEST2_VOLUMES} "${DOCKER_IMAGE}" \
    mpirun -n "${NP}" ${MPI_OPT} \
    "${MPI_BASE[@]}" \
    "${GIN_PLUGIN_X[@]}" \
    -x NCCL_CUMEM_ENABLE=1 \
    -x NCCL_NET_PLUGIN=none \
    -x NCCL_ENV_PLUGIN=none \
    -x ROCSHMEM_SDMA_ENABLED=0 \
    -x NCCL_DEBUG="${NCCL_DEBUG:-VERSION}" \
    -x NCCL_GIN_ENABLE=1 \
    -x NCCL_GIN_TYPE=2 \
    -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
    rccl-tests/alltoall_perf -b 128 -e "${MAX_BYTES}" -f 2 -g 1 -R 2 -D 3 -A 1 -V 1 -w "${A2A_WARMUP}" -n "${A2A_ITERS}"
  _trace_off
fi

if _should_run_test4; then
  _trace_on
  echo "=== Test#4: A2A, ${NP} gpus, GIN GDA ==="
  ${DOCKER_CMD} run ${DOCKER_GPU} "${DOCKER_IMAGE}" \
    mpirun -n "${NP}" ${MPI_OPT} \
    "${MPI_BASE[@]}" \
    "${GIN_PLUGIN_X[@]}" \
    -x NCCL_CUMEM_ENABLE=1 \
    -x NCCL_NET_PLUGIN=none \
    -x ROCSHMEM_SDMA_ENABLED=1 \
    -x NCCL_GIN_ENABLE=1 \
    -x NCCL_GIN_TYPE=4 \
    rccl-tests/alltoall_perf -b 128 -e "${MAX_BYTES}" -f 2 -g 1 -R 2 -D 3 -A 1 -V 1 -w "${A2A_WARMUP}" -n "${A2A_ITERS}"
  _trace_off
fi

if _should_run_test5; then
  _trace_on
  # Backend gin.put SDMA control (rsCtx->sdmaThreshold): 0 = every GIN put uses
  # the copy engine. Kept at 0 so the -D 3 SDMA branch always drives SDMA.
  NCCL_GIN_ANVIL_SDMA_THRESHOLD="${NCCL_GIN_ANVIL_SDMA_THRESHOLD:-0}"
  NCCL_GIN_ANVIL_SDMA_MAX_COPY_CHUNK="${NCCL_GIN_ANVIL_SDMA_MAX_COPY_CHUNK:-8192}"
  # Kernel-level LSA<->SDMA switch for the -D 3 size-hybrid (per-peer bytes).
  # Set explicitly so it takes precedence over the shared THRESHOLD=0 above
  # (which would otherwise force the kernel all-SDMA). Default matches the
  # rccl-tests built-in default (2097152 = 2 MiB/peer = 16M total on 8 ranks,
  # the F1/F2/F3-measured LSA<->SDMA crossover). Set to 0 to force all-SDMA, or
  # a huge value to force all-LSA (sweeps).
  TEST5_A2A_THRESHOLD="${TEST5_A2A_THRESHOLD:-2097152}"
  # Optional LSA-tier CTA-count override (F1 tuning knob): 0 = use the adaptive
  # a2aLsaCtaCount ladder (default), >0 = force that grid for the LSA tier.
  TEST5_A2A_LSA_CTAS="${TEST5_A2A_LSA_CTAS:-0}"
  # Multi-CTA LL tier (small-message tail prototype): per-peer bytes cap to enable
  # the barrier-free LL scatter/gather (0/unset = disabled -> direct-LSA copy),
  # and an optional CTA-count override for that tier (0 = use kA2aLLCtas default).
  TEST5_A2A_LL_MAX="${TEST5_A2A_LL_MAX:-0}"
  TEST5_A2A_LL_CTAS="${TEST5_A2A_LL_CTAS:-0}"
  # LSA-tier cross-rank sync mode: 3 = single exit barrier (default, +9-17% vs
  # legacy), 0 = two LSA barriers (legacy), 1 = none (diagnostic ceiling, correct
  # only under external sync), 2 = point-to-point done-flag completion (diagnostic).
  TEST5_A2A_SYNC_MODE="${TEST5_A2A_SYNC_MODE:-3}"
  # Device-side (in-kernel wall_clock64) timing (AICOMRCCL-1459, rocSHMEM method).
  # Reports the pure GPU device-function execution time (excludes host launch /
  # teardown / graph overhead). Modes:
  #   0 = off (default).
  #   1 = augment: normal graph/hipEvent run PLUS an extra "#[a2a-devtime]" line.
  #   2 = device-time-only: skip the graph/hipEvent timed loop for the
  #       out-of-place pass and report the device time as THE metric (warmup +
  #       datacheck still run for #wrong). Fastest, cleanest single number.
  # LOOP/SKIP mirror rocSHMEM defaults (10/10); auto-reduced for large chunks.
  TEST5_A2A_DEVICE_TIMING="${TEST5_A2A_DEVICE_TIMING:-0}"
  # Default the device-timing skip/loop to the shared harness counts so Test#5's
  # in-kernel measurement window matches the -w/-n used by Test#1-#4.
  TEST5_A2A_DEVTIME_LOOP="${TEST5_A2A_DEVTIME_LOOP:-${A2A_ITERS}}"
  TEST5_A2A_DEVTIME_SKIP="${TEST5_A2A_DEVTIME_SKIP:-${A2A_WARMUP}}"
  TEST5_MPI_EXTRA=(
    -x "NCCL_GIN_ANVIL_SDMA_THRESHOLD=${NCCL_GIN_ANVIL_SDMA_THRESHOLD}"
    -x "NCCL_GIN_ANVIL_SDMA_MAX_COPY_CHUNK=${NCCL_GIN_ANVIL_SDMA_MAX_COPY_CHUNK}"
    -x "NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLTOALL=${TEST5_A2A_THRESHOLD}"
    -x "NCCL_GIN_ANVIL_A2A_LSA_CTAS=${TEST5_A2A_LSA_CTAS}"
    -x "NCCL_GIN_ANVIL_A2A_LL_MAX_BYTES=${TEST5_A2A_LL_MAX}"
    -x "NCCL_GIN_ANVIL_A2A_LL_CTAS=${TEST5_A2A_LL_CTAS}"
    -x "NCCL_GIN_ANVIL_A2A_SYNC_MODE=${TEST5_A2A_SYNC_MODE}"
    -x "NCCL_GIN_ANVIL_A2A_DEVICE_TIMING=${TEST5_A2A_DEVICE_TIMING}"
    -x "NCCL_GIN_ANVIL_A2A_DEVTIME_LOOP=${TEST5_A2A_DEVTIME_LOOP}"
    -x "NCCL_GIN_ANVIL_A2A_DEVTIME_SKIP=${TEST5_A2A_DEVTIME_SKIP}"
  )
  # GIN Anvil-SDMA A2A device paths (NCCL_GIN_TYPE=5), measured 8x MI355X
  # (2026-07-26), out-of-place:
  #   * -D 3 GinHybridAlltoAllKernel: size-hybrid. Per-peer chunk <=threshold
  #     uses a direct LSA all-peers copy (all CTAs; ~11us small-msg latency,
  #     ~2x mid-range busbw vs SDMA); above threshold uses all-peers GIN puts
  #     (SDMA copy engines; scales to ~390 GB/s @128M). Needs -V 8 so the LSA
  #     branch gets enough CTAs; SDMA is CTA-insensitive. Best across the range.
  #   * -D 4 HybridAlltoAllKernel: topology split (CTA 0 = remote via GIN, CTAs
  #     1..N = intra-node via LSA). On a single node LSA carries all traffic:
  #     low small-msg latency but scalar SM copy caps large BW. Kept for debug.
  # TEST5_MODE: d3 (default, kernel size-hybrid -V 8) | d4 (LSA topology split).
  TEST5_MODE="${TEST5_MODE:-d3}"
  TEST5_D4_CTA_COUNT="${TEST5_D4_CTA_COUNT:-${TEST5_CTA_COUNT:-8}}"
  TEST5_D3_CTA_COUNT="${TEST5_D3_CTA_COUNT:-8}"
  # HIP-graph capture for the Test#5 measurement (-G/--cudagraph). Captures the
  # timed iters loop once and replays it TEST5_CUDAGRAPH times, so per-iteration
  # *host* kernel-launch overhead is removed from the reported GPU time. Smoke-
  # tested on 8x MI355X: the GIN -D 3 kernel captures and validates cleanly
  # (#wrong=0) across both the LSA (small) and SDMA (large) branches.
  # Caveat: -G removes only host launch latency. The in-kernel opening cross-
  # node barrier and closing waitSignal+flush execute on every replay and are
  # still counted (isolating those needs device-side timing, out of scope here).
  # Warmup must be >=1 so GIN connection setup / first-touch allocation happen
  # BEFORE capture (a cudaMalloc during ThreadLocal stream capture is illegal).
  # Set TEST5_CUDAGRAPH=0 to disable -G and fall back to launch-inclusive timing.
  # Only Test#5 uses -G; Test#1 (host baseline) stays launch-inclusive by design.
  # Default is 0 (eager): HIP graph capture cannot drive the out-of-band SDMA
  # hardware queue, so -G forces the LSA-only graph-safe fallback (see Option A in
  # alltoall.cu), which pins large-message busbw at ~20 GB/s vs ~390 GB/s eager and
  # can also deadlock. Set TEST5_CUDAGRAPH=4 explicitly only to measure the
  # launch-latency-removed LSA path.
  TEST5_CUDAGRAPH="${TEST5_CUDAGRAPH:-0}"
  TEST5_WARMUP="${TEST5_WARMUP:-${A2A_WARMUP}}"
  TEST5_ITERS="${TEST5_ITERS:-${A2A_ITERS}}"
  TEST5_GRAPH_ARGS=()
  if [[ "${TEST5_CUDAGRAPH}" != 0 ]]; then
    TEST5_GRAPH_ARGS=(-G "${TEST5_CUDAGRAPH}")
    if [[ "${TEST5_WARMUP}" == 0 ]]; then
      echo "warning: TEST5_CUDAGRAPH>0 with TEST5_WARMUP=0 can fail graph capture (lazy alloc during capture); using -w 1" >&2
      TEST5_WARMUP=1
    fi
  fi
  _a2a_gin() {  # $1=deviceImpl $2=ctaCount $3=minBytes $4=maxBytes
    ${DOCKER_CMD} run ${DOCKER_GPU}${DOCKER_TEST5_MLX5_VOLUMES} "${DOCKER_IMAGE}" \
      mpirun -n "${NP}" ${MPI_OPT} \
      "${MPI_BASE[@]}" \
      "${GIN_PLUGIN_X[@]}" \
      -x NCCL_CUMEM_ENABLE=1 \
      -x NCCL_NET_PLUGIN=none \
      -x ROCSHMEM_SDMA_ENABLED=0 \
      -x NCCL_DEBUG="${NCCL_DEBUG:-VERSION}" \
      -x NCCL_GIN_ENABLE=1 \
      -x NCCL_GIN_TYPE=5 \
      -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS="${TEST5_NUM_CHANNELS:-1}" \
      -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
      "${TEST5_MPI_EXTRA[@]}" \
      rccl-tests/alltoall_perf -b "$3" -e "$4" -f 2 -g 1 -R 2 -D "$1" -A 1 -V "$2" \
      -w "${TEST5_WARMUP}" -n "${TEST5_ITERS}" "${TEST5_GRAPH_ARGS[@]}"
  }
  case "${TEST5_MODE}" in
    d3)
      echo "=== Test#5: A2A, ${NP} gpus, GIN Anvil SDMA -D 3 size-hybrid (LSA<=${TEST5_A2A_THRESHOLD}B/peer, SDMA above; V=${TEST5_D3_CTA_COUNT}, NCCL_GIN_TYPE=5, cudagraph=${TEST5_CUDAGRAPH}, w=${TEST5_WARMUP}, n=${TEST5_ITERS}) ==="
      _a2a_gin 3 "${TEST5_D3_CTA_COUNT}" 128 "${MAX_BYTES}"
      ;;
    d4)
      echo "=== Test#5: A2A, ${NP} gpus, GIN hybrid -D 4 LSA (V=${TEST5_D4_CTA_COUNT}, NCCL_GIN_TYPE=5, cudagraph=${TEST5_CUDAGRAPH}, w=${TEST5_WARMUP}, n=${TEST5_ITERS}) ==="
      _a2a_gin 4 "${TEST5_D4_CTA_COUNT}" 128 "${MAX_BYTES}"
      ;;
    *)
      echo "error: TEST5_MODE must be d3 or d4" >&2
      exit 1
      ;;
  esac
  _trace_off
fi
