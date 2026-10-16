#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Remote worker: Metric Health test report on one GPU architecture.
#
# Default workload iterations: 500 (median-stable). Override with --iterations.
#   vcopy:       -n 81920 -b 256 -i <iterations>
#   nbody:       mini-nbody-block 81920 <iterations>
#   mega_kernel: -b <mega-batch> -n <iterations>
#
# --arch selects the host profile. Differences that used to be copied scripts:
#   gfx1151  mega_kernel profile batch 4096 (RDNA3.5); smoke uses that batch
#   CDNA     mega_kernel profile batch 65536; smoke stays -b 4096
#   gfx908   roofline note: MI100 has no roofline microbenchmarks
#            Roofline is off unless --roof is passed
#   gfx908   gfx90a  skip mega_kernel when its build fails
#   gfx942   gfx1151 attempt native-tool, then retry once with --no-native-tool
#   gfx908   gfx90a gfx950  native tool stays off (no retry)
#
# Usage (on the GPU host, after syncing rocprofiler-compute):
#   bash scripts/run_metric_health_test_report.sh \
#     --arch gfx942 --tag 261001-gfx942-500med --therock /path/to/therock
#   bash scripts/run_metric_health_test_report.sh --arch gfx1151 --therock /path/to/rocm
#   bash scripts/run_metric_health_test_report.sh --arch gfx908 --therock /opt/rocm-7.2.0
#   bash scripts/run_metric_health_test_report.sh --arch gfx90a --therock /opt/rocm-7.2.0
#   bash scripts/run_metric_health_test_report.sh --arch gfx950 --therock /opt/rocm-7.1.0
#
# 500-iter medians are --iterations 500, including gfx942.
#
# One invocation profiles one build of rocprof-compute. The health report
# compares two separate runs: this branch and rocprofiler-compute-develop.
#   --run            name in output paths (default single-pass). Use --run develop
#                    for the baseline run.
#   --profiler-root  rocprofiler-compute checkout whose built/installed
#                    rocprof-compute runs profile and analyze (default --root).
#                    For the baseline, point it at a rocprofiler-compute-develop
#                    checkout. Samples still build from --root so both runs
#                    profile the same binaries.
#
# profile/analyze run the first rocprof-compute found, in order:
#   1. ${PROFILER_ROOT}/install/bin/rocprof-compute   (cmake --target install,
#      CMAKE_INSTALL_PREFIX=install; see README.md "manual testing")
#   2. ${PROFILER_ROOT}/bin/rocprof-compute            (--profiler-root given
#      directly as an install prefix)
#   3. ${THEROCK}/bin/rocprof-compute                  (ROCm/TheRock prefix
#      that already ships a built rocprof-compute)
#   4. rocprof-compute on PATH
#   5. python3 ${PROFILER_ROOT}/src/rocprof-compute    (fallback only; logs a
#      WARN because this runs uninstalled source, not a built artifact)
#
# Steps 3 and 4 resolve to the same binary for every run on a host, so passing
# --profiler-root explicitly drops them and resolves only within that tree
# (install/bin, bin, then its src/). Pass it on the candidate run too when
# building a baseline comparison, so each run profiles its own checkout.
#
#   bash scripts/run_metric_health_test_report.sh \
#     --arch gfx942 --tag 261011-gfx942-500-develop --run develop \
#     --profiler-root /path/to/develop/projects/rocprofiler-compute
#
# Produces under <root>/workloads/:
#   {vcopy,nbody,mega_kernel}_<tag>_<run>/
#   logs_<arch>_<tag>/{vcopy,nbody,mega_kernel}_<run>.log
#   logs_<arch>_<tag>/run_meta.env
# Ends with ALL_DONE in the log file.

set -euo pipefail

usage() {
  cat <<EOF
Usage: bash $(basename "$0") --arch <gfx942|gfx950|gfx90a|gfx908|gfx1151> [options]

Options:
  --tag <name>              Output name (default: YYMMDD-<arch>-500)
  --root <dir>              Sample tree (default: this repo)
  --profiler-root <dir>     rocprof-compute tree/install-prefix used to profile
                            (default: --root; see the file header for how a
                            built rocprof-compute is resolved)
  --run <name>              Run name in output paths (default: single-pass)
  --iterations <n>          Workload iterations (default: 500)
  --mega-batch <n>          mega_kernel batch (default: 4096 on gfx1151, else 65536)
  --therock <dir>           ROCm prefix (default: per-architecture path)
  --log <path>              Log file (default: per-architecture path)
  --local-prefix <dir>      Local pkg-config prefix (default: /tmp/rpc_local)
  --site-rocm <dir>         Site ROCm prefix for gfx908, gfx90a, and gfx950 (default: /opt/rocm)
  --roof                    Enable roofline collection (default: off)
  --no-native-tool          Skip the native counter tool (default on gfx908, gfx90a, gfx950)

ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES stay in the environment because the GPU runtime reads them.
EOF
}

require_option_value() {
  local option_name=$1
  local option_value=${2:-}
  if [[ -z "$option_value" || "$option_value" == --* ]]; then
    echo "${option_name} needs a value" >&2
    usage >&2
    exit 1
  fi
}

ARCH=""
TAG_ARG=""
ROOT_ARG=""
PROFILER_ROOT_ARG=""
RUN_ARG=""
ITERS_ARG=""
MEGA_BATCH_ARG=""
THEROCK_ARG=""
LOG_ARG=""
LOCAL_PREFIX_ARG=""
SITE_ROCM_ARG=""
ROOF=0
NO_NATIVE_TOOL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)
      require_option_value "$1" "${2:-}"
      ARCH=$2
      shift
      ;;
    --tag)
      require_option_value "$1" "${2:-}"
      TAG_ARG=$2
      shift
      ;;
    --root)
      require_option_value "$1" "${2:-}"
      ROOT_ARG=$2
      shift
      ;;
    --profiler-root)
      require_option_value "$1" "${2:-}"
      PROFILER_ROOT_ARG=$2
      shift
      ;;
    --run)
      require_option_value "$1" "${2:-}"
      RUN_ARG=$2
      shift
      ;;
    --iterations)
      require_option_value "$1" "${2:-}"
      ITERS_ARG=$2
      shift
      ;;
    --mega-batch)
      require_option_value "$1" "${2:-}"
      MEGA_BATCH_ARG=$2
      shift
      ;;
    --therock)
      require_option_value "$1" "${2:-}"
      THEROCK_ARG=$2
      shift
      ;;
    --log)
      require_option_value "$1" "${2:-}"
      LOG_ARG=$2
      shift
      ;;
    --local-prefix)
      require_option_value "$1" "${2:-}"
      LOCAL_PREFIX_ARG=$2
      shift
      ;;
    --site-rocm)
      require_option_value "$1" "${2:-}"
      SITE_ROCM_ARG=$2
      shift
      ;;
    --roof) ROOF=1 ;;
    --no-native-tool) NO_NATIVE_TOOL=1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

if [[ -z "$ARCH" ]]; then
  usage >&2
  exit 1
fi

apply_worker_profile() {
  SOC_LEAVES=()
  ENV_FAMILY=""
  NATIVE_MODE=""
  THEROCK_STYLE=""
  LOG_STYLE=home
  MEGA_BATCH_DEFAULT=65536
  SMOKE_FOLLOWS_MEGA_BATCH=0
  SKIP_MEGA_ON_BUILD_FAIL=0
  CDNA1_SMOKE_WARN=0
  WL_SEARCH=find
  WL_TREE_FALLBACK=0
  VENV_KIND=""
  ROOF_NOTE=" for median health wall time"
  META_SITE=0
  case "$ARCH" in
    gfx1151)
      ENV_FAMILY=apu
      NATIVE_MODE=probe
      THEROCK_STYLE=apu
      MEGA_BATCH_DEFAULT=4096
      SMOKE_FOLLOWS_MEGA_BATCH=1
      WL_SEARCH=candidates
      WL_TREE_FALLBACK=1
      SOC_LEAVES=(RDNA35_HALO)
      VENV_KIND=apu
      ;;
    gfx942)
      ENV_FAMILY=therock
      NATIVE_MODE=probe
      THEROCK_STYLE=fixed
      LOG_STYLE=therock
      WL_SEARCH=candidates
      WL_TREE_FALLBACK=1
      SOC_LEAVES=(MI300X_A1 MI300A_A0)
      VENV_KIND=rpc
      ;;
    gfx908)
      ENV_FAMILY=site
      NATIVE_MODE=force
      THEROCK_STYLE=rocm72
      SKIP_MEGA_ON_BUILD_FAIL=1
      CDNA1_SMOKE_WARN=1
      SOC_LEAVES=(MI100)
      VENV_KIND=site-single-pass-first
      ROOF_NOTE="; MI100 has no roofline microbenchmarks"
      META_SITE=1
      ;;
    gfx90a)
      ENV_FAMILY=site
      NATIVE_MODE=force
      THEROCK_STYLE=rocm72
      SKIP_MEGA_ON_BUILD_FAIL=1
      SOC_LEAVES=(MI250 MI250X)
      VENV_KIND=site-home-venv-first
      META_SITE=1
      ;;
    gfx950)
      ENV_FAMILY=site
      NATIVE_MODE=force
      THEROCK_STYLE=rocm71
      SOC_LEAVES=(MI350 MI350X MI355X)
      VENV_KIND=site-single-pass-first
      META_SITE=1
      ;;
    *)
      echo "Unknown ARCH: ${ARCH}" >&2
      usage >&2
      exit 1
      ;;
  esac
}

apply_worker_profile

TAG=${TAG_ARG:-$(date +%y%m%d)-${ARCH}-500}
ROOT=${ROOT_ARG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PROFILER_ROOT=${PROFILER_ROOT_ARG:-$ROOT}
RUN=${RUN_ARG:-single-pass}
ITERS=${ITERS_ARG:-500}
if [[ "$ROOF" == "1" ]]; then
  NO_ROOF=0
else
  NO_ROOF=1
fi
MEGA_BATCH=${MEGA_BATCH_ARG:-$MEGA_BATCH_DEFAULT}
LOCAL_PREFIX=${LOCAL_PREFIX_ARG:-/tmp/rpc_local}
SITE_ROCM_CORE=${SITE_ROCM_ARG:-/opt/rocm}
if [[ "$SMOKE_FOLLOWS_MEGA_BATCH" == "1" ]]; then
  SMOKE_BATCH=$MEGA_BATCH
else
  SMOKE_BATCH=4096
fi
if [[ "$NO_NATIVE_TOOL" == "1" || "$NATIVE_MODE" == "force" ]]; then
  FORCE_NO_NATIVE_TOOL=1
else
  FORCE_NO_NATIVE_TOOL=0
fi

case "$THEROCK_STYLE" in
  apu)
    if [[ -n "$THEROCK_ARG" ]]; then
      THEROCK=$THEROCK_ARG
    elif [[ -d "${HOME}/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel" ]]; then
      THEROCK=${HOME}/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel
    else
      THEROCK=${HOME}/rocm-venv
    fi
    ;;
  fixed)
    THEROCK=${THEROCK_ARG:-${HOME}/workdir/therock-work/therock-rocm-7.15.0a20260728}
    ;;
  rocm72)
    THEROCK=${THEROCK_ARG:-/opt/rocm-7.2.0}
    ;;
  rocm71)
    THEROCK=${THEROCK_ARG:-/opt/rocm-7.1.0}
    ;;
  *)
    echo "Internal error: THEROCK_STYLE=${THEROCK_STYLE}" >&2
    exit 1
    ;;
esac

if [[ "$LOG_STYLE" == "therock" ]]; then
  LOG=${LOG_ARG:-${HOME}/workdir/metric_health_${TAG}.log}
else
  LOG=${LOG_ARG:-${HOME}/metric_health_${TAG}.log}
fi

: >"$LOG"
exec >>"$LOG" 2>&1

echo "=== START $(date -u) tag=$TAG arch=$ARCH run=$RUN iters=$ITERS no_roof=$NO_ROOF root=$ROOT profiler_root=$PROFILER_ROOT therock=$THEROCK ==="
if [[ "$META_SITE" == "1" ]]; then
  echo "host=$(hostname) ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-}"
else
  echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-}"
fi
if [[ "$SMOKE_FOLLOWS_MEGA_BATCH" == "1" ]]; then
  echo "WORKLOAD_ITERS vcopy=-i ${ITERS} nbody=arg2 ${ITERS} mega_kernel=-n ${ITERS} mega_batch=${MEGA_BATCH}"
else
  echo "WORKLOAD_ITERS vcopy=-i ${ITERS} nbody=arg2 ${ITERS} mega_kernel=-n ${ITERS}"
fi

link_pkg_config() {
  if [[ ! -x "${LOCAL_PREFIX}/bin/pkg-config" && -x "${LOCAL_PREFIX}/bin/pkgconf" ]]; then
    ln -sf pkgconf "${LOCAL_PREFIX}/bin/pkg-config"
  fi
}

export_rocm_common() {
  export ROCM_PATH="${THEROCK}"
  export HIP_PATH="${THEROCK}"
  export ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-0}
  export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
  export PYTHONPATH="${PROFILER_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
  export CMAKE_PREFIX_PATH="${THEROCK}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  if [[ -d "${THEROCK}/lib/rocm_sysdeps/lib/pkgconfig" ]]; then
    export PKG_CONFIG_PATH="${THEROCK}/lib/rocm_sysdeps/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
  fi
  export PKG_CONFIG_EXECUTABLE="${PKG_CONFIG_EXECUTABLE:-$(command -v pkg-config || true)}"
}

source_site_venvs() {
  local venv
  local -a venvs
  if [[ "$VENV_KIND" == "site-home-venv-first" ]]; then
    venvs=(
      "${HOME}/rocprofiler-compute-venv/bin/activate"
      "${HOME}/rocprofiler-compute/venv-single-pass/bin/activate"
      "${HOME}/rocm-systems/projects/rocprofiler-compute/venv/bin/activate"
      /tmp/rpc_venv/bin/activate
    )
  else
    venvs=(
      "${HOME}/rocprofiler-compute/venv-single-pass/bin/activate"
      "${HOME}/rocprofiler-compute-venv/bin/activate"
      "${HOME}/rocm-systems/projects/rocprofiler-compute/venv/bin/activate"
      /tmp/rpc_venv/bin/activate
    )
  fi
  for venv in "${venvs[@]}"; do
    if [[ -f "$venv" ]]; then
      # shellcheck disable=SC1090
      . "$venv"
      echo "venv: $venv"
      break
    fi
  done
}

setup_env() {
  link_pkg_config
  case "$ENV_FAMILY" in
    apu)
      if [[ -f "${HOME}/rocm-venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        . "${HOME}/rocm-venv/bin/activate"
      fi
      if [[ -f /tmp/rpc_venv/bin/activate ]]; then
        # shellcheck disable=SC1091
        . /tmp/rpc_venv/bin/activate
      fi
      export PATH="${LOCAL_PREFIX}/bin:${THEROCK}/bin:${HOME}/rocm-venv/bin:${PATH:-}"
      export LD_LIBRARY_PATH="${THEROCK}/lib:${THEROCK}/lib/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      export_rocm_common
      cd "$ROOT"
      CMAKE_PIP_BIN=${HOME}/rocm-venv/lib/python3.12/site-packages/cmake/data/bin
      if [[ -x "${CMAKE_PIP_BIN}/cmake" ]]; then
        export PATH="${CMAKE_PIP_BIN}:${PATH}"
      fi
      ;;
    therock)
      export PATH="${LOCAL_PREFIX}/bin:${THEROCK}/bin:${PATH:-}"
      export LD_LIBRARY_PATH="${THEROCK}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      export_rocm_common
      cd "$ROOT"
      if [[ -f /tmp/rpc_venv/bin/activate ]]; then
        # shellcheck disable=SC1091
        . /tmp/rpc_venv/bin/activate
      fi
      CMAKE_PIP_BIN=/tmp/rpc_venv/lib/python3.12/site-packages/cmake/data/bin
      if [[ -x "${CMAKE_PIP_BIN}/cmake" ]]; then
        export PATH="${CMAKE_PIP_BIN}:/tmp/rpc_venv/bin:${PATH}"
      elif [[ -x /tmp/rpc_venv/bin/cmake ]]; then
        export PATH="/tmp/rpc_venv/bin:${PATH}"
      fi
      ;;
    site)
      export PATH="${LOCAL_PREFIX}/bin:${THEROCK}/bin:${SITE_ROCM_CORE}/bin:${PATH:-}"
      export LD_LIBRARY_PATH="${THEROCK}/lib:${THEROCK}/lib64:${SITE_ROCM_CORE}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      export_rocm_common
      cd "$ROOT"
      source_site_venvs
      ;;
    *)
      echo "Internal error: ENV_FAMILY=${ENV_FAMILY}" >&2
      exit 1
      ;;
  esac
}

toolchain_probe() {
  echo "--- toolchain ---"
  command -v hipcc && hipcc --version 2>&1 | head -3 || true
  command -v rocprofv3 && rocprofv3 --version 2>&1 | head -3 || true
  command -v pkg-config && pkg-config --version || echo "WARN: pkg-config missing"
  if command -v pkg-config >/dev/null 2>&1; then
    pkg-config --exists libdw && echo "pkg-config libdw: OK" || echo "WARN: libdw.pc not found"
  fi
  if [[ -x "${THEROCK}/bin/cmake" ]]; then
    export PATH="${THEROCK}/bin:${PATH}"
  fi
  if ! command -v cmake >/dev/null 2>&1; then
    if [[ "$ENV_FAMILY" == "therock" ]]; then
      echo "WARN: cmake not on PATH — attempting pip install into /tmp/rpc_venv"
      if [[ -x /tmp/rpc_venv/bin/pip ]]; then
        /tmp/rpc_venv/bin/pip install -q cmake || true
        export PATH="/tmp/rpc_venv/bin:${CMAKE_PIP_BIN}:${PATH}"
      fi
    else
      echo "WARN: cmake not on PATH — attempting pip install"
      if command -v pip >/dev/null 2>&1; then
        pip install -q cmake || true
      fi
    fi
  fi
  if command -v cmake >/dev/null 2>&1; then
    echo "cmake: $(command -v cmake)"
    cmake --version 2>&1 | head -1 || true
  else
    echo "WARN: cmake still missing after venv probe"
  fi
  python3 --version
  echo "python3: $(command -v python3)"
  if [[ "$ENV_FAMILY" == "apu" ]]; then
    rocminfo 2>/dev/null | grep -E 'Name:|gfx' | head -15 || true
  fi
}

toolchain_site() {
  echo "--- toolchain ---"
  command -v hipcc && hipcc --version 2>&1 | head -3 || true
  command -v rocprofv3 && rocprofv3 --version 2>&1 | head -3 || true
  python3 --version
  echo "python3: $(command -v python3)"
}

resolve_rocprof_compute() {
  # Prefer a built/installed rocprof-compute over running from source. Checked
  # in order: an install under --profiler-root (either the checkout root with
  # an install/ subdir per README.md, or --profiler-root given directly as the
  # install prefix), then the --therock prefix (ROCm/TheRock may already ship
  # a built rocprof-compute), then PATH. Only fall back to the uninstalled
  # source tree if none of those resolve, and say so loudly.
  #
  # An explicit --profiler-root pins resolution to that tree. ${THEROCK}/bin and
  # PATH are the same for every run on a host, so a baseline run pointed at a
  # rocprofiler-compute-develop checkout would otherwise profile with the very
  # binary the candidate run uses and report a comparison of a build against
  # itself. Those two entries stay in the chain only when --profiler-root is
  # left at its default, where there is no second tree to confuse it with.
  local candidate
  local -a candidates=(
    "${PROFILER_ROOT}/install/bin/rocprof-compute"
    "${PROFILER_ROOT}/bin/rocprof-compute"
  )
  local pinned=0
  if [[ -n "$PROFILER_ROOT_ARG" ]]; then
    pinned=1
  else
    candidates+=("${THEROCK}/bin/rocprof-compute")
  fi
  ROCPROF_COMPUTE_CMD=()
  ROCPROF_COMPUTE_SOURCE=""
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      ROCPROF_COMPUTE_CMD=("$candidate")
      ROCPROF_COMPUTE_SOURCE="$candidate"
      break
    fi
  done
  if ((${#ROCPROF_COMPUTE_CMD[@]} == 0)) && [[ "$pinned" == "0" ]] \
    && command -v rocprof-compute >/dev/null 2>&1; then
    ROCPROF_COMPUTE_CMD=("$(command -v rocprof-compute)")
    ROCPROF_COMPUTE_SOURCE="${ROCPROF_COMPUTE_CMD[0]} (PATH)"
  fi
  if ((${#ROCPROF_COMPUTE_CMD[@]} == 0)); then
    ROCPROF_COMPUTE_CMD=(python3 "${PROFILER_ROOT}/src/rocprof-compute")
    ROCPROF_COMPUTE_SOURCE="source fallback: ${PROFILER_ROOT}/src/rocprof-compute"
    if [[ "$pinned" == "1" ]]; then
      echo "WARN: no built rocprof-compute under \${PROFILER_ROOT}/install/bin or" \
        "\${PROFILER_ROOT}/bin — falling back to the source tree of the pinned" \
        "--profiler-root (\${THEROCK}/bin and PATH are skipped because they are" \
        "shared by every run on this host)"
    else
      echo "WARN: no built rocprof-compute found under \${PROFILER_ROOT}/install/bin," \
        "\${PROFILER_ROOT}/bin, \${THEROCK}/bin, or PATH — falling back to source tree" \
        "(build + install rocprof-compute into one of those locations to use the built binary)"
    fi
  fi
  echo "rocprof-compute: ${ROCPROF_COMPUTE_SOURCE}"
}

decide_native() {
  PROFILE_EXTRA=()
  USE_NATIVE_TOOL=false
  if [[ "$NATIVE_MODE" == "force" ]]; then
    if [[ "$FORCE_NO_NATIVE_TOOL" == "1" ]]; then
      PROFILE_EXTRA=(--no-native-tool)
      echo "native-tool: forced off (--no-native-tool)"
    elif ! command -v cmake >/dev/null 2>&1; then
      echo "WARN: cmake missing — using --no-native-tool"
      PROFILE_EXTRA=(--no-native-tool)
    fi
  else
    if ! command -v cmake >/dev/null 2>&1; then
      echo "WARN: cmake missing — using --no-native-tool"
      PROFILE_EXTRA=(--no-native-tool)
    elif ! command -v pkg-config >/dev/null 2>&1; then
      echo "WARN: pkg-config missing — using --no-native-tool"
      PROFILE_EXTRA=(--no-native-tool)
    elif ! pkg-config --exists libdw 2>/dev/null; then
      echo "WARN: libdw.pc missing — using --no-native-tool"
      PROFILE_EXTRA=(--no-native-tool)
    else
      echo "native-tool: cmake + pkg-config libdw OK — will attempt native-tool first"
    fi
    USE_NATIVE_TOOL=true
    if ((${#PROFILE_EXTRA[@]})) && [[ "${PROFILE_EXTRA[0]}" == "--no-native-tool" ]]; then
      USE_NATIVE_TOOL=false
    fi
  fi
  if [[ "$NO_ROOF" == "1" ]]; then
    PROFILE_EXTRA+=(--no-roof)
    echo "roofline: disabled (--no-roof)${ROOF_NOTE}"
  fi
}

smoke_mega() {
  set +e
  "./sample/mega_kernel/mega_kernel_test_${ARCH}" -b "${SMOKE_BATCH}" -n 1
  mk_rc=$?
  set -e
  echo "MEGA_SMOKE_OK mega_kernel_rc=$mk_rc"
  if [[ "$CDNA1_SMOKE_WARN" == "1" && "$mk_rc" -ne 0 ]]; then
    echo "WARN: mega_kernel smoke non-zero — continuing profiles anyway (bypasses OK on CDNA1)"
  fi
}

prepare_mega() {
  echo "--- mega_kernel smoke ---"
  SKIP_MEGA=0
  make -C sample/mega_kernel clean || true
  if [[ "$SKIP_MEGA_ON_BUILD_FAIL" == "1" ]]; then
    set +e
    make -C sample/mega_kernel "$ARCH"
    mk_build=$?
    set -e
    if [[ $mk_build -ne 0 || ! -x "./sample/mega_kernel/mega_kernel_test_${ARCH}" ]]; then
      echo "WARN: mega_kernel ${ARCH} build failed rc=$mk_build — skipping mega_kernel profiles"
      SKIP_MEGA=1
      return 0
    fi
    smoke_mega
    return 0
  fi
  make -C sample/mega_kernel "$ARCH"
  smoke_mega
}

build_apps() {
  echo "--- build apps ---"
  hipcc -O3 -std=c++17 --offload-arch="${ARCH}" -o sample/vc sample/vcopy.cpp
  hipcc -O2 -std=c++17 -DSHMOO --offload-arch="${ARCH}" \
    -o sample/mini-nbody-block sample/mini-nbody-block.cpp
  chmod +x sample/mini-nbody-block
  ./sample/vc -n 65536 -b 256 -i 1 >/dev/null
  ./sample/mini-nbody-block 16384 1 >/dev/null || true
  echo "APPS_OK"
}

write_run_meta() {
  LOGDIR=${ROOT}/workloads/logs_${ARCH}_${TAG}
  mkdir -p "$LOGDIR"
  {
    echo "TAG=${TAG}"
    echo "ARCH=${ARCH}"
    echo "RUN=${RUN}"
    echo "PROFILER_ROOT=${PROFILER_ROOT}"
    echo "ROCPROF_COMPUTE_SOURCE=${ROCPROF_COMPUTE_SOURCE}"
    echo "ITERS=${ITERS}"
    echo "NO_ROOF=${NO_ROOF}"
    if [[ "$META_SITE" == "1" ]]; then
      echo "HOST=$(hostname)"
      echo "THEROCK=${THEROCK}"
    fi
    echo "VCOPY_ARGS=-n 81920 -b 256 -i ${ITERS}"
    echo "NBODY_ARGS=81920 ${ITERS}"
    echo "MEGA_KERNEL_ARGS=-b ${MEGA_BATCH} -n ${ITERS}"
    echo "STAT=Median"
  } >"${LOGDIR}/run_meta.env"
}

dir_has_workload() {
  [[ -f "${1}/sysinfo.csv" || -d "${1}/perfmon" ]]
}

consider_candidates() {
  local wl_name=$1 leaf cand
  for cand in "workloads/${wl_name}" "workloads/${wl_name}/0"; do
    if dir_has_workload "$cand"; then
      printf '%s' "$cand"
      return 0
    fi
  done
  for leaf in "${SOC_LEAVES[@]}"; do
    cand="workloads/${wl_name}/${leaf}"
    if dir_has_workload "$cand"; then
      printf '%s' "$cand"
      return 0
    fi
  done
  return 1
}

find_sysinfo_dir() {
  local wl_name=$1
  find "workloads/${wl_name}" -maxdepth 3 -type f -name sysinfo.csv 2>/dev/null \
    | head -1 | xargs -r dirname || true
}

resolve_wl_dir() {
  local wl_name=$1
  local wl_dir=""
  if [[ "$WL_SEARCH" == "find" ]]; then
    wl_dir=$(find_sysinfo_dir "$wl_name")
  fi
  if [[ -z "$wl_dir" ]]; then
    wl_dir=$(consider_candidates "$wl_name" || true)
  fi
  if [[ -z "$wl_dir" && "$WL_SEARCH" == "candidates" ]]; then
    wl_dir=$(find_sysinfo_dir "$wl_name")
  fi
  if [[ -z "$wl_dir" && "$WL_TREE_FALLBACK" == "1" ]]; then
    wl_dir=$(find workloads -maxdepth 4 -type f -path "*/${wl_name}/*/sysinfo.csv" 2>/dev/null \
      | head -1 | xargs -r dirname || true)
  fi
  printf '%s' "$wl_dir"
}

run_one() {
  local mode=$1 name=$2
  shift 2
  local wl_name="${name}_${TAG}_${mode}"
  local t0 dur prof_status=FAIL anal_status=FAIL
  t0=$(date +%s)

  echo "--- PROFILE mode=$mode name=$name iters=$ITERS cmd=$* ---"

  set +e
  "${PROFILE[@]}" -n "$wl_name" -- "$@"
  local prc=$?
  if [[ $prc -ne 0 && "$USE_NATIVE_TOOL" == true ]]; then
    echo "WARN: native-tool profile $wl_name failed rc=$prc — retrying with --no-native-tool"
    local retry=("${ROCPROF_COMPUTE_CMD[@]}" profile --overwrite --no-native-tool)
    if [[ "$NO_ROOF" == "1" ]]; then
      retry+=(--no-roof)
    fi
    "${retry[@]}" -n "$wl_name" -- "$@"
    prc=$?
  fi
  set -e
  if [[ $prc -eq 0 ]]; then
    prof_status=OK
  else
    echo "WARN: profile $wl_name failed rc=$prc"
  fi

  local wl_dir=""
  wl_dir=$(resolve_wl_dir "$wl_name")

  if [[ -n "$wl_dir" ]]; then
    echo "--- ANALYZE $wl_dir ---"
    set +e
    "${ROCPROF_COMPUTE_CMD[@]}" analyze -p "$wl_dir" -k 0 --view table \
      >"${LOGDIR}/${name}_${mode}.log" 2>&1
    local arc=$?
    set -e
    if [[ $arc -eq 0 ]]; then
      anal_status=OK
    else
      echo "WARN: analyze $name $mode failed rc=$arc"
    fi
    if [[ -f "${wl_dir}/pmc_dispatch_info.csv" ]]; then
      local nd
      nd=$(($(wc -l <"${wl_dir}/pmc_dispatch_info.csv") - 1))
      echo "DISPATCH_COUNT name=$name mode=$mode n=$nd iters_requested=$ITERS"
    fi
  else
    echo "WARN: no workload dir for $wl_name"
  fi

  dur=$(( $(date +%s) - t0 ))
  echo "WORKLOAD_DONE name=$name mode=$mode profile=$prof_status analyze=$anal_status duration_s=$dur iters=$ITERS"
}

setup_env
if [[ "$NATIVE_MODE" == "probe" ]]; then
  toolchain_probe
else
  toolchain_site
fi
decide_native
resolve_rocprof_compute
prepare_mega
build_apps
write_run_meta # after resolve_rocprof_compute so ROCPROF_COMPUTE_SOURCE is logged

PROFILE=("${ROCPROF_COMPUTE_CMD[@]}" profile --overwrite "${PROFILE_EXTRA[@]}")

run_one "$RUN" vcopy ./sample/vc -n 81920 -b 256 -i "$ITERS"
# 81920 bodies at 128 threads/block launch 640 blocks. On the largest current
# health target (304 CUs), that supplies at least two blocks and four wave64s
# per CU, one wave for each SIMD. The kernel tiles body positions through LDS.
run_one "$RUN" nbody ./sample/mini-nbody-block 81920 "$ITERS"
if [[ "$SKIP_MEGA" != "1" ]]; then
  run_one "$RUN" mega_kernel \
    "./sample/mega_kernel/mega_kernel_test_${ARCH}" -b "$MEGA_BATCH" -n "$ITERS"
else
  echo "WORKLOAD_DONE name=mega_kernel mode=$RUN profile=SKIP analyze=SKIP duration_s=0 iters=$ITERS"
fi

echo "ALL_DONE $(date -u) tag=$TAG iters=$ITERS"
