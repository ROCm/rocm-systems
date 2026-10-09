#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Remote worker: SPP metric health test report on one GPU architecture.
#
# Default workload iterations: 500 (median-stable). Override with ITERS=.
#   vcopy:       -n 81920 -b 256 -i ${ITERS}
#   nbody:       mini-nbody-block 131072 ${ITERS}
#   mega_kernel: -b ${MEGA_BATCH} -n ${ITERS}
#
# ARCH selects the host profile. Differences that used to be copied scripts:
#   gfx1151  mega_kernel profile batch 4096 (RDNA3.5); smoke uses that batch
#   CDNA     mega_kernel profile batch 65536; smoke stays -b 4096
#   gfx908   roofline note: MI100 has no roofline microbenchmarks
#            (NO_ROOF=1 is the default on every arch)
#   gfx908   gfx90a  skip mega_kernel when its build fails
#   gfx942   gfx1151 attempt native-tool, then retry once with --no-native-tool
#   gfx908   gfx90a gfx950  FORCE_NO_NATIVE_TOOL=1 (no retry)
#
# Usage (on the GPU host, after syncing rocprofiler-compute):
#   ARCH=gfx942 TAG=261001-gfx942-500med THEROCK=/path/to/therock \
#     bash scripts/run_spp_health.sh
#   ARCH=gfx1151 THEROCK=/path/to/rocm bash scripts/run_spp_health.sh
#   ARCH=gfx908  THEROCK=/opt/rocm-7.2.0 bash scripts/run_spp_health.sh
#   ARCH=gfx90a  THEROCK=/opt/rocm-7.2.0 bash scripts/run_spp_health.sh
#   ARCH=gfx950  THEROCK=/opt/rocm-7.1.0 bash scripts/run_spp_health.sh
#
# 500-iter medians are ITERS=500, including gfx942. Median CSV is
# tools/compare_spp_legacy_medians.py, not a separate worker.
#
# Produces under $ROOT/workloads/:
#   {vcopy,nbody,mega_kernel}_${TAG}_{spp,legacy}/
#   logs_${ARCH}_${TAG}/*.log
#   logs_${ARCH}_${TAG}/run_meta.env
# Ends with ALL_DONE in $LOG.

set -euo pipefail

usage() {
  cat <<EOF
Usage: ARCH=<gfx> [ITERS=500] [NO_ROOF=1] [THEROCK=...] bash $(basename "$0")
       bash $(basename "$0") <gfx942|gfx950|gfx90a|gfx908|gfx1151>

Env: TAG ITERS NO_ROOF MEGA_BATCH THEROCK LOG ROOT
     FORCE_NO_NATIVE_TOOL (site arches default 1)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == gfx* ]]; then
  ARCH=$1
  shift
fi
if [[ $# -gt 0 ]]; then
  echo "Unknown: $1" >&2
  usage >&2
  exit 1
fi
ARCH=${ARCH:-}
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
      ENV_FAMILY=darkstar
      NATIVE_MODE=probe
      THEROCK_STYLE=fixed
      LOG_STYLE=darkstar
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
      VENV_KIND=site-spp-first
      ROOF_NOTE="; MI100 has no roofline microbenchmarks"
      META_SITE=1
      ;;
    gfx90a)
      ENV_FAMILY=site
      NATIVE_MODE=force
      THEROCK_STYLE=rocm72
      SKIP_MEGA_ON_BUILD_FAIL=1
      SOC_LEAVES=(MI250 MI250X)
      VENV_KIND=site-865-first
      META_SITE=1
      ;;
    gfx950)
      ENV_FAMILY=site
      NATIVE_MODE=force
      THEROCK_STYLE=rocm71
      SOC_LEAVES=(MI350 MI350X MI355X)
      VENV_KIND=site-spp-first
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

TAG=${TAG:-$(date +%y%m%d)-${ARCH}-500}
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ITERS=${ITERS:-500}
NO_ROOF=${NO_ROOF:-1}
MEGA_BATCH=${MEGA_BATCH:-$MEGA_BATCH_DEFAULT}
if [[ "$SMOKE_FOLLOWS_MEGA_BATCH" == "1" ]]; then
  SMOKE_BATCH=$MEGA_BATCH
else
  SMOKE_BATCH=4096
fi
if [[ "$NATIVE_MODE" == "force" ]]; then
  FORCE_NO_NATIVE_TOOL=${FORCE_NO_NATIVE_TOOL:-1}
fi

case "$THEROCK_STYLE" in
  apu)
    if [[ -z "${THEROCK:-}" ]]; then
      if [[ -d "${HOME}/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel" ]]; then
        THEROCK=${HOME}/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel
      else
        THEROCK=${HOME}/rocm-venv
      fi
    fi
    ;;
  fixed)
    THEROCK=${THEROCK:-/home/AMD/feizheng/aiprofcomp78/therock-work/therock-rocm-7.15.0a20260728}
    ;;
  rocm72)
    THEROCK=${THEROCK:-${ROCM_ROOT:-/opt/rocm-7.2.0}}
    ;;
  rocm71)
    THEROCK=${THEROCK:-${ROCM_ROOT:-/opt/rocm-7.1.0}}
    ;;
  *)
    echo "Internal error: THEROCK_STYLE=${THEROCK_STYLE}" >&2
    exit 1
    ;;
esac

if [[ "$LOG_STYLE" == "darkstar" ]]; then
  LOG=${LOG:-/home/AMD/feizheng/aiprofcomp78/spp_health_${TAG}.log}
else
  LOG=${LOG:-${HOME}/spp_health_${TAG}.log}
fi

: >"$LOG"
exec >>"$LOG" 2>&1

echo "=== START $(date -u) tag=$TAG arch=$ARCH iters=$ITERS no_roof=$NO_ROOF root=$ROOT therock=$THEROCK ==="
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
  LOCAL_PREFIX=${LOCAL_PREFIX:-/tmp/rpc_local}
  if [[ ! -x "${LOCAL_PREFIX}/bin/pkg-config" && -x "${LOCAL_PREFIX}/bin/pkgconf" ]]; then
    ln -sf pkgconf "${LOCAL_PREFIX}/bin/pkg-config"
  fi
}

export_rocm_common() {
  export ROCM_PATH="${THEROCK}"
  export HIP_PATH="${THEROCK}"
  export ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-0}
  export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
  export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
  export CMAKE_PREFIX_PATH="${THEROCK}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  if [[ -d "${THEROCK}/lib/rocm_sysdeps/lib/pkgconfig" ]]; then
    export PKG_CONFIG_PATH="${THEROCK}/lib/rocm_sysdeps/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
  fi
  export PKG_CONFIG_EXECUTABLE="${PKG_CONFIG_EXECUTABLE:-$(command -v pkg-config || true)}"
}

source_site_venvs() {
  local venv
  local -a venvs
  if [[ "$VENV_KIND" == "site-865-first" ]]; then
    venvs=(
      "${HOME}/aiprofcomp865-venv/bin/activate"
      "${HOME}/aiprofcomp865/venv-spp/bin/activate"
      "${HOME}/rocm-systems/projects/rocprofiler-compute/venv/bin/activate"
      /tmp/rpc_venv/bin/activate
    )
  else
    venvs=(
      "${HOME}/aiprofcomp865/venv-spp/bin/activate"
      "${HOME}/aiprofcomp865-venv/bin/activate"
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
    darkstar)
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
      SITE_ROCM_CORE=${SITE_ROCM_CORE:-/opt/rocm}
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
    if [[ "$ENV_FAMILY" == "darkstar" ]]; then
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

decide_native() {
  PROFILE_EXTRA=()
  USE_NATIVE_TOOL=false
  if [[ "$NATIVE_MODE" == "force" ]]; then
    if [[ "$FORCE_NO_NATIVE_TOOL" == "1" ]]; then
      PROFILE_EXTRA=(--no-native-tool)
      echo "native-tool: forced off (FORCE_NO_NATIVE_TOOL=1)"
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
    echo "ITERS=${ITERS}"
    echo "NO_ROOF=${NO_ROOF}"
    if [[ "$META_SITE" == "1" ]]; then
      echo "HOST=$(hostname)"
      echo "THEROCK=${THEROCK}"
    fi
    echo "VCOPY_ARGS=-n 81920 -b 256 -i ${ITERS}"
    echo "NBODY_ARGS=131072 ${ITERS}"
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
  unset ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC || true
  unset ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE || true
  if [[ "$mode" == "legacy" ]]; then
    export ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC=1
  fi

  set +e
  "${PROFILE[@]}" -n "$wl_name" -- "$@"
  local prc=$?
  if [[ $prc -ne 0 && "$USE_NATIVE_TOOL" == true ]]; then
    echo "WARN: native-tool profile $wl_name failed rc=$prc — retrying with --no-native-tool"
    local retry=(python3 src/rocprof-compute profile --overwrite --no-native-tool)
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
    python3 src/rocprof-compute analyze -p "$wl_dir" -k 0 --view table \
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
prepare_mega
build_apps
write_run_meta

PROFILE=(python3 src/rocprof-compute profile --overwrite "${PROFILE_EXTRA[@]}")

for mode in spp legacy; do
  run_one "$mode" vcopy ./sample/vc -n 81920 -b 256 -i "$ITERS"
  run_one "$mode" nbody ./sample/mini-nbody-block 131072 "$ITERS"
  if [[ "$SKIP_MEGA" != "1" ]]; then
    run_one "$mode" mega_kernel \
      "./sample/mega_kernel/mega_kernel_test_${ARCH}" -b "$MEGA_BATCH" -n "$ITERS"
  else
    echo "WORKLOAD_DONE name=mega_kernel mode=$mode profile=SKIP analyze=SKIP duration_s=0 iters=$ITERS"
  fi
done

echo "ALL_DONE $(date -u) tag=$TAG iters=$ITERS"
