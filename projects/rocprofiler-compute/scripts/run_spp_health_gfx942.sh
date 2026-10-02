#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Remote worker: SPP metric-health validation on gfx942 (MI300).
# Expected to run on a GPU node with TheRock on PATH (see env below).
#
# Default workload iterations: 500 (median-stable). Override with ITERS=.
#   vcopy:       -n 81920 -b 256 -i ${ITERS}
#   nbody:       mini-nbody-block 131072 ${ITERS}
#   mega_kernel: -b 65536 -n ${ITERS}
#
# Usage (on host, after syncing rocprofiler-compute tree):
#   TAG=261001b-gfx942-500 THEROCK=/path/to/therock \
#     bash scripts/run_spp_health_gfx942.sh
#
# Produces under $ROOT/workloads/:
#   {vcopy,nbody,mega_kernel}_${TAG}_{spp,legacy}/
#   logs_gfx942_${TAG}/*.log
#   logs_gfx942_${TAG}/run_meta.env  (ITERS, TAG, arch, …)
# Ends with ALL_DONE in $LOG.

set -euo pipefail

TAG=${TAG:-$(date +%y%m%d)-gfx942-500}
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
THEROCK=${THEROCK:-/home/AMD/feizheng/aiprofcomp78/therock-work/therock-rocm-7.15.0a20260728}
LOG=${LOG:-/home/AMD/feizheng/aiprofcomp78/spp_health_${TAG}.log}
ARCH=gfx942
ITERS=${ITERS:-500}
# Skip empirical roofline by default (wall time on PMC medians). Override: NO_ROOF=0.
NO_ROOF=${NO_ROOF:-1}

: >"$LOG"
exec >>"$LOG" 2>&1

echo "=== START $(date -u) tag=$TAG arch=$ARCH iters=$ITERS no_roof=$NO_ROOF root=$ROOT therock=$THEROCK ==="
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-}"
echo "WORKLOAD_ITERS vcopy=-i ${ITERS} nbody=arg2 ${ITERS} mega_kernel=-n ${ITERS}"

# User-local pkg-config (darkstar has none system-wide); same pattern as
# scripts/run_ctest_gfx942_therock.sh. Native-tool builds need libdw via pkg-config.
LOCAL_PREFIX=${LOCAL_PREFIX:-/tmp/rpc_local}
if [[ ! -x "${LOCAL_PREFIX}/bin/pkg-config" && -x "${LOCAL_PREFIX}/bin/pkgconf" ]]; then
  ln -sf pkgconf "${LOCAL_PREFIX}/bin/pkg-config"
fi
export PATH="${LOCAL_PREFIX}/bin:${THEROCK}/bin:${PATH:-}"
export LD_LIBRARY_PATH="${THEROCK}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ROCM_PATH="${THEROCK}"
export HIP_PATH="${THEROCK}"
export ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-0}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
# Prefer TheRock hipcc/cmake for native tool builds
export CMAKE_PREFIX_PATH="${THEROCK}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
if [[ -d "${THEROCK}/lib/rocm_sysdeps/lib/pkgconfig" ]]; then
  export PKG_CONFIG_PATH="${THEROCK}/lib/rocm_sysdeps/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
fi
export PKG_CONFIG_EXECUTABLE="${PKG_CONFIG_EXECUTABLE:-$(command -v pkg-config || true)}"

cd "$ROOT"

# Optional venv for analyze deps / pip cmake (source even if not +x).
if [[ -f /tmp/rpc_venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  . /tmp/rpc_venv/bin/activate
fi
# Pip-installed cmake may live under site-packages; ensure it is on PATH.
CMAKE_PIP_BIN=/tmp/rpc_venv/lib/python3.12/site-packages/cmake/data/bin
if [[ -x "${CMAKE_PIP_BIN}/cmake" ]]; then
  export PATH="${CMAKE_PIP_BIN}:/tmp/rpc_venv/bin:${PATH}"
elif [[ -x /tmp/rpc_venv/bin/cmake ]]; then
  export PATH="/tmp/rpc_venv/bin:${PATH}"
fi

echo "--- toolchain ---"
command -v hipcc && hipcc --version 2>&1 | head -3 || true
command -v rocprofv3 && rocprofv3 --version 2>&1 | head -3 || true
command -v pkg-config && pkg-config --version || echo "WARN: pkg-config missing"
if command -v pkg-config >/dev/null 2>&1; then
  pkg-config --exists libdw && echo "pkg-config libdw: OK" || echo "WARN: libdw.pc not found"
fi
# Native-tool build needs cmake; prefer TheRock then PATH / venv.
if [[ -x "${THEROCK}/bin/cmake" ]]; then
  export PATH="${THEROCK}/bin:${PATH}"
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "WARN: cmake not on PATH — attempting pip install into /tmp/rpc_venv"
  if [[ -x /tmp/rpc_venv/bin/pip ]]; then
    /tmp/rpc_venv/bin/pip install -q cmake || true
    export PATH="/tmp/rpc_venv/bin:${CMAKE_PIP_BIN}:${PATH}"
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

# Prefer native-tool when cmake + pkg-config (libdw) look usable.
PROFILE_EXTRA=()
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
if [[ "$NO_ROOF" == "1" ]]; then
  PROFILE_EXTRA+=(--no-roof)
  echo "roofline: disabled (--no-roof) for median health wall time"
fi

echo "--- mega_kernel smoke ---"
make -C sample/mega_kernel clean || true
make -C sample/mega_kernel gfx942
# Mega-kernel may exit non-zero when a category fails; treat as smoke, not gate.
set +e
./sample/mega_kernel/mega_kernel_test_gfx942 -b 4096 -n 1
mk_rc=$?
set -e
echo "MEGA_SMOKE_OK mega_kernel_rc=$mk_rc"

echo "--- build apps ---"
hipcc -O3 -std=c++17 --offload-arch="${ARCH}" -o sample/vc sample/vcopy.cpp
hipcc -O2 -std=c++17 -DSHMOO --offload-arch="${ARCH}" \
  -o sample/mini-nbody-block sample/mini-nbody-block.cpp
chmod +x sample/mini-nbody-block
./sample/vc -n 65536 -b 256 -i 1 >/dev/null
./sample/mini-nbody-block 16384 1 >/dev/null || true
echo "APPS_OK"

LOGDIR=${ROOT}/workloads/logs_gfx942_${TAG}
mkdir -p "$LOGDIR"
cat >"${LOGDIR}/run_meta.env" <<EOF
TAG=${TAG}
ARCH=${ARCH}
ITERS=${ITERS}
NO_ROOF=${NO_ROOF}
VCOPY_ARGS=-n 81920 -b 256 -i ${ITERS}
NBODY_ARGS=131072 ${ITERS}
MEGA_KERNEL_ARGS=-b 65536 -n ${ITERS}
STAT=Median
EOF

PROFILE=(python3 src/rocprof-compute profile --overwrite "${PROFILE_EXTRA[@]}")
USE_NATIVE_TOOL=true
if ((${#PROFILE_EXTRA[@]})) && [[ "${PROFILE_EXTRA[0]}" == "--no-native-tool" ]]; then
  USE_NATIVE_TOOL=false
fi

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
  # If native-tool failed for a non-pkg-config reason, retry once without it.
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
  for cand in \
    "workloads/${wl_name}" \
    "workloads/${wl_name}/0" \
    "workloads/${wl_name}/MI300X_A1" \
    "workloads/${wl_name}/MI300A_A0"; do
    if [[ -f "${cand}/sysinfo.csv" ]] || [[ -d "${cand}/perfmon" ]]; then
      wl_dir=$cand
      break
    fi
  done
  # Prefer leaf dir that contains sysinfo.csv under the named workload tree
  if [[ -z "$wl_dir" ]]; then
    wl_dir=$(find "workloads/${wl_name}" -maxdepth 3 -type f -name sysinfo.csv 2>/dev/null \
      | head -1 | xargs -r dirname || true)
  fi
  if [[ -z "$wl_dir" ]]; then
    wl_dir=$(find workloads -maxdepth 4 -type f -path "*/${wl_name}/*/sysinfo.csv" 2>/dev/null \
      | head -1 | xargs -r dirname || true)
  fi

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
    # Dispatch count for median sample-size checks
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

for mode in spp legacy; do
  run_one "$mode" vcopy ./sample/vc -n 81920 -b 256 -i "$ITERS"
  run_one "$mode" nbody ./sample/mini-nbody-block 131072 "$ITERS"
  run_one "$mode" mega_kernel ./sample/mega_kernel/mega_kernel_test_gfx942 -b 65536 -n "$ITERS"
done

echo "ALL_DONE $(date -u) tag=$TAG iters=$ITERS"
