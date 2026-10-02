#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Remote worker: SPP metric-health validation on gfx950 (MI350/MI355).
# Adapted from scripts/run_spp_health_gfx942.sh for Alola site ROCm / TheRock.
#
# Default workload iterations: 500 (median-stable). Override with ITERS=.
#   vcopy:       -n 81920 -b 256 -i ${ITERS}
#   nbody:       mini-nbody-block 131072 ${ITERS}
#   mega_kernel: -b 65536 -n ${ITERS}
#
# Usage (on GPU node, after syncing rocprofiler-compute tree):
#   TAG=261002-gfx950-500med THEROCK=/opt/rocm-7.1.0 \
#     bash scripts/run_spp_health_gfx950.sh
#
# Ends with ALL_DONE in $LOG.

set -euo pipefail

TAG=${TAG:-$(date +%y%m%d)-gfx950-500}
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
THEROCK=${THEROCK:-${ROCM_ROOT:-/opt/rocm-7.1.0}}
LOG=${LOG:-${HOME}/spp_health_${TAG}.log}
ARCH=gfx950
ITERS=${ITERS:-500}
NO_ROOF=${NO_ROOF:-1}
FORCE_NO_NATIVE_TOOL=${FORCE_NO_NATIVE_TOOL:-1}

: >"$LOG"
exec >>"$LOG" 2>&1

echo "=== START $(date -u) tag=$TAG arch=$ARCH iters=$ITERS no_roof=$NO_ROOF root=$ROOT therock=$THEROCK ==="
echo "host=$(hostname) ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-}"
echo "WORKLOAD_ITERS vcopy=-i ${ITERS} nbody=arg2 ${ITERS} mega_kernel=-n ${ITERS}"

LOCAL_PREFIX=${LOCAL_PREFIX:-/tmp/rpc_local}
if [[ ! -x "${LOCAL_PREFIX}/bin/pkg-config" && -x "${LOCAL_PREFIX}/bin/pkgconf" ]]; then
  ln -sf pkgconf "${LOCAL_PREFIX}/bin/pkg-config"
fi

SITE_ROCM_CORE=${SITE_ROCM_CORE:-/opt/rocm}
export PATH="${LOCAL_PREFIX}/bin:${THEROCK}/bin:${SITE_ROCM_CORE}/bin:${PATH:-}"
export LD_LIBRARY_PATH="${THEROCK}/lib:${THEROCK}/lib64:${SITE_ROCM_CORE}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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

cd "$ROOT"

for venv in \
  "${HOME}/aiprofcomp865/venv-spp/bin/activate" \
  "${HOME}/aiprofcomp865-venv/bin/activate" \
  "${HOME}/rocm-systems/projects/rocprofiler-compute/venv/bin/activate" \
  /tmp/rpc_venv/bin/activate; do
  if [[ -f "$venv" ]]; then
    # shellcheck disable=SC1090
    . "$venv"
    echo "venv: $venv"
    break
  fi
done

echo "--- toolchain ---"
command -v hipcc && hipcc --version 2>&1 | head -3 || true
command -v rocprofv3 && rocprofv3 --version 2>&1 | head -3 || true
python3 --version
echo "python3: $(command -v python3)"

PROFILE_EXTRA=()
if [[ "$FORCE_NO_NATIVE_TOOL" == "1" ]]; then
  PROFILE_EXTRA=(--no-native-tool)
  echo "native-tool: forced off (FORCE_NO_NATIVE_TOOL=1)"
elif ! command -v cmake >/dev/null 2>&1; then
  echo "WARN: cmake missing — using --no-native-tool"
  PROFILE_EXTRA=(--no-native-tool)
fi
if [[ "$NO_ROOF" == "1" ]]; then
  PROFILE_EXTRA+=(--no-roof)
  echo "roofline: disabled (--no-roof) for median health wall time"
fi

echo "--- mega_kernel smoke ---"
make -C sample/mega_kernel clean || true
make -C sample/mega_kernel gfx950
set +e
./sample/mega_kernel/mega_kernel_test_gfx950 -b 4096 -n 1
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

LOGDIR=${ROOT}/workloads/logs_gfx950_${TAG}
mkdir -p "$LOGDIR"
cat >"${LOGDIR}/run_meta.env" <<EOF
TAG=${TAG}
ARCH=${ARCH}
ITERS=${ITERS}
NO_ROOF=${NO_ROOF}
HOST=$(hostname)
THEROCK=${THEROCK}
VCOPY_ARGS=-n 81920 -b 256 -i ${ITERS}
NBODY_ARGS=131072 ${ITERS}
MEGA_KERNEL_ARGS=-b 65536 -n ${ITERS}
STAT=Median
EOF

PROFILE=(python3 src/rocprof-compute profile --overwrite "${PROFILE_EXTRA[@]}")

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
  set -e
  if [[ $prc -eq 0 ]]; then
    prof_status=OK
  else
    echo "WARN: profile $wl_name failed rc=$prc"
  fi

  local wl_dir=""
  wl_dir=$(find "workloads/${wl_name}" -maxdepth 3 -type f -name sysinfo.csv 2>/dev/null \
    | head -1 | xargs -r dirname || true)
  if [[ -z "$wl_dir" ]]; then
    for cand in \
      "workloads/${wl_name}" \
      "workloads/${wl_name}/0" \
      "workloads/${wl_name}/MI350" \
      "workloads/${wl_name}/MI350X" \
      "workloads/${wl_name}/MI355X"; do
      if [[ -f "${cand}/sysinfo.csv" ]] || [[ -d "${cand}/perfmon" ]]; then
        wl_dir=$cand
        break
      fi
    done
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
  run_one "$mode" mega_kernel ./sample/mega_kernel/mega_kernel_test_gfx950 -b 65536 -n "$ITERS"
done

echo "ALL_DONE $(date -u) tag=$TAG iters=$ITERS"
