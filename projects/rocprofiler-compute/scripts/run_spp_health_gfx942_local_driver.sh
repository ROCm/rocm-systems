#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Local driver for gfx942 SPP health validation on darkstar (TheRock).
# Default: 500 iterations per workload (median-stable), TAG …-gfx942-500.
#
#   ./scripts/run_spp_health_gfx942_local_driver.sh            # sync + start
#   TAG=261001-gfx942-500med ./scripts/run_spp_health_gfx942_local_driver.sh \
#     --fetch --report   # reuse existing artifacts / regen HTML

set -euo pipefail

PROXY=${PROXY:-'nc -X 5 -x 127.0.0.1:1080 %h %p'}
HOST=${HOST:-hpe-darkstar-ccs-aus-e12-03.cs-aus.dcgpu}
REMOTE_USER=${REMOTE_USER:-feizheng}
REMOTE_ROOT=${REMOTE_ROOT:-/home/AMD/feizheng/aiprofcomp78/rocm-systems/projects/rocprofiler-compute}
THEROCK=${THEROCK:-/home/AMD/feizheng/aiprofcomp78/therock-work/therock-rocm-7.15.0a20260728}
TAG=${TAG:-$(date +%y%m%d)-gfx942-500}
ITERS=${ITERS:-500}
NO_ROOF=${NO_ROOF:-1}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS=${ARTIFACTS:-${REPO_ROOT}/validation-artifacts/spp-health-${TAG}}
REMOTE_LOG=/home/AMD/feizheng/aiprofcomp78/spp_health_${TAG}.log
REMOTE_NOHUP=/home/AMD/feizheng/aiprofcomp78/spp_health_${TAG}.nohup

DO_WAIT=false
DO_FETCH=false
DO_REPORT=false

usage() {
  cat <<EOF
Usage: $(basename "$0") [--wait] [--fetch] [--report]
Env: TAG ITERS=${ITERS} NO_ROOF=${NO_ROOF} ROCR_VISIBLE_DEVICES ARTIFACTS
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait) DO_WAIT=true ;;
    --fetch) DO_FETCH=true ;;
    --report) DO_REPORT=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown: $1" >&2; usage >&2; exit 1 ;;
  esac
  shift
done

ssh_cmd() {
  ssh -o BatchMode=yes -o ProxyCommand="$PROXY" "${REMOTE_USER}@${HOST}" "$@"
}

rsync_rpc() {
  rsync -az -e "ssh -o BatchMode=yes -o ProxyCommand='$PROXY'" \
    --exclude 'workloads/' \
    --exclude '.git/' \
    --exclude '.venv*/' \
    --exclude 'graphify-out/' \
    --exclude '*.html' \
    --exclude 'validation-artifacts/' \
    "$@"
}

sync_and_start() {
  echo "=== rsync -> ${HOST}:${REMOTE_ROOT}/ ==="
  rsync_rpc "${REPO_ROOT}/" "${REMOTE_USER}@${HOST}:${REMOTE_ROOT}/"

  local gpu="${ROCR_VISIBLE_DEVICES:-17}"
  echo "=== start remote health run (TAG=${TAG} ROCR=${gpu} ITERS=${ITERS}) ==="
  ssh_cmd "chmod +x '${REMOTE_ROOT}/scripts/run_spp_health_gfx942.sh' && \
    nohup env TAG='${TAG}' ROOT='${REMOTE_ROOT}' THEROCK='${THEROCK}' \
      LOCAL_PREFIX='${LOCAL_PREFIX:-/tmp/rpc_local}' \
      ROCR_VISIBLE_DEVICES='${gpu}' HIP_VISIBLE_DEVICES='0' \
      ITERS='${ITERS}' NO_ROOF='${NO_ROOF}' \
      bash '${REMOTE_ROOT}/scripts/run_spp_health_gfx942.sh' \
      >'${REMOTE_NOHUP}' 2>&1 & echo PID=\$!"
  echo "Remote log: ${REMOTE_LOG}"
  echo "Re-run: TAG=${TAG} $(basename "$0") --wait --fetch --report"
}

wait_for_done() {
  echo "=== waiting for ALL_DONE in ${REMOTE_LOG} ==="
  while true; do
    if ssh_cmd "grep -q 'ALL_DONE' '${REMOTE_LOG}' 2>/dev/null"; then
      echo "ALL_DONE"
      ssh_cmd "grep -E 'WORKLOAD_DONE|ALL_DONE|MEGA_SMOKE|WARN:|DISPATCH_COUNT|iters=' '${REMOTE_LOG}' | tail -40" || true
      return 0
    fi
    ssh_cmd "grep -E 'WORKLOAD_DONE|PROFILE |MEGA_SMOKE|WARN:|DISPATCH_COUNT' '${REMOTE_LOG}' 2>/dev/null | tail -5" || true
    sleep 120
  done
}

fetch_results() {
  mkdir -p "${ARTIFACTS}/gfx942" "${ARTIFACTS}/logs/gfx942"
  echo "=== fetch workloads + logs ==="
  rsync -az -e "ssh -o BatchMode=yes -o ProxyCommand='$PROXY'" \
    "${REMOTE_USER}@${HOST}:${REMOTE_ROOT}/workloads/logs_gfx942_${TAG}/" \
    "${ARTIFACTS}/logs/gfx942/" || true
  rsync -az -e "ssh -o BatchMode=yes -o ProxyCommand='$PROXY'" \
    --include='*/' --include="*_${TAG}_*/**" --exclude='*' \
    "${REMOTE_USER}@${HOST}:${REMOTE_ROOT}/workloads/" \
    "${ARTIFACTS}/gfx942/" || true
  scp -o BatchMode=yes -o ProxyCommand="$PROXY" \
    "${REMOTE_USER}@${HOST}:${REMOTE_LOG}" \
    "${ARTIFACTS}/spp_health_${TAG}.log" || true
}

generate_report() {
  local logs="${ARTIFACTS}/logs/gfx942"
  local reports="${ARTIFACTS}/reports"
  mkdir -p "$reports"
  local spp_args=()
  local base_args=()
  local wl_dir_args=()
  local name mode wl_root
  for name in vcopy nbody mega_kernel; do
    [[ -f "${logs}/${name}_spp.log" ]] || {
      echo "ERROR: missing ${logs}/${name}_spp.log" >&2
      exit 1
    }
    spp_args+=("${name}:${logs}/${name}_spp.log")
    if [[ -f "${logs}/${name}_legacy.log" ]]; then
      base_args+=("${name}:${logs}/${name}_legacy.log")
    fi
    for mode in spp legacy; do
      wl_root="${ARTIFACTS}/gfx942/${name}_${TAG}_${mode}"
      if [[ -d "$wl_root" ]]; then
        local sys
        sys=$(find "$wl_root" -maxdepth 3 -type f -name sysinfo.csv 2>/dev/null | head -1 || true)
        if [[ -n "$sys" ]]; then
          wl_dir_args+=("${name}_${mode}:$(dirname "$sys")")
        fi
      fi
    done
  done
  local iters="${ITERS}"
  if [[ -f "${logs}/run_meta.env" ]]; then
    # shellcheck disable=SC1090
    iters=$(grep -E '^ITERS=' "${logs}/run_meta.env" | cut -d= -f2- || echo "$ITERS")
  fi
  local cmd=(
    python3 "${REPO_ROOT}/tools/generate_metric_health_report.py"
    --arch gfx942
    --out "${reports}/gfx942_health_spp.html"
    --host "${HOST}"
    --baseline-label "legacy heuristic"
    --iterations "${iters}"
    --stat Median
    "${spp_args[@]}"
  )
  if ((${#base_args[@]})); then
    cmd+=(--baseline "${base_args[@]}")
  fi
  if ((${#wl_dir_args[@]})); then
    cmd+=(--workload-dir "${wl_dir_args[@]}")
  fi
  "${cmd[@]}"
  echo "Report: ${reports}/gfx942_health_spp.html"
}

if ! "${DO_WAIT}" && ! "${DO_FETCH}" && ! "${DO_REPORT}"; then
  sync_and_start
  exit 0
fi

"${DO_WAIT}" && wait_for_done
"${DO_FETCH}" && fetch_results
"${DO_REPORT}" && generate_report
