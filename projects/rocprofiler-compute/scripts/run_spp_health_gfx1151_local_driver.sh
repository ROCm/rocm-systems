#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Local driver for gfx1151 SPP health validation on rocprof-apu3 (Strix Halo).
#
#   TAG=261001d-gfx1151-500med ./scripts/run_spp_health_gfx1151_local_driver.sh
#   TAG=261001d-gfx1151-500med ./scripts/run_spp_health_gfx1151_local_driver.sh \
#     --wait --fetch --report

set -euo pipefail

PROXY=${PROXY:-'socat - SOCKS4A:127.0.0.1:%h:%p,socksport=1080'}
HOST=${HOST:-10.6.205.100}
REMOTE_USER=${REMOTE_USER:-rocprof}
REMOTE_PASS=${REMOTE_PASS:-}
REMOTE_ROOT=${REMOTE_ROOT:-/home/rocprof/aiprofcomp865/rocm-systems/projects/rocprofiler-compute}
THEROCK=${THEROCK:-/home/rocprof/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel}
TAG=${TAG:-$(date +%y%m%d)-gfx1151-500}
ITERS=${ITERS:-500}
NO_ROOF=${NO_ROOF:-1}
MEGA_BATCH=${MEGA_BATCH:-4096}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS=${ARTIFACTS:-${REPO_ROOT}/validation-artifacts/spp-health-${TAG}}
REMOTE_LOG=${REMOTE_LOG:-/home/rocprof/spp_health_${TAG}.log}
REMOTE_NOHUP=${REMOTE_NOHUP:-/home/rocprof/spp_health_${TAG}.nohup}

DO_WAIT=false
DO_FETCH=false
DO_REPORT=false

usage() {
  cat <<EOF
Usage: $(basename "$0") [--wait] [--fetch] [--report]
Env: TAG ITERS=${ITERS} HOST=${HOST} REMOTE_ROOT ARTIFACTS REMOTE_PASS
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

# Password auth helper (rocprof-apu3). Prefer REMOTE_PASS / SSHPASS env.
_ssh_base=(ssh -o BatchMode=no -o StrictHostKeyChecking=no
  -o PreferredAuthentications=password -o PubkeyAuthentication=no
  -o ProxyCommand="${PROXY}")
_rsync_e="ssh -o BatchMode=no -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no -o ProxyCommand='${PROXY}'"

_with_pass() {
  if [[ -n "${REMOTE_PASS}" ]]; then
    if command -v sshpass >/dev/null 2>&1; then
      SSHPASS="${REMOTE_PASS}" sshpass -e "$@"
      return
    fi
  fi
  "$@"
}

ssh_cmd() {
  _with_pass "${_ssh_base[@]}" "${REMOTE_USER}@${HOST}" "$@"
}

rsync_rpc() {
  if [[ -n "${REMOTE_PASS}" ]] && command -v sshpass >/dev/null 2>&1; then
    SSHPASS="${REMOTE_PASS}" rsync -az -e "sshpass -e ${_rsync_e}" "$@"
  else
    rsync -az -e "${_rsync_e}" "$@"
  fi
}

sync_and_start() {
  echo "=== mkdir remote root ==="
  ssh_cmd "mkdir -p '${REMOTE_ROOT}'"

  echo "=== rsync -> ${HOST}:${REMOTE_ROOT}/ ==="
  rsync_rpc \
    --exclude 'workloads/' \
    --exclude '.git/' \
    --exclude '.venv*/' \
    --exclude 'graphify-out/' \
    --exclude '*.html' \
    --exclude 'validation-artifacts/' \
    "${REPO_ROOT}/" "${REMOTE_USER}@${HOST}:${REMOTE_ROOT}/"

  echo "=== start remote health run (TAG=${TAG} ITERS=${ITERS}) ==="
  ssh_cmd "chmod +x '${REMOTE_ROOT}/scripts/run_spp_health_gfx1151.sh' && \
    nohup env TAG='${TAG}' ROOT='${REMOTE_ROOT}' THEROCK='${THEROCK}' \
      LOG='${REMOTE_LOG}' \
      ROCR_VISIBLE_DEVICES='${ROCR_VISIBLE_DEVICES:-0}' HIP_VISIBLE_DEVICES='0' \
      ITERS='${ITERS}' NO_ROOF='${NO_ROOF}' MEGA_BATCH='${MEGA_BATCH}' \
      bash '${REMOTE_ROOT}/scripts/run_spp_health_gfx1151.sh' \
      >'${REMOTE_NOHUP}' 2>&1 & echo PID=\$!"
  echo "Remote log: ${REMOTE_LOG}"
  echo "Re-run: TAG=${TAG} REMOTE_PASS=… $(basename "$0") --wait --fetch --report"
}

wait_for_done() {
  echo "=== waiting for ALL_DONE in ${REMOTE_LOG} ==="
  while true; do
    if ssh_cmd "grep -q 'ALL_DONE' '${REMOTE_LOG}' 2>/dev/null"; then
      echo "ALL_DONE"
      ssh_cmd "grep -E 'WORKLOAD_DONE|ALL_DONE|MEGA_SMOKE|WARN:|DISPATCH_COUNT|iters=' '${REMOTE_LOG}' | tail -40" || true
      return 0
    fi
    ssh_cmd "grep -E 'WORKLOAD_DONE|PROFILE |MEGA_SMOKE|WARN:|DISPATCH_COUNT|START' '${REMOTE_LOG}' 2>/dev/null | tail -5" || true
    sleep 120
  done
}

fetch_results() {
  mkdir -p "${ARTIFACTS}/gfx1151" "${ARTIFACTS}/logs/gfx1151"
  echo "=== fetch workloads + logs ==="
  if [[ -n "${REMOTE_PASS}" ]] && command -v sshpass >/dev/null 2>&1; then
    SSHPASS="${REMOTE_PASS}" rsync -az -e "sshpass -e ${_rsync_e}" \
      "${REMOTE_USER}@${HOST}:${REMOTE_ROOT}/workloads/logs_gfx1151_${TAG}/" \
      "${ARTIFACTS}/logs/gfx1151/" || true
    SSHPASS="${REMOTE_PASS}" rsync -az -e "sshpass -e ${_rsync_e}" \
      --include='*/' --include="*_${TAG}_*/**" --exclude='*' \
      "${REMOTE_USER}@${HOST}:${REMOTE_ROOT}/workloads/" \
      "${ARTIFACTS}/gfx1151/" || true
    SSHPASS="${REMOTE_PASS}" scp -o StrictHostKeyChecking=no \
      -o PreferredAuthentications=password -o PubkeyAuthentication=no \
      -o ProxyCommand="${PROXY}" \
      "${REMOTE_USER}@${HOST}:${REMOTE_LOG}" \
      "${ARTIFACTS}/spp_health_${TAG}.log" || true
  else
    rsync -az -e "${_rsync_e}" \
      "${REMOTE_USER}@${HOST}:${REMOTE_ROOT}/workloads/logs_gfx1151_${TAG}/" \
      "${ARTIFACTS}/logs/gfx1151/" || true
    rsync -az -e "${_rsync_e}" \
      --include='*/' --include="*_${TAG}_*/**" --exclude='*' \
      "${REMOTE_USER}@${HOST}:${REMOTE_ROOT}/workloads/" \
      "${ARTIFACTS}/gfx1151/" || true
    scp -o StrictHostKeyChecking=no \
      -o PreferredAuthentications=password -o PubkeyAuthentication=no \
      -o ProxyCommand="${PROXY}" \
      "${REMOTE_USER}@${HOST}:${REMOTE_LOG}" \
      "${ARTIFACTS}/spp_health_${TAG}.log" || true
  fi
}

generate_report() {
  local logs="${ARTIFACTS}/logs/gfx1151"
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
      wl_root="${ARTIFACTS}/gfx1151/${name}_${TAG}_${mode}"
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
    --arch gfx1151
    --out "${reports}/gfx1151_health_spp.html"
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
  echo "Report: ${reports}/gfx1151_health_spp.html"
}

if ! "${DO_WAIT}" && ! "${DO_FETCH}" && ! "${DO_REPORT}"; then
  sync_and_start
  exit 0
fi

"${DO_WAIT}" && wait_for_done
"${DO_FETCH}" && fetch_results
"${DO_REPORT}" && generate_report
