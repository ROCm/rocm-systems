#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Local driver for gfx950 (MI350) SPP health on Alola Slurm.
# Node bg-1w300-h3-2a shares Markham home with alola-login.
#
#   TAG=261002-gfx950-500med ./scripts/run_spp_health_gfx950_local_driver.sh
#   TAG=261002-gfx950-500med ./scripts/run_spp_health_gfx950_local_driver.sh \
#     --fetch --report

set -euo pipefail

HOST_LOGIN=${HOST_LOGIN:-alola-login}
NODE=${NODE:-bg-1w300-h3-2a}
PARTITION=${PARTITION:-defq}
GRES=${GRES:-gpu:gfx950-mi350x:1}
THEROCK=${THEROCK:-/opt/rocm-7.1.0}
TAG=${TAG:-$(date +%y%m%d)-gfx950-500med}
ITERS=${ITERS:-500}
NO_ROOF=${NO_ROOF:-1}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS=${ARTIFACTS:-${REPO_ROOT}/validation-artifacts/spp-health-${TAG}}
REMOTE_ROOT=${REMOTE_ROOT:-/home/AMD/feizheng/aiprofcomp865/rocprofiler-compute}
REMOTE_LOG=${REMOTE_LOG:-/home/AMD/feizheng/spp_health_${TAG}.log}
SBATCH_OUT=${SBATCH_OUT:-/home/AMD/feizheng/spp_health_${TAG}.sbatch.out}
WALLTIME=${WALLTIME:-24:00:00}

DO_WAIT=false
DO_FETCH=false
DO_REPORT=false

usage() {
  cat <<EOF
Usage: $(basename "$0") [--wait] [--fetch] [--report]
Env: TAG ITERS NODE PARTITION GRES THEROCK ARTIFACTS
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

ssh_login() {
  ssh -o BatchMode=yes -o ServerAliveInterval=30 "${HOST_LOGIN}" "$@"
}

sync_and_start() {
  echo "=== rsync -> ${HOST_LOGIN}:${REMOTE_ROOT}/ ==="
  rsync -az -e "ssh -o BatchMode=yes" \
    --exclude 'workloads/' \
    --exclude '.git/' \
    --exclude '.venv*/' \
    --exclude '.venv-*/' \
    --exclude 'graphify-out/' \
    --exclude 'validation-artifacts/' \
    --exclude 'install/' \
    --exclude 'roofline_demo/' \
    --exclude 'docs/_preview/' \
    --exclude 'docs/*.tar.gz' \
    --exclude 'tests/workloads/' \
    --exclude 'src/workloads/' \
    --exclude 'projects/' \
    --exclude '*.html' \
    --exclude '*.tar.gz' \
    --exclude '__pycache__/' \
    "${REPO_ROOT}/" "${HOST_LOGIN}:${REMOTE_ROOT}/"

  echo "=== sbatch health run TAG=${TAG} NODE=${NODE} ==="
  local sbatch_local
  sbatch_local=$(mktemp)
  cat >"${sbatch_local}" <<SBATCH
#!/bin/bash
#SBATCH -J spp-health-gfx950
#SBATCH -p ${PARTITION}
#SBATCH -w ${NODE}
#SBATCH --gres=${GRES}
#SBATCH -t ${WALLTIME}
#SBATCH -o ${SBATCH_OUT}
#SBATCH -e ${SBATCH_OUT}
set -euo pipefail
export TAG='${TAG}'
export ROOT='${REMOTE_ROOT}'
export THEROCK='${THEROCK}'
export ROCM_ROOT='${THEROCK}'
export LOG='${REMOTE_LOG}'
export ITERS='${ITERS}'
export NO_ROOF='${NO_ROOF}'
export FORCE_NO_NATIVE_TOOL=1
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export SITE_ROCM_CORE=/opt/rocm
chmod +x "\${ROOT}/scripts/run_spp_health_gfx950.sh"
bash "\${ROOT}/scripts/run_spp_health_gfx950.sh"
SBATCH
  scp -o BatchMode=yes "${sbatch_local}" "${HOST_LOGIN}:/tmp/spp_health_${TAG}.sbatch"
  rm -f "${sbatch_local}"
  ssh_login "chmod +x /tmp/spp_health_${TAG}.sbatch && \
    JOB=\$(sbatch /tmp/spp_health_${TAG}.sbatch | awk '{print \$NF}') && \
    echo JOBID=\$JOB && echo Remote log: ${REMOTE_LOG} && echo Sbatch out: ${SBATCH_OUT}"
  echo "Re-run: TAG=${TAG} $(basename "$0") --wait --fetch --report"
}

wait_for_done() {
  echo "=== waiting for ALL_DONE in ${REMOTE_LOG} ==="
  while true; do
    if ssh_login "grep -q 'ALL_DONE' '${REMOTE_LOG}' 2>/dev/null"; then
      echo "ALL_DONE"
      ssh_login "grep -E 'WORKLOAD_DONE|ALL_DONE|MEGA_SMOKE|WARN:|DISPATCH_COUNT|iters=' '${REMOTE_LOG}' | tail -40" || true
      return 0
    fi
    ssh_login "squeue -u feizheng -o '%.18i %.9P %.8j %.2t %.10M %N' | head -10; \
      grep -E 'WORKLOAD_DONE|PROFILE |MEGA_SMOKE|WARN:' '${REMOTE_LOG}' 2>/dev/null | tail -5" || true
    sleep 180
  done
}

fetch_results() {
  mkdir -p "${ARTIFACTS}/gfx950" "${ARTIFACTS}/logs/gfx950"
  echo "=== fetch workloads + logs ==="
  rsync -az -e "ssh -o BatchMode=yes" \
    "${HOST_LOGIN}:${REMOTE_ROOT}/workloads/logs_gfx950_${TAG}/" \
    "${ARTIFACTS}/logs/gfx950/" || true
  rsync -az -e "ssh -o BatchMode=yes" \
    --include='*/' --include="*_${TAG}_*/**" --exclude='*' \
    "${HOST_LOGIN}:${REMOTE_ROOT}/workloads/" \
    "${ARTIFACTS}/gfx950/" || true
  scp -o BatchMode=yes \
    "${HOST_LOGIN}:${REMOTE_LOG}" \
    "${ARTIFACTS}/spp_health_${TAG}.log" || true
}

generate_report() {
  local logs="${ARTIFACTS}/logs/gfx950"
  local reports="${ARTIFACTS}/reports"
  mkdir -p "$reports"
  local spp_args=() base_args=() wl_dir_args=() name mode wl_root sys
  for name in vcopy nbody mega_kernel; do
    [[ -f "${logs}/${name}_spp.log" ]] || {
      echo "ERROR: missing ${logs}/${name}_spp.log" >&2
      exit 1
    }
    spp_args+=("${name}:${logs}/${name}_spp.log")
    [[ -f "${logs}/${name}_legacy.log" ]] && base_args+=("${name}:${logs}/${name}_legacy.log")
    for mode in spp legacy; do
      wl_root="${ARTIFACTS}/gfx950/${name}_${TAG}_${mode}"
      if [[ -d "$wl_root" ]]; then
        sys=$(find "$wl_root" -maxdepth 3 -type f -name sysinfo.csv 2>/dev/null | head -1 || true)
        [[ -n "$sys" ]] && wl_dir_args+=("${name}_${mode}:$(dirname "$sys")")
      fi
    done
  done
  local iters="${ITERS}"
  [[ -f "${logs}/run_meta.env" ]] && iters=$(grep -E '^ITERS=' "${logs}/run_meta.env" | cut -d= -f2- || echo "$ITERS")
  local cmd=(
    python3 "${REPO_ROOT}/tools/generate_metric_health_report.py"
    --arch gfx950
    --out "${reports}/gfx950_health_spp.html"
    --host "${NODE}"
    --baseline-label "legacy heuristic"
    --iterations "${iters}"
    --stat Median
    "${spp_args[@]}"
  )
  ((${#base_args[@]})) && cmd+=(--baseline "${base_args[@]}")
  ((${#wl_dir_args[@]})) && cmd+=(--workload-dir "${wl_dir_args[@]}")
  "${cmd[@]}"
  echo "Report: ${reports}/gfx950_health_spp.html"
}

if ! "${DO_WAIT}" && ! "${DO_FETCH}" && ! "${DO_REPORT}"; then
  sync_and_start
  exit 0
fi

"${DO_WAIT}" && wait_for_done
"${DO_FETCH}" && fetch_results
"${DO_REPORT}" && generate_report
