#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Local driver for the SPP metric health test report.
# Syncs rocprofiler-compute to the GPU host, starts scripts/run_spp_health.sh,
# and can wait, fetch, and regenerate the HTML report.
#
#   ARCH=gfx942 ./scripts/run_spp_health_local_driver.sh
#   ARCH=gfx942 TAG=261001-gfx942-500med \
#     ./scripts/run_spp_health_local_driver.sh --fetch --report
#   ARCH=gfx1151 ./scripts/run_spp_health_local_driver.sh --wait --fetch --report
#   ARCH=gfx908 TAG=261002-gfx908-500med \
#     ./scripts/run_spp_health_local_driver.sh --wait --fetch --report
#   ARCH=gfx90a TAG=261002-gfx90a-500med \
#     ./scripts/run_spp_health_local_driver.sh --fetch --report
#   ARCH=gfx950 TAG=261002-gfx950-500med \
#     ./scripts/run_spp_health_local_driver.sh --fetch --report
#
# 500-iter medians are the default ITERS=500, including gfx942. There is no
# separate 500med script. The health run uses the default single-pass packable
# path only. The legacy allocator mode is gone.
#
# --report calls tools/generate_metric_health_report.py.

set -euo pipefail

usage() {
  cat <<EOF
Usage: ARCH=<gfx942|gfx950|gfx90a|gfx908|gfx1151> $(basename "$0") [--wait] [--fetch] [--report]
       $(basename "$0") <arch> [--wait] [--fetch] [--report]

Env: TAG ITERS NO_ROOF THEROCK ARTIFACTS HOST NODE PARTITION GRES
EOF
}

DO_WAIT=false
DO_FETCH=false
DO_REPORT=false
ARCH_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait) DO_WAIT=true ;;
    --fetch) DO_FETCH=true ;;
    --report) DO_REPORT=true ;;
    -h|--help) usage; exit 0 ;;
    gfx*)
      if [[ -n "$ARCH_ARG" ]]; then
        echo "Unknown: $1" >&2
        usage >&2
        exit 1
      fi
      ARCH_ARG=$1
      ;;
    *)
      echo "Unknown: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

ARCH=${ARCH_ARG:-${ARCH:-}}
if [[ -z "$ARCH" ]]; then
  usage >&2
  exit 1
fi

ITERS=${ITERS:-500}
NO_ROOF=${NO_ROOF:-1}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

apply_driver_profile() {
  TRANSPORT=""
  RSYNC_STYLE=short
  PASS_MEGA_BATCH=0
  PASS_LOCAL_PREFIX=0
  ROCR_DEFAULT=0
  POLL_SLEEP=120
  REPORT_MISSING=error
  REPORT_HOST_KIND=host
  SITE_ROCM_IN_LAUNCH=0
  TAG_SUFFIX=500
  case "$ARCH" in
    gfx1151)
      TRANSPORT=ssh-pass
      PROXY=${PROXY:-'socat - SOCKS4A:127.0.0.1:%h:%p,socksport=1080'}
      HOST=${HOST:-10.6.205.100}
      REMOTE_USER=${REMOTE_USER:-rocprof}
      REMOTE_PASS=${REMOTE_PASS:-}
      REMOTE_ROOT=${REMOTE_ROOT:-/home/rocprof/aiprofcomp865/rocm-systems/projects/rocprofiler-compute}
      THEROCK=${THEROCK:-/home/rocprof/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel}
      TAG_SUFFIX=500
      PASS_MEGA_BATCH=1
      MEGA_BATCH=${MEGA_BATCH:-4096}
      ROCR_DEFAULT=0
      POLL_SLEEP=120
      ;;
    gfx942)
      TRANSPORT=ssh-key
      PROXY=${PROXY:-'nc -X 5 -x 127.0.0.1:1080 %h %p'}
      HOST=${HOST:-hpe-darkstar-ccs-aus-e12-03.cs-aus.dcgpu}
      REMOTE_USER=${REMOTE_USER:-feizheng}
      REMOTE_ROOT=${REMOTE_ROOT:-/home/AMD/feizheng/aiprofcomp78/rocm-systems/projects/rocprofiler-compute}
      THEROCK=${THEROCK:-/home/AMD/feizheng/aiprofcomp78/therock-work/therock-rocm-7.15.0a20260728}
      TAG_SUFFIX=500
      PASS_LOCAL_PREFIX=1
      ROCR_DEFAULT=17
      POLL_SLEEP=120
      ;;
    gfx908)
      TRANSPORT=slurm-home
      RSYNC_STYLE=site
      HOST_LOGIN=${HOST_LOGIN:-alola-login}
      NODE=${NODE:-ctr-mi100-rack33a-5}
      PARTITION=${PARTITION:-defq}
      GRES=${GRES:-gpu:gfx908-mi100:1}
      THEROCK=${THEROCK:-/opt/rocm-7.2.0}
      TAG_SUFFIX=500med
      REMOTE_ROOT=${REMOTE_ROOT:-/home/AMD/feizheng/aiprofcomp865/rocprofiler-compute}
      WALLTIME=${WALLTIME:-24:00:00}
      POLL_SLEEP=180
      REPORT_MISSING=warn
      REPORT_HOST_KIND=node
      SITE_ROCM_IN_LAUNCH=1
      ;;
    gfx950)
      TRANSPORT=slurm-home
      RSYNC_STYLE=site
      HOST_LOGIN=${HOST_LOGIN:-alola-login}
      NODE=${NODE:-bg-1w300-h3-2a}
      PARTITION=${PARTITION:-defq}
      GRES=${GRES:-gpu:gfx950-mi350x:1}
      THEROCK=${THEROCK:-/opt/rocm-7.1.0}
      TAG_SUFFIX=500med
      REMOTE_ROOT=${REMOTE_ROOT:-/home/AMD/feizheng/aiprofcomp865/rocprofiler-compute}
      WALLTIME=${WALLTIME:-24:00:00}
      POLL_SLEEP=180
      REPORT_HOST_KIND=node
      SITE_ROCM_IN_LAUNCH=1
      ;;
    gfx90a)
      TRANSPORT=slurm-austin
      RSYNC_STYLE=site
      HOST_LOGIN=${HOST_LOGIN:-alola-login}
      NODE=${NODE:-smci250-ccs-aus-c05-23}
      PARTITION=${PARTITION:-defq}
      GRES=${GRES:-gpu:gfx90a-mi250:1}
      THEROCK=${THEROCK:-/opt/rocm-7.2.0}
      TAG_SUFFIX=500med
      MARKHAM_ROOT=${MARKHAM_ROOT:-/home/AMD/feizheng/aiprofcomp865/rocprofiler-compute}
      AUSTIN_ROOT=${AUSTIN_ROOT:-/home/AMD/feizheng/aiprofcomp865/rocprofiler-compute}
      WALLTIME=${WALLTIME:-24:00:00}
      POLL_SLEEP=180
      REPORT_HOST_KIND=node
      SITE_ROCM_IN_LAUNCH=0
      ;;
    *)
      echo "Unknown ARCH: ${ARCH}" >&2
      usage >&2
      exit 1
      ;;
  esac

  TAG=${TAG:-$(date +%y%m%d)-${ARCH}-${TAG_SUFFIX}}
  ARTIFACTS=${ARTIFACTS:-${REPO_ROOT}/validation-artifacts/spp-health-${TAG}}
  if [[ "$TRANSPORT" == "ssh-key" ]]; then
    REMOTE_LOG=${REMOTE_LOG:-/home/AMD/feizheng/aiprofcomp78/spp_health_${TAG}.log}
    REMOTE_NOHUP=${REMOTE_NOHUP:-/home/AMD/feizheng/aiprofcomp78/spp_health_${TAG}.nohup}
  elif [[ "$TRANSPORT" == "ssh-pass" ]]; then
    REMOTE_LOG=${REMOTE_LOG:-/home/rocprof/spp_health_${TAG}.log}
    REMOTE_NOHUP=${REMOTE_NOHUP:-/home/rocprof/spp_health_${TAG}.nohup}
  else
    REMOTE_LOG=${REMOTE_LOG:-/home/AMD/feizheng/spp_health_${TAG}.log}
    SBATCH_OUT=${SBATCH_OUT:-/home/AMD/feizheng/spp_health_${TAG}.sbatch.out}
  fi
  if [[ "$REPORT_HOST_KIND" == "node" ]]; then
    REPORT_HOST=$NODE
  else
    REPORT_HOST=$HOST
  fi
}

apply_driver_profile

if [[ "$RSYNC_STYLE" == "site" ]]; then
  RSYNC_EXCLUDES=(
    --exclude 'workloads/'
    --exclude '.git/'
    --exclude '.venv*/'
    --exclude '.venv-*/'
    --exclude 'graphify-out/'
    --exclude 'validation-artifacts/'
    --exclude 'install/'
    --exclude 'roofline_demo/'
    --exclude 'docs/_preview/'
    --exclude 'docs/*.tar.gz'
    --exclude 'tests/workloads/'
    --exclude 'src/workloads/'
    --exclude 'projects/'
    --exclude '*.html'
    --exclude '*.tar.gz'
    --exclude '__pycache__/'
  )
else
  RSYNC_EXCLUDES=(
    --exclude 'workloads/'
    --exclude '.git/'
    --exclude '.venv*/'
    --exclude 'graphify-out/'
    --exclude '*.html'
    --exclude 'validation-artifacts/'
  )
fi

_with_pass() {
  if [[ -n "${REMOTE_PASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
    SSHPASS="${REMOTE_PASS}" sshpass -e "$@"
    return
  fi
  "$@"
}

ssh_cmd() {
  if [[ "$TRANSPORT" == "ssh-pass" ]]; then
    _with_pass ssh -o BatchMode=no -o StrictHostKeyChecking=no \
      -o PreferredAuthentications=password -o PubkeyAuthentication=no \
      -o ProxyCommand="${PROXY}" \
      "${REMOTE_USER}@${HOST}" "$@"
    return
  fi
  ssh -o BatchMode=yes -o ProxyCommand="$PROXY" "${REMOTE_USER}@${HOST}" "$@"
}

ssh_login() {
  ssh -o BatchMode=yes -o ServerAliveInterval=30 "${HOST_LOGIN}" "$@"
}

remote_rsync() {
  if [[ "$TRANSPORT" == "ssh-pass" ]]; then
    local rsync_e="ssh -o BatchMode=no -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no -o ProxyCommand='${PROXY}'"
    if [[ -n "${REMOTE_PASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
      SSHPASS="${REMOTE_PASS}" rsync -az -e "sshpass -e ${rsync_e}" "$@"
    else
      rsync -az -e "${rsync_e}" "$@"
    fi
    return
  fi
  rsync -az -e "ssh -o BatchMode=yes -o ProxyCommand='${PROXY}'" "$@"
}

remote_scp() {
  if [[ "$TRANSPORT" == "ssh-pass" ]]; then
    if [[ -n "${REMOTE_PASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
      SSHPASS="${REMOTE_PASS}" scp -o StrictHostKeyChecking=no \
        -o PreferredAuthentications=password -o PubkeyAuthentication=no \
        -o ProxyCommand="${PROXY}" "$@"
    else
      scp -o StrictHostKeyChecking=no \
        -o PreferredAuthentications=password -o PubkeyAuthentication=no \
        -o ProxyCommand="${PROXY}" "$@"
    fi
    return
  fi
  scp -o BatchMode=yes -o ProxyCommand="$PROXY" "$@"
}

launch_ssh() {
  local gpu="${ROCR_VISIBLE_DEVICES:-$ROCR_DEFAULT}"
  local -a env_parts=(
    "TAG='${TAG}'"
    "ROOT='${REMOTE_ROOT}'"
    "THEROCK='${THEROCK}'"
    "ARCH='${ARCH}'"
  )
  if [[ "$PASS_LOCAL_PREFIX" == "1" ]]; then
    env_parts+=("LOCAL_PREFIX='${LOCAL_PREFIX:-/tmp/rpc_local}'")
  fi
  env_parts+=(
    "LOG='${REMOTE_LOG}'"
    "ROCR_VISIBLE_DEVICES='${gpu}'"
    "HIP_VISIBLE_DEVICES='0'"
    "ITERS='${ITERS}'"
    "NO_ROOF='${NO_ROOF}'"
  )
  if [[ "$PASS_MEGA_BATCH" == "1" ]]; then
    env_parts+=("MEGA_BATCH='${MEGA_BATCH}'")
  fi
  local env_s="${env_parts[*]}"
  if [[ "$TRANSPORT" == "ssh-key" ]]; then
    echo "=== start remote health run (TAG=${TAG} ROCR=${gpu} ITERS=${ITERS}) ==="
  else
    echo "=== start remote health run (TAG=${TAG} ITERS=${ITERS}) ==="
  fi
  ssh_cmd "chmod +x '${REMOTE_ROOT}/scripts/run_spp_health.sh' && \
    nohup env ${env_s} \
      bash '${REMOTE_ROOT}/scripts/run_spp_health.sh' \
      >'${REMOTE_NOHUP}' 2>&1 & echo PID=\$!"
  echo "Remote log: ${REMOTE_LOG}"
  if [[ "$TRANSPORT" == "ssh-pass" ]]; then
    echo "Re-run: ARCH=${ARCH} TAG=${TAG} REMOTE_PASS=… $(basename "$0") --wait --fetch --report"
  else
    echo "Re-run: ARCH=${ARCH} TAG=${TAG} $(basename "$0") --wait --fetch --report"
  fi
}

sync_ssh() {
  if [[ "$TRANSPORT" == "ssh-pass" ]]; then
    echo "=== mkdir remote root ==="
    ssh_cmd "mkdir -p '${REMOTE_ROOT}'"
  fi
  echo "=== rsync -> ${HOST}:${REMOTE_ROOT}/ ==="
  remote_rsync "${RSYNC_EXCLUDES[@]}" \
    "${REPO_ROOT}/" "${REMOTE_USER}@${HOST}:${REMOTE_ROOT}/"
  launch_ssh
}

write_sbatch() {
  local root_path=$1 dest=$2
  cat >"$dest" <<SBATCH
#!/bin/bash
#SBATCH -J spp-health-${ARCH}
#SBATCH -p ${PARTITION}
#SBATCH -w ${NODE}
#SBATCH --gres=${GRES}
#SBATCH -t ${WALLTIME}
#SBATCH -o ${SBATCH_OUT}
#SBATCH -e ${SBATCH_OUT}
set -euo pipefail
export ARCH='${ARCH}'
export TAG='${TAG}'
export ROOT='${root_path}'
export THEROCK='${THEROCK}'
export ROCM_ROOT='${THEROCK}'
export LOG='${REMOTE_LOG}'
export ITERS='${ITERS}'
export NO_ROOF='${NO_ROOF}'
export FORCE_NO_NATIVE_TOOL=1
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
SBATCH
  if [[ "$SITE_ROCM_IN_LAUNCH" == "1" ]]; then
    echo "export SITE_ROCM_CORE=/opt/rocm" >>"$dest"
  fi
  cat >>"$dest" <<'SBATCH'
chmod +x "${ROOT}/scripts/run_spp_health.sh"
bash "${ROOT}/scripts/run_spp_health.sh"
SBATCH
}

submit_sbatch() {
  local sbatch_local
  sbatch_local=$(mktemp)
  write_sbatch "$1" "$sbatch_local"
  scp -o BatchMode=yes "${sbatch_local}" "${HOST_LOGIN}:/tmp/spp_health_${TAG}.sbatch"
  rm -f "${sbatch_local}"
  ssh_login "chmod +x /tmp/spp_health_${TAG}.sbatch && \
    JOB=\$(sbatch /tmp/spp_health_${TAG}.sbatch | awk '{print \$NF}') && \
    echo JOBID=\$JOB && echo Remote log: ${REMOTE_LOG} && echo Sbatch out: ${SBATCH_OUT}"
  echo "Re-run: ARCH=${ARCH} TAG=${TAG} $(basename "$0") --wait --fetch --report"
}

sync_slurm_home() {
  echo "=== rsync -> ${HOST_LOGIN}:${REMOTE_ROOT}/ ==="
  rsync -az -e "ssh -o BatchMode=yes" \
    "${RSYNC_EXCLUDES[@]}" \
    "${REPO_ROOT}/" "${HOST_LOGIN}:${REMOTE_ROOT}/"
  echo "=== sbatch health run TAG=${TAG} NODE=${NODE} ==="
  submit_sbatch "${REMOTE_ROOT}"
}

sync_slurm_austin() {
  echo "=== rsync -> ${HOST_LOGIN}:${MARKHAM_ROOT}/ ==="
  rsync -az -e "ssh -o BatchMode=yes" \
    "${RSYNC_EXCLUDES[@]}" \
    "${REPO_ROOT}/" "${HOST_LOGIN}:${MARKHAM_ROOT}/"

  echo "=== sync Markham -> Austin home on ${NODE} ==="
  ssh_login "tar czf - -C '${MARKHAM_ROOT}' \
      --exclude='workloads' --exclude='.git' --exclude='.venv*' \
      --exclude='graphify-out' --exclude='validation-artifacts' . \
    | srun -p '${PARTITION}' -w '${NODE}' --gres='${GRES}' -t 00:20:00 -J spp-sync-${ARCH} \
      bash -lc 'mkdir -p \"${AUSTIN_ROOT}\" && tar xzf - -C \"${AUSTIN_ROOT}\" && \
        test -f \"${AUSTIN_ROOT}/scripts/run_spp_health.sh\" && echo SYNC_OK'"

  echo "=== sbatch health run TAG=${TAG} NODE=${NODE} ==="
  submit_sbatch "${AUSTIN_ROOT}"
}

sync_and_start() {
  case "$TRANSPORT" in
    ssh-pass|ssh-key) sync_ssh ;;
    slurm-home) sync_slurm_home ;;
    slurm-austin) sync_slurm_austin ;;
    *)
      echo "Internal error: TRANSPORT=${TRANSPORT}" >&2
      exit 1
      ;;
  esac
}

wait_ssh() {
  local poll_re='WORKLOAD_DONE|PROFILE |MEGA_SMOKE|WARN:|DISPATCH_COUNT'
  if [[ "$TRANSPORT" == "ssh-pass" ]]; then
    poll_re="${poll_re}|START"
  fi
  echo "=== waiting for ALL_DONE in ${REMOTE_LOG} ==="
  while true; do
    if ssh_cmd "grep -q 'ALL_DONE' '${REMOTE_LOG}' 2>/dev/null"; then
      echo "ALL_DONE"
      ssh_cmd "grep -E 'WORKLOAD_DONE|ALL_DONE|MEGA_SMOKE|WARN:|DISPATCH_COUNT|iters=' '${REMOTE_LOG}' | tail -40" || true
      return 0
    fi
    ssh_cmd "grep -E '${poll_re}' '${REMOTE_LOG}' 2>/dev/null | tail -5" || true
    sleep "$POLL_SLEEP"
  done
}

wait_slurm_home() {
  echo "=== waiting for ALL_DONE in ${REMOTE_LOG} ==="
  while true; do
    if ssh_login "grep -q 'ALL_DONE' '${REMOTE_LOG}' 2>/dev/null"; then
      echo "ALL_DONE"
      ssh_login "grep -E 'WORKLOAD_DONE|ALL_DONE|MEGA_SMOKE|WARN:|DISPATCH_COUNT|iters=' '${REMOTE_LOG}' | tail -40" || true
      return 0
    fi
    ssh_login "squeue -u feizheng -o '%.18i %.9P %.8j %.2t %.10M %N' | head -10; \
      grep -E 'WORKLOAD_DONE|PROFILE |MEGA_SMOKE|WARN:' '${REMOTE_LOG}' 2>/dev/null | tail -5" || true
    sleep "$POLL_SLEEP"
  done
}

wait_slurm_austin() {
  echo "=== waiting for ALL_DONE in ${REMOTE_LOG} (Austin home via srun) ==="
  while true; do
    if ssh_login "srun -p '${PARTITION}' -w '${NODE}' --gres='${GRES}' -t 00:05:00 -J spp-poll \
        bash -lc 'grep -q ALL_DONE \"${REMOTE_LOG}\" 2>/dev/null'"; then
      echo "ALL_DONE"
      ssh_login "srun -p '${PARTITION}' -w '${NODE}' --gres='${GRES}' -t 00:05:00 \
        bash -lc 'grep -E \"WORKLOAD_DONE|ALL_DONE|MEGA_SMOKE|WARN:|DISPATCH_COUNT|iters=\" \"${REMOTE_LOG}\" | tail -40'" || true
      return 0
    fi
    ssh_login "squeue -u feizheng -o '%.18i %.9P %.8j %.2t %.10M %N' | head -10" || true
    sleep "$POLL_SLEEP"
  done
}

wait_for_done() {
  case "$TRANSPORT" in
    ssh-pass|ssh-key) wait_ssh ;;
    slurm-home) wait_slurm_home ;;
    slurm-austin) wait_slurm_austin ;;
    *)
      echo "Internal error: TRANSPORT=${TRANSPORT}" >&2
      exit 1
      ;;
  esac
}

fetch_ssh() {
  mkdir -p "${ARTIFACTS}/${ARCH}" "${ARTIFACTS}/logs/${ARCH}"
  echo "=== fetch workloads + logs ==="
  remote_rsync \
    "${REMOTE_USER}@${HOST}:${REMOTE_ROOT}/workloads/logs_${ARCH}_${TAG}/" \
    "${ARTIFACTS}/logs/${ARCH}/" || true
  remote_rsync \
    --include='*/' --include="*_${TAG}_*/**" --exclude='*' \
    "${REMOTE_USER}@${HOST}:${REMOTE_ROOT}/workloads/" \
    "${ARTIFACTS}/${ARCH}/" || true
  remote_scp \
    "${REMOTE_USER}@${HOST}:${REMOTE_LOG}" \
    "${ARTIFACTS}/spp_health_${TAG}.log" || true
}

fetch_slurm_home() {
  mkdir -p "${ARTIFACTS}/${ARCH}" "${ARTIFACTS}/logs/${ARCH}"
  echo "=== fetch workloads + logs ==="
  rsync -az -e "ssh -o BatchMode=yes" \
    "${HOST_LOGIN}:${REMOTE_ROOT}/workloads/logs_${ARCH}_${TAG}/" \
    "${ARTIFACTS}/logs/${ARCH}/" || true
  rsync -az -e "ssh -o BatchMode=yes" \
    --include='*/' --include="*_${TAG}_*/**" --exclude='*' \
    "${HOST_LOGIN}:${REMOTE_ROOT}/workloads/" \
    "${ARTIFACTS}/${ARCH}/" || true
  scp -o BatchMode=yes \
    "${HOST_LOGIN}:${REMOTE_LOG}" \
    "${ARTIFACTS}/spp_health_${TAG}.log" || true
}

fetch_slurm_austin() {
  mkdir -p "${ARTIFACTS}/${ARCH}" "${ARTIFACTS}/logs/${ARCH}"
  echo "=== fetch via Austin home on ${NODE} ==="
  ssh_login "srun -p '${PARTITION}' -w '${NODE}' --gres='${GRES}' -t 00:30:00 -J spp-fetch-${ARCH} \
    bash -lc 'tar czf - -C \"${AUSTIN_ROOT}/workloads\" \"logs_${ARCH}_${TAG}\" 2>/dev/null'" \
    | tar xzf - -C "${ARTIFACTS}/logs/" 2>/dev/null \
    && mv "${ARTIFACTS}/logs/logs_${ARCH}_${TAG}"/* "${ARTIFACTS}/logs/${ARCH}/" 2>/dev/null || true
  rmdir "${ARTIFACTS}/logs/logs_${ARCH}_${TAG}" 2>/dev/null || true
  ssh_login "srun -p '${PARTITION}' -w '${NODE}' --gres='${GRES}' -t 00:45:00 -J spp-fetch-wl \
    bash -lc 'cd \"${AUSTIN_ROOT}/workloads\" && tar czf - --wildcards \"*_${TAG}_*\" 2>/dev/null'" \
    | tar xzf - -C "${ARTIFACTS}/${ARCH}/" 2>/dev/null || true
  ssh_login "srun -p '${PARTITION}' -w '${NODE}' --gres='${GRES}' -t 00:10:00 \
    bash -lc 'cat \"${REMOTE_LOG}\"'" > "${ARTIFACTS}/spp_health_${TAG}.log" 2>/dev/null || true
}

fetch_results() {
  case "$TRANSPORT" in
    ssh-pass|ssh-key) fetch_ssh ;;
    slurm-home) fetch_slurm_home ;;
    slurm-austin) fetch_slurm_austin ;;
    *)
      echo "Internal error: TRANSPORT=${TRANSPORT}" >&2
      exit 1
      ;;
  esac
}

generate_report() {
  local logs="${ARTIFACTS}/logs/${ARCH}"
  local reports="${ARTIFACTS}/reports"
  mkdir -p "$reports"
  local spp_args=() wl_dir_args=() name mode wl_root sys
  for name in vcopy nbody mega_kernel; do
    if [[ ! -f "${logs}/${name}_spp.log" ]]; then
      if [[ "$REPORT_MISSING" == "warn" ]]; then
        echo "WARN: missing ${logs}/${name}_spp.log — skipping that workload in report" >&2
        continue
      fi
      echo "ERROR: missing ${logs}/${name}_spp.log" >&2
      exit 1
    fi
    spp_args+=("${name}:${logs}/${name}_spp.log")
    for mode in spp; do
      wl_root="${ARTIFACTS}/${ARCH}/${name}_${TAG}_${mode}"
      if [[ -d "$wl_root" ]]; then
        sys=$(find "$wl_root" -maxdepth 3 -type f -name sysinfo.csv 2>/dev/null | head -1 || true)
        if [[ -n "$sys" ]]; then
          wl_dir_args+=("${name}_${mode}:$(dirname "$sys")")
        fi
      fi
    done
  done
  if [[ "$REPORT_MISSING" == "warn" && ${#spp_args[@]} -eq 0 ]]; then
    echo "ERROR: no SPP analyze logs under ${logs}" >&2
    exit 1
  fi
  local iters="${ITERS}"
  if [[ -f "${logs}/run_meta.env" ]]; then
    iters=$(grep -E '^ITERS=' "${logs}/run_meta.env" | cut -d= -f2- || echo "$ITERS")
  fi
  local cmd=(
    python3 "${REPO_ROOT}/tools/generate_metric_health_report.py"
    --arch "$ARCH"
    --out "${reports}/${ARCH}_health_spp.html"
    --host "${REPORT_HOST}"
    --iterations "${iters}"
    --stat Median
    "${spp_args[@]}"
  )
  if ((${#wl_dir_args[@]})); then
    cmd+=(--workload-dir "${wl_dir_args[@]}")
  fi
  "${cmd[@]}"
  echo "Report: ${reports}/${ARCH}_health_spp.html"
}

if ! "${DO_WAIT}" && ! "${DO_FETCH}" && ! "${DO_REPORT}"; then
  sync_and_start
  exit 0
fi

if "${DO_WAIT}"; then
  wait_for_done
fi
if "${DO_FETCH}"; then
  fetch_results
fi
if "${DO_REPORT}"; then
  generate_report
fi
