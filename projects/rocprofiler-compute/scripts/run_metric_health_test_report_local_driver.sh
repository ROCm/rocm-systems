#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Local driver for the Metric Health test report.
# Syncs rocprofiler-compute to the GPU host, starts scripts/run_metric_health_test_report.sh,
# and can wait, fetch, and regenerate the HTML report.
#
#   ARCH=gfx942 ./scripts/run_metric_health_test_report_local_driver.sh
#   ARCH=gfx942 TAG=261001-gfx942-500med \
#     ./scripts/run_metric_health_test_report_local_driver.sh --fetch --report
#   ARCH=gfx1151 ./scripts/run_metric_health_test_report_local_driver.sh --wait --fetch --report
#   ARCH=gfx908 TAG=261002-gfx908-500med \
#     ./scripts/run_metric_health_test_report_local_driver.sh --wait --fetch --report
#   ARCH=gfx90a TAG=261002-gfx90a-500med \
#     ./scripts/run_metric_health_test_report_local_driver.sh --fetch --report
#   ARCH=gfx950 TAG=261002-gfx950-500med \
#     ./scripts/run_metric_health_test_report_local_driver.sh --fetch --report
#
# 500-iter medians are the default ITERS=500, including gfx942. There is no
# separate 500med script.
#
# The report compares two artifact directories from two separate runs. Make
# the baseline with a rocprofiler-compute-develop checkout as PROFILER_ROOT
# (path on the GPU host) and RUN=develop, then fetch it:
#
#   ARCH=gfx942 TAG=261011-gfx942-500-develop RUN=develop \
#     PROFILER_ROOT=/path/on/host/develop/projects/rocprofiler-compute \
#     ./scripts/run_metric_health_test_report_local_driver.sh --wait --fetch
#
# Then report this branch's run against it:
#
#   ARCH=gfx942 TAG=261011-gfx942-500 \
#     BASELINE=validation-artifacts/metric-health-261011-gfx942-500-develop \
#     ./scripts/run_metric_health_test_report_local_driver.sh --report
#
# --report calls tools/generate_metric_health_report.py with
# --candidate "$ARTIFACTS" --baseline "$BASELINE".
#
# Hosts, nodes, proxies, accounts, and remote paths are site values and are
# not stored in this script. Copy scripts/.metric_health_sites.env.example to
# scripts/.metric_health_sites.env (gitignored) and fill it in, or export the
# variables. METRIC_HEALTH_SITES_ENV overrides the config path. Variables
# already set in the environment win over the file.

set -euo pipefail

usage() {
  cat <<EOF
Usage: ARCH=<gfx942|gfx950|gfx90a|gfx908|gfx1151> $(basename "$0") [--wait] [--fetch] [--report]
       $(basename "$0") <arch> [--wait] [--fetch] [--report]

Env: TAG ITERS NO_ROOF THEROCK ARTIFACTS HOST NODE PARTITION GRES
     RUN PROFILER_ROOT BASELINE BASELINE_RUN METRIC_HEALTH_SITES_ENV

Site values (per-arch HOST, PROXY, NODE, roots, ...) come from the environment
or from scripts/.metric_health_sites.env; see scripts/.metric_health_sites.env.example.
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
RUN=${RUN:-single-pass}
BASELINE_RUN=${BASELINE_RUN:-develop}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SITE_CONFIG=${METRIC_HEALTH_SITES_ENV:-${SCRIPT_DIR}/.metric_health_sites.env}
SITE_CONFIG_VARS=(
  GFX1151_HOST GFX1151_PROXY GFX1151_REMOTE_USER GFX1151_REMOTE_ROOT
  GFX1151_THEROCK GFX1151_REMOTE_LOG_DIR
  GFX942_HOST GFX942_PROXY GFX942_REMOTE_USER GFX942_REMOTE_ROOT
  GFX942_THEROCK GFX942_REMOTE_LOG_DIR
  SLURM_HOST_LOGIN SLURM_PARTITION SLURM_REMOTE_HOME SLURM_REMOTE_ROOT
  GFX908_NODE GFX908_GRES GFX950_NODE GFX950_GRES GFX90A_NODE GFX90A_GRES
)

load_site_config() {
  if [[ ! -f "$SITE_CONFIG" ]]; then
    return 0
  fi
  local name
  local -A preset=()
  for name in "${SITE_CONFIG_VARS[@]}"; do
    if [[ -n "${!name+set}" ]]; then
      preset[$name]=${!name}
    fi
  done
  # shellcheck disable=SC1090
  . "$SITE_CONFIG"
  for name in "${!preset[@]}"; do
    printf -v "$name" '%s' "${preset[$name]}"
  done
}

# Stops before any remote step when a site value has no setting.
require_site_value() {
  local name=$1 source_name=$2
  if [[ -z "${!name:-}" ]]; then
    local exports=$name
    if [[ "$source_name" != "$name" ]]; then
      exports="${name} or ${source_name}"
    fi
    echo "ERROR: ${name} is not set for --arch ${ARCH}." >&2
    echo "Export ${exports}, or set ${source_name} in ${SITE_CONFIG}" >&2
    echo "(template: ${SCRIPT_DIR}/.metric_health_sites.env.example)." >&2
    exit 1
  fi
}

load_site_config

# Assumed remote username when a site doesn't have its own REMOTE_USER
# (slurm transports log in via HOST_LOGIN, which is expected to carry its
# own user). Override by exporting USER, or by setting REMOTE_ROOT /
# NODE_ROOT / REMOTE_LOG / SBATCH_OUT directly.
LOCAL_USER=${USER:-$(id -un)}

require_slurm_site() {
  HOST_LOGIN=${HOST_LOGIN:-${SLURM_HOST_LOGIN:-}}
  PARTITION=${PARTITION:-${SLURM_PARTITION:-}}
  require_site_value HOST_LOGIN SLURM_HOST_LOGIN
  require_site_value PARTITION SLURM_PARTITION
  require_site_value SLURM_REMOTE_HOME SLURM_REMOTE_HOME
  SLURM_DEFAULT_ROOT=${SLURM_REMOTE_ROOT:-${SLURM_REMOTE_HOME}/rocprofiler-compute}
}

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
      PROXY=${PROXY:-${GFX1151_PROXY:-none}}
      HOST=${HOST:-${GFX1151_HOST:-}}
      REMOTE_USER=${REMOTE_USER:-${GFX1151_REMOTE_USER:-}}
      REMOTE_PASS=${REMOTE_PASS:-}
      REMOTE_ROOT=${REMOTE_ROOT:-${GFX1151_REMOTE_ROOT:-}}
      THEROCK=${THEROCK:-${GFX1151_THEROCK:-}}
      REMOTE_LOG_DIR=${REMOTE_LOG_DIR:-${GFX1151_REMOTE_LOG_DIR:-}}
      require_site_value HOST GFX1151_HOST
      require_site_value REMOTE_USER GFX1151_REMOTE_USER
      require_site_value REMOTE_ROOT GFX1151_REMOTE_ROOT
      require_site_value THEROCK GFX1151_THEROCK
      TAG_SUFFIX=500
      PASS_MEGA_BATCH=1
      MEGA_BATCH=${MEGA_BATCH:-4096}
      ROCR_DEFAULT=0
      POLL_SLEEP=120
      ;;
    gfx942)
      TRANSPORT=ssh-key
      PROXY=${PROXY:-${GFX942_PROXY:-none}}
      HOST=${HOST:-${GFX942_HOST:-}}
      REMOTE_USER=${REMOTE_USER:-${GFX942_REMOTE_USER:-${LOCAL_USER}}}
      REMOTE_ROOT=${REMOTE_ROOT:-${GFX942_REMOTE_ROOT:-}}
      THEROCK=${THEROCK:-${GFX942_THEROCK:-}}
      REMOTE_LOG_DIR=${REMOTE_LOG_DIR:-${GFX942_REMOTE_LOG_DIR:-}}
      require_site_value HOST GFX942_HOST
      require_site_value REMOTE_ROOT GFX942_REMOTE_ROOT
      require_site_value THEROCK GFX942_THEROCK
      TAG_SUFFIX=500
      PASS_LOCAL_PREFIX=1
      ROCR_DEFAULT=17
      POLL_SLEEP=120
      ;;
    gfx908)
      TRANSPORT=slurm-home
      RSYNC_STYLE=site
      require_slurm_site
      NODE=${NODE:-${GFX908_NODE:-}}
      require_site_value NODE GFX908_NODE
      GRES=${GRES:-${GFX908_GRES:-}}
      require_site_value GRES GFX908_GRES
      THEROCK=${THEROCK:-/opt/rocm-7.2.0}
      TAG_SUFFIX=500med
      REMOTE_ROOT=${REMOTE_ROOT:-${SLURM_DEFAULT_ROOT}}
      WALLTIME=${WALLTIME:-24:00:00}
      POLL_SLEEP=180
      REPORT_MISSING=warn
      REPORT_HOST_KIND=node
      SITE_ROCM_IN_LAUNCH=1
      ;;
    gfx950)
      TRANSPORT=slurm-home
      RSYNC_STYLE=site
      require_slurm_site
      NODE=${NODE:-${GFX950_NODE:-}}
      require_site_value NODE GFX950_NODE
      GRES=${GRES:-${GFX950_GRES:-}}
      require_site_value GRES GFX950_GRES
      THEROCK=${THEROCK:-/opt/rocm-7.1.0}
      TAG_SUFFIX=500med
      REMOTE_ROOT=${REMOTE_ROOT:-${SLURM_DEFAULT_ROOT}}
      WALLTIME=${WALLTIME:-24:00:00}
      POLL_SLEEP=180
      REPORT_HOST_KIND=node
      SITE_ROCM_IN_LAUNCH=1
      ;;
    gfx90a)
      TRANSPORT=slurm-remote
      RSYNC_STYLE=site
      require_slurm_site
      NODE=${NODE:-${GFX90A_NODE:-}}
      require_site_value NODE GFX90A_NODE
      GRES=${GRES:-${GFX90A_GRES:-}}
      require_site_value GRES GFX90A_GRES
      THEROCK=${THEROCK:-/opt/rocm-7.2.0}
      TAG_SUFFIX=500med
      REMOTE_ROOT=${REMOTE_ROOT:-${SLURM_DEFAULT_ROOT}}
      NODE_ROOT=${NODE_ROOT:-${SLURM_DEFAULT_ROOT}}
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
  ARTIFACTS=${ARTIFACTS:-${REPO_ROOT}/validation-artifacts/metric-health-${TAG}}
  if [[ "$TRANSPORT" == "ssh-key" || "$TRANSPORT" == "ssh-pass" ]]; then
    if [[ -z "${REMOTE_LOG:-}" || -z "${REMOTE_NOHUP:-}" ]]; then
      require_site_value REMOTE_LOG_DIR "${ARCH^^}_REMOTE_LOG_DIR"
    fi
    REMOTE_LOG=${REMOTE_LOG:-${REMOTE_LOG_DIR}/metric_health_${TAG}.log}
    REMOTE_NOHUP=${REMOTE_NOHUP:-${REMOTE_LOG_DIR}/metric_health_${TAG}.nohup}
  else
    REMOTE_LOG=${REMOTE_LOG:-${SLURM_REMOTE_HOME}/metric_health_${TAG}.log}
    SBATCH_OUT=${SBATCH_OUT:-${SLURM_REMOTE_HOME}/metric_health_${TAG}.sbatch.out}
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
    "RUN='${RUN}'"
  )
  if [[ -n "${PROFILER_ROOT:-}" ]]; then
    env_parts+=("PROFILER_ROOT='${PROFILER_ROOT}'")
  fi
  if [[ "$PASS_MEGA_BATCH" == "1" ]]; then
    env_parts+=("MEGA_BATCH='${MEGA_BATCH}'")
  fi
  local env_s="${env_parts[*]}"
  if [[ "$TRANSPORT" == "ssh-key" ]]; then
    echo "=== start remote health run (TAG=${TAG} ROCR=${gpu} ITERS=${ITERS}) ==="
  else
    echo "=== start remote health run (TAG=${TAG} ITERS=${ITERS}) ==="
  fi
  ssh_cmd "chmod +x '${REMOTE_ROOT}/scripts/run_metric_health_test_report.sh' && \
    nohup env ${env_s} \
      bash '${REMOTE_ROOT}/scripts/run_metric_health_test_report.sh' \
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
#SBATCH -J metric-health-${ARCH}
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
export RUN='${RUN}'
export FORCE_NO_NATIVE_TOOL=1
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
SBATCH
  if [[ "$SITE_ROCM_IN_LAUNCH" == "1" ]]; then
    echo "export SITE_ROCM_CORE=/opt/rocm" >>"$dest"
  fi
  if [[ -n "${PROFILER_ROOT:-}" ]]; then
    echo "export PROFILER_ROOT='${PROFILER_ROOT}'" >>"$dest"
  fi
  cat >>"$dest" <<'SBATCH'
chmod +x "${ROOT}/scripts/run_metric_health_test_report.sh"
bash "${ROOT}/scripts/run_metric_health_test_report.sh"
SBATCH
}

submit_sbatch() {
  local sbatch_local
  sbatch_local=$(mktemp)
  write_sbatch "$1" "$sbatch_local"
  scp -o BatchMode=yes "${sbatch_local}" "${HOST_LOGIN}:/tmp/metric_health_${TAG}.sbatch"
  rm -f "${sbatch_local}"
  ssh_login "chmod +x /tmp/metric_health_${TAG}.sbatch && \
    JOB=\$(sbatch /tmp/metric_health_${TAG}.sbatch | awk '{print \$NF}') && \
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

sync_slurm_remote() {
  echo "=== rsync -> ${HOST_LOGIN}:${REMOTE_ROOT}/ ==="
  rsync -az -e "ssh -o BatchMode=yes" \
    "${RSYNC_EXCLUDES[@]}" \
    "${REPO_ROOT}/" "${HOST_LOGIN}:${REMOTE_ROOT}/"

  echo "=== sync login home -> node home on ${NODE} ==="
  ssh_login "tar czf - -C '${REMOTE_ROOT}' \
      --exclude='workloads' --exclude='.git' --exclude='.venv*' \
      --exclude='graphify-out' --exclude='validation-artifacts' . \
    | srun -p '${PARTITION}' -w '${NODE}' --gres='${GRES}' -t 00:20:00 -J metric-sync-${ARCH} \
      bash -lc 'mkdir -p \"${NODE_ROOT}\" && tar xzf - -C \"${NODE_ROOT}\" && \
        test -f \"${NODE_ROOT}/scripts/run_metric_health_test_report.sh\" && echo SYNC_OK'"

  echo "=== sbatch health run TAG=${TAG} NODE=${NODE} ==="
  submit_sbatch "${NODE_ROOT}"
}

sync_and_start() {
  case "$TRANSPORT" in
    ssh-pass|ssh-key) sync_ssh ;;
    slurm-home) sync_slurm_home ;;
    slurm-remote) sync_slurm_remote ;;
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
    ssh_login "squeue -u ${LOCAL_USER} -o '%.18i %.9P %.8j %.2t %.10M %N' | head -10; \
      grep -E 'WORKLOAD_DONE|PROFILE |MEGA_SMOKE|WARN:' '${REMOTE_LOG}' 2>/dev/null | tail -5" || true
    sleep "$POLL_SLEEP"
  done
}

wait_slurm_remote() {
  echo "=== waiting for ALL_DONE in ${REMOTE_LOG} (node home via srun) ==="
  while true; do
    if ssh_login "srun -p '${PARTITION}' -w '${NODE}' --gres='${GRES}' -t 00:05:00 -J metric-poll \
        bash -lc 'grep -q ALL_DONE \"${REMOTE_LOG}\" 2>/dev/null'"; then
      echo "ALL_DONE"
      ssh_login "srun -p '${PARTITION}' -w '${NODE}' --gres='${GRES}' -t 00:05:00 \
        bash -lc 'grep -E \"WORKLOAD_DONE|ALL_DONE|MEGA_SMOKE|WARN:|DISPATCH_COUNT|iters=\" \"${REMOTE_LOG}\" | tail -40'" || true
      return 0
    fi
    ssh_login "squeue -u ${LOCAL_USER} -o '%.18i %.9P %.8j %.2t %.10M %N' | head -10" || true
    sleep "$POLL_SLEEP"
  done
}

wait_for_done() {
  case "$TRANSPORT" in
    ssh-pass|ssh-key) wait_ssh ;;
    slurm-home) wait_slurm_home ;;
    slurm-remote) wait_slurm_remote ;;
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
    "${ARTIFACTS}/metric_health_${TAG}.log" || true
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
    "${ARTIFACTS}/metric_health_${TAG}.log" || true
}

fetch_slurm_remote() {
  mkdir -p "${ARTIFACTS}/${ARCH}" "${ARTIFACTS}/logs/${ARCH}"
  echo "=== fetch via node home on ${NODE} ==="
  ssh_login "srun -p '${PARTITION}' -w '${NODE}' --gres='${GRES}' -t 00:30:00 -J metric-fetch-${ARCH} \
    bash -lc 'tar czf - -C \"${NODE_ROOT}/workloads\" \"logs_${ARCH}_${TAG}\" 2>/dev/null'" \
    | tar xzf - -C "${ARTIFACTS}/logs/" 2>/dev/null \
    && mv "${ARTIFACTS}/logs/logs_${ARCH}_${TAG}"/* "${ARTIFACTS}/logs/${ARCH}/" 2>/dev/null || true
  rmdir "${ARTIFACTS}/logs/logs_${ARCH}_${TAG}" 2>/dev/null || true
  ssh_login "srun -p '${PARTITION}' -w '${NODE}' --gres='${GRES}' -t 00:45:00 -J metric-fetch-wl \
    bash -lc 'cd \"${NODE_ROOT}/workloads\" && tar czf - --wildcards \"*_${TAG}_*\" 2>/dev/null'" \
    | tar xzf - -C "${ARTIFACTS}/${ARCH}/" 2>/dev/null || true
  ssh_login "srun -p '${PARTITION}' -w '${NODE}' --gres='${GRES}' -t 00:10:00 \
    bash -lc 'cat \"${REMOTE_LOG}\"'" > "${ARTIFACTS}/metric_health_${TAG}.log" 2>/dev/null || true
}

fetch_results() {
  case "$TRANSPORT" in
    ssh-pass|ssh-key) fetch_ssh ;;
    slurm-home) fetch_slurm_home ;;
    slurm-remote) fetch_slurm_remote ;;
    *)
      echo "Internal error: TRANSPORT=${TRANSPORT}" >&2
      exit 1
      ;;
  esac
}

generate_report() {
  local logs="${ARTIFACTS}/logs/${ARCH}"
  local reports="${ARTIFACTS}/reports"
  if [[ -z "${BASELINE:-}" ]]; then
    echo "ERROR: --report needs BASELINE=<artifact dir of a rocprofiler-compute-develop run>" >&2
    exit 1
  fi
  if [[ ! -d "$BASELINE" ]]; then
    echo "ERROR: baseline artifact directory not found: ${BASELINE}" >&2
    exit 1
  fi
  mkdir -p "$reports"
  local name
  for name in vcopy nbody mega_kernel; do
    if [[ ! -f "${logs}/${name}_${RUN}.log" ]]; then
      if [[ "$REPORT_MISSING" == "warn" ]]; then
        echo "WARN: missing ${logs}/${name}_${RUN}.log — skipping that workload in report" >&2
        continue
      fi
      echo "ERROR: missing ${logs}/${name}_${RUN}.log" >&2
      exit 1
    fi
  done
  local iters="${ITERS}"
  if [[ -f "${logs}/run_meta.env" ]]; then
    iters=$(grep -E '^ITERS=' "${logs}/run_meta.env" | cut -d= -f2- || echo "$ITERS")
  fi
  python3 "${REPO_ROOT}/tools/generate_metric_health_report.py" \
    --arch "$ARCH" \
    --candidate "$ARTIFACTS" \
    --candidate-run "$RUN" \
    --baseline "$BASELINE" \
    --baseline-run "$BASELINE_RUN" \
    --out "${reports}/${ARCH}_health_vs_develop.html" \
    --host "${REPORT_HOST}" \
    --iterations "${iters}" \
    --stat Median
  echo "Report: ${reports}/${ARCH}_health_vs_develop.html"
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
