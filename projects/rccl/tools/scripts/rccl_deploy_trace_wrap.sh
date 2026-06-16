#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --trace-dir DIR -- command [args...]" >&2
  echo "Builds and preloads rccl_deploy_trace_preload.c, then runs the command." >&2
}

trace_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --trace-dir)
      trace_dir="${2:-}"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$trace_dir" || $# -eq 0 ]]; then
  usage
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$trace_dir"
mkdir -p "$trace_dir/bootstrap"

preload_so="${RCCL_DEPLOY_TRACE_PRELOAD:-$trace_dir/librccl_deploy_trace_preload.so}"
if [[ ! -f "$preload_so" || "$script_dir/rccl_deploy_trace_preload.c" -nt "$preload_so" ]]; then
  "${CC:-cc}" -shared -fPIC -O2 -Wall -Wextra \
    "$script_dir/rccl_deploy_trace_preload.c" -ldl -o "$preload_so"
fi

export RCCL_DEPLOY_TRACE_FILE="${RCCL_DEPLOY_TRACE_FILE:-$trace_dir/deploy_rank%r_pid%p.log}"
export NCCL_BOOTSTRAP_TRACE="${NCCL_BOOTSTRAP_TRACE:-1}"
export NCCL_BOOTSTRAP_TRACE_DIR="${NCCL_BOOTSTRAP_TRACE_DIR:-$trace_dir/bootstrap}"
if [[ -n "${LD_PRELOAD:-}" ]]; then
  export LD_PRELOAD="$preload_so:$LD_PRELOAD"
else
  export LD_PRELOAD="$preload_so"
fi

echo "RCCL deploy trace: trace_dir=$trace_dir" >&2
echo "RCCL deploy trace: preload=$preload_so" >&2
echo "RCCL deploy trace: command=$*" >&2
exec "$@"
