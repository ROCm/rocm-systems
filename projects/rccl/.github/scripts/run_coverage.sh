#!/usr/bin/env bash
# Primary code-coverage entry point. Delegates to the Python implementation
# (run_workload.py). The workflow calls this, so the Python path is the default.
#
# Bash backup (kept runnable): bash .github/scripts/run_workload.sh  (WORKLOAD=coverage)
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
exec python3 "$HERE/run_workload.py" --workload coverage "$@"
