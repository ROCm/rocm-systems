#!/bin/bash
# Wrapper script to run hip_trace with increased file descriptor limit

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run hip_trace with all arguments passed to this script
# Use sudo with proper argument handling
exec sudo -E bash -c "ulimit -n 65536 && ulimit -l unlimited && exec '$SCRIPT_DIR/build/hip_trace' \"\$@\"" -- "$@"

