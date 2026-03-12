#!/bin/bash
# Wrapper script to run rocprof-compute with clean environment
# This unsets Python environment variables that may interfere with the embedded interpreter

# Unset Python environment variables
unset PYTHONHOME
unset PYTHONPATH
unset PYTHONSTARTUP
unset PYTHONUSERBASE

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the rocprof-compute binary
exec "${SCRIPT_DIR}/rocprof-compute" "$@"
