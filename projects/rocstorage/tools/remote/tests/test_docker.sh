#!/bin/bash
# test_docker.sh - Docker entrypoint for running tests
#
# This script is the entrypoint for the Docker test container.
# It simply invokes test_local.sh which contains all the test logic.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# In Docker, tools are in /app and test script is also in /app
if [ -d "/app" ] && [ -x "/app/rocstorage-server" ]; then
    export TOOLS_DIR="/app"
    exec bash /app/test_local.sh "$@"
else
    # Running outside Docker (shouldn't happen, but handle gracefully)
    exec "$SCRIPT_DIR/test_local.sh" "$@"
fi
