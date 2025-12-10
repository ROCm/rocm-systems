#!/bin/bash
# run_docker_tests.sh - Build and run tests in isolated Docker container
#
# This script builds a Docker image and runs the remote tools tests
# in an isolated environment for security.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="rocstorage-remote-tools-test"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check Docker is available
if ! command -v docker &>/dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check Docker daemon is running
if ! docker info &>/dev/null; then
    log_error "Docker daemon is not running"
    exit 1
fi

REMOTE_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REMOTE_DIR"

# Build the test image (context is parent dir so we can copy tools)
log_info "Building Docker test image..."
docker build -t "$IMAGE_NAME" -f tests/Dockerfile.test .

# Run tests in container
log_info "Running tests in isolated container..."
echo ""

docker run --rm \
    --security-opt=no-new-privileges \
    --cap-drop=ALL \
    --tmpfs /tmp:rw,nosuid,size=64m \
    --tmpfs /home/testuser:rw,nosuid,size=64m \
    "$IMAGE_NAME"

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    log_info "All Docker tests completed successfully!"
else
    log_error "Docker tests failed with exit code $exit_code"
fi

exit $exit_code