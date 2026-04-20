#!/bin/bash

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Run a Docker container with rocjitsu providing the simulated GPU.
#
# Two modes of operation:
#
# 1. Pre-installed: If rocjitsu was installed into the container image
#    (via rocjitsu-docker-build.sh), just set LD_PRELOAD and run.
#
# 2. Bind-mount (default): Mount the host-built library, config, and
#    schema into the container. Used during development.
#
# Usage:
#   ./scripts/rocjitsu-docker-run.sh [docker-flags] <image> [command...]
#
# Examples:
#   # Run rocminfo (bind-mount mode):
#   ./scripts/rocjitsu-docker-run.sh vllm/vllm-openai-rocm rocminfo
#
#   # Interactive shell:
#   ./scripts/rocjitsu-docker-run.sh -it vllm/vllm-openai-rocm bash
#
#   # Pre-installed mode (rocjitsu already in the image):
#   RJ_INSTALLED=1 ./scripts/rocjitsu-docker-run.sh my-image rocminfo
#
# Environment variables:
#   RJ_BUILD_DIR   - Host build directory (default: ./build)
#   RJ_CONFIG      - Config file (default: configs/amdgpu_cdna4_kmd.json)
#   RJ_INSTALL_DIR - Install prefix inside container (default: /opt/rocm)
#   RJ_INSTALLED   - Set to 1 if rocjitsu is pre-installed in the image
#   RJ_LOG_LEVEL   - Log verbosity: 0=silent, 1=errors, 2=info (default: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

INSTALL_DIR="${RJ_INSTALL_DIR:-/opt/rocm}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [docker-flags] <image> [command...]" >&2
    echo "Example: $0 vllm/vllm-openai-rocm rocminfo" >&2
    exit 1
fi

# Parse docker flags (like -it, --entrypoint bash) vs image name.
DOCKER_FLAGS=()
while [[ $# -gt 0 && "$1" == -* ]]; do
    DOCKER_FLAGS+=("$1")
    # Flags that take a value argument.
    case "$1" in
        --entrypoint|--workdir|-w|-u|--user|--name|--hostname|-h|--network|--pid|--ipc)
            shift
            DOCKER_FLAGS+=("$1")
            ;;
    esac
    shift
done

IMAGE="$1"
shift

# Common environment for both modes.
ENV_ARGS=(
    -e "LD_PRELOAD=${INSTALL_DIR}/lib/librocjitsu_kmd.so"
    -e "RJ_CONFIG=${INSTALL_DIR}/share/rocjitsu/configs/amdgpu_cdna4_kmd.json"
    -e "RJ_SCHEMA=${INSTALL_DIR}/share/rocjitsu/schemas/simulation_config.fbs"
    -e "RJ_LOG_LEVEL=${RJ_LOG_LEVEL:-1}"
    -e "HSA_ENABLE_SDMA=0"
    -e "ROCPROFILER_REGISTER_ENABLED=0"
)

VOLUME_ARGS=()

if [[ "${RJ_INSTALLED:-0}" != "1" ]]; then
    # Bind-mount mode: mount host files into the container at the install paths.
    BUILD_DIR="${RJ_BUILD_DIR:-${PROJECT_DIR}/build}"
    CONFIG="${RJ_CONFIG:-${PROJECT_DIR}/configs/amdgpu_cdna4_kmd.json}"
    SCHEMA="${PROJECT_DIR}/schemas/simulation_config.fbs"
    LIB_PATH="${BUILD_DIR}/lib/rocjitsu/src/rocjitsu/kmd/librocjitsu_kmd.so"

    if [[ ! -f "$LIB_PATH" ]]; then
        echo "Error: librocjitsu_kmd.so not found at $LIB_PATH" >&2
        echo "Build first (cd build && ninja) or set RJ_BUILD_DIR." >&2
        exit 1
    fi
    if [[ ! -f "$CONFIG" ]]; then
        echo "Error: config not found at $CONFIG" >&2
        exit 1
    fi

    VOLUME_ARGS=(
        -v "${LIB_PATH}:${INSTALL_DIR}/lib/librocjitsu_kmd.so:ro"
        -v "${CONFIG}:${INSTALL_DIR}/share/rocjitsu/configs/amdgpu_cdna4_kmd.json:ro"
        -v "${SCHEMA}:${INSTALL_DIR}/share/rocjitsu/schemas/simulation_config.fbs:ro"
    )

    echo "rocjitsu: simulating GPU via LD_PRELOAD (bind-mount)" >&2
    echo "  config: $(basename "$CONFIG")" >&2
else
    echo "rocjitsu: simulating GPU via LD_PRELOAD (pre-installed)" >&2
fi
echo "  image:  $IMAGE" >&2

exec docker run --rm \
    "${DOCKER_FLAGS[@]}" \
    "${VOLUME_ARGS[@]}" \
    "${ENV_ARGS[@]}" \
    "$IMAGE" \
    "$@"
