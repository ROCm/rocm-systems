#!/bin/bash

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Build rocjitsu inside a Docker container and install to /opt/rocm.
# Installs GCC 14 from the ubuntu-toolchain-r PPA (needed for C++20
# <format> support) and builds with the container's native glibc.
#
# Usage:
#   ./scripts/rocjitsu-docker-build.sh <image> [install-prefix]
#
# Example:
#   ./scripts/rocjitsu-docker-build.sh vllm/vllm-openai-rocm:latest
#   ./scripts/rocjitsu-docker-build.sh vllm/vllm-openai-rocm:latest /opt/rocm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MONOREPO_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"

IMAGE="${1:?Usage: $0 <image> [install-prefix]}"
INSTALL_PREFIX="${2:-/opt/rocm}"

echo "rocjitsu: building inside container ($IMAGE)" >&2
echo "  source:  $PROJECT_DIR" >&2
echo "  install: $INSTALL_PREFIX" >&2

docker run --rm \
    -v "${MONOREPO_DIR}:/src:ro" \
    --tmpfs /src/emulation/rocjitsu/third_party \
    --entrypoint bash \
    "$IMAGE" \
    -c '
set -e
INSTALL_PREFIX='"'${INSTALL_PREFIX}'"'

echo "Installing GCC 14 from ubuntu-toolchain-r PPA..."

# add-apt-repository is broken in some containers (apt_pkg not found
# for the container Python). Add the PPA manually via curl + gpg.
mkdir -p /root/.gnupg && chmod 700 /root/.gnupg
curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x1E9377A2BA9EF27F" \
    | gpg --dearmor > /usr/share/keyrings/toolchain.gpg
echo "deb [signed-by=/usr/share/keyrings/toolchain.gpg] http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu jammy main" \
    > /etc/apt/sources.list.d/toolchain.list

apt-get update -qq
apt-get install -qq -y gcc-14 g++-14 ninja-build > /dev/null 2>&1

echo "GCC version: $(g++-14 --version | head -1)"

echo "Configuring..."
mkdir -p /tmp/build && cd /tmp/build
cmake -G Ninja \
    -DCMAKE_C_COMPILER=gcc-14 \
    -DCMAKE_CXX_COMPILER=g++-14 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_SCAN_FOR_MODULES=OFF \
    -DCMAKE_SHARED_LINKER_FLAGS="-static-libstdc++" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DFETCHCONTENT_BASE_DIR=/tmp/deps \
    -DBUILD_TESTING=OFF \
    /src/emulation/rocjitsu 2>&1 | tail -5

echo "Building..."
ninja -j$(nproc) 2>&1 | tail -5

echo "Installing to $INSTALL_PREFIX..."
cmake --install . 2>&1 | tail -10

echo "Installed files:"
ls -lh "$INSTALL_PREFIX"/lib/librocjitsu*.so
ls "$INSTALL_PREFIX"/include/rocjitsu/
ls "$INSTALL_PREFIX"/share/rocjitsu/configs/

echo "Dynamic dependencies:"
ldd "$INSTALL_PREFIX"/lib/librocjitsu.so
'

echo "rocjitsu: installed to $INSTALL_PREFIX inside $IMAGE" >&2
