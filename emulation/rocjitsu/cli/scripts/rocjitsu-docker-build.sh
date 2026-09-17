#!/bin/bash

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Build rocjitsu via emulation/Dockerfile and install it into a host
# output directory.
#
# WHY: rocjitsu's per-session host bind-mounts its own `rocjitsu` binary and
# the rocjitsu KMD interposer into the workload container. When those are
# built on a modern host they link a newer glibc than older images carry
# (e.g. vllm jammy = glibc 2.35), so the in-container binary fails with
# `version 'GLIBC_2.39' not found`. The Dockerfile builds inside the
# TheRock manylinux image, which links against an old glibc with broad
# forward compatibility, so the resulting binaries run in (almost) any
# target container WITHOUT the `--hack`/derived-image glibc workaround.
#
# This script is a thin wrapper around `docker build`: it builds the
# image (with BuildKit cache mounts so re-runs are fast) and then extracts
# the assembled `/opt/rocjitsu` prefix out of the image into a host
# directory with the standard layout:
#   <prefix>/bin/rocjitsu
#   <prefix>/bin/mirage                    (symlink, the CLI's old name)
#   <prefix>/lib/librocjitsu.so            (combined rocjitsu library:
#                                          VM API + KMD interposer)
#   <prefix>/lib/librocjitsu_hooks.so      (DBT HSA hooks)
#   <prefix>/share/rocjitsu/configs/*.json
# The CLI searches `<prefix>/lib` beside its own binary (see
# cli/core/src/discovery.rs, module docs), ahead of any rocjitsu build
# it can reach by walking up from itself, so `<prefix>/bin/rocjitsu` finds
# its sibling `<prefix>/lib/librocjitsu*.so` and not a host-glibc build
# left in the checkout.
#
# Usage:
#   ./scripts/rocjitsu-docker-build.sh [output-prefix]
#
# Examples:
#   ./scripts/rocjitsu-docker-build.sh
#   ./scripts/rocjitsu-docker-build.sh ./build/manylinux
#
# Environment variables:
#   ROCJITSU_BUILD_IMAGE  - manylinux builder image passed as the Dockerfile
#                         BUILD_IMAGE arg
#                         (default: ghcr.io/rocm/therock_build_manylinux_x86_64:main)
#   ROCJITSU_IMAGE_TAG    - tag for the built image (default: rocjitsu:local)
#   CONTAINER_ENGINE    - docker or podman (default: docker)
#   CARGO_PROFILE       - cargo profile: release or debug (default: release)
#   RJ_LOG_GROUPS       - rocjitsu compile-time log groups passed as the
#                         Dockerfile RJ_LOG_GROUPS arg (default: VM; see
#                         rocjitsu/cmake/rj_log.cmake -- OFF, ALL, or
#                         comma-separated names like VM,CP,DBT_HOOKS)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROCJITSU_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EMULATION_DIR="$(cd "$ROCJITSU_DIR/.." && pwd)"
DOCKERFILE="$EMULATION_DIR/Dockerfile"

BUILD_IMAGE="${ROCJITSU_BUILD_IMAGE:-ghcr.io/rocm/therock_build_manylinux_x86_64:main}"
IMAGE_TAG="${ROCJITSU_IMAGE_TAG:-rocjitsu:local}"
ENGINE="${CONTAINER_ENGINE:-docker}"
CARGO_PROFILE="${CARGO_PROFILE:-release}"
RJ_LOG_GROUPS="${RJ_LOG_GROUPS:-OFF}"

# Resolve the output prefix to an absolute host path and create it so we
# can copy artifacts into it.
OUTPUT_PREFIX="${1:-${ROCJITSU_DIR}/build/manylinux}"
mkdir -p "$OUTPUT_PREFIX"
OUTPUT_PREFIX="$(cd "$OUTPUT_PREFIX" && pwd)"

echo "rocjitsu: building via emulation/Dockerfile" >&2
echo "  dockerfile: $DOCKERFILE" >&2
echo "  context:    $EMULATION_DIR" >&2
echo "  builder:    $BUILD_IMAGE" >&2
echo "  image tag:  $IMAGE_TAG" >&2
echo "  engine:     $ENGINE" >&2
echo "  profile:    $CARGO_PROFILE" >&2
echo "  log groups: $RJ_LOG_GROUPS" >&2
echo "  install:    $OUTPUT_PREFIX" >&2

# --- Build the image --------------------------------------------------
# Enable BuildKit for docker so the cache mounts in the Dockerfile work.
# podman supports the same BuildKit features natively.
DOCKER_BUILDKIT=1 "$ENGINE" build \
    -f "$DOCKERFILE" \
    -t "$IMAGE_TAG" \
    --build-arg "BUILD_IMAGE=${BUILD_IMAGE}" \
    --build-arg "CARGO_PROFILE=${CARGO_PROFILE}" \
    --build-arg "RJ_LOG_GROUPS=${RJ_LOG_GROUPS}" \
    "$EMULATION_DIR"

# --- Extract /opt/rocjitsu out of the image into the host prefix --------
# Create a throwaway container (never started) and copy the assembled
# prefix out of it. `docker cp <id>:/opt/rocjitsu/.` copies the *contents*
# into OUTPUT_PREFIX so the layout becomes <prefix>/bin, <prefix>/lib, ...
echo "== Extracting artifacts into $OUTPUT_PREFIX ==" >&2
CID="$("$ENGINE" create "$IMAGE_TAG")"
trap '"$ENGINE" rm -f "$CID" >/dev/null 2>&1 || true' EXIT
"$ENGINE" cp "$CID:/opt/rocjitsu/." "$OUTPUT_PREFIX/"

# --- Report -----------------------------------------------------------
echo "== Installed artifacts ==" >&2
ls -lh "$OUTPUT_PREFIX"/bin/rocjitsu "$OUTPUT_PREFIX"/lib/librocjitsu*.so 2>&1 || true

if command -v objdump >/dev/null 2>&1; then
    echo "== glibc version requirements (max GLIBC_* symbol per binary) ==" >&2
    for f in "$OUTPUT_PREFIX"/bin/rocjitsu "$OUTPUT_PREFIX"/lib/librocjitsu*.so; do
        [ -f "$f" ] || continue
        maxglibc=$(objdump -T "$f" 2>/dev/null \
            | grep -oE "GLIBC_[0-9]+\.[0-9]+" \
            | sort -V | tail -1)
        echo "  $(basename "$f"): ${maxglibc:-none}" >&2
    done
fi

echo "rocjitsu: build complete; artifacts in $OUTPUT_PREFIX" >&2
echo "  run with: $OUTPUT_PREFIX/bin/rocjitsu --help" >&2
