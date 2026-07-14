#!/bin/bash

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Build mirage AND rocjitsu via the multi-stage emulation/Dockerfile and
# install both into a host output directory.
#
# WHY: mirage's per-session host bind-mounts its own `mirage` binary and
# the rocjitsu KMD interposer into the workload container. When those are
# built on a modern host they link a newer glibc than older images carry
# (e.g. vllm jammy = glibc 2.35), so the in-container binary fails with
# `version 'GLIBC_2.39' not found`. The Dockerfile builds inside the
# TheRock manylinux image, which links against an old glibc with broad
# forward compatibility, so the resulting binaries run in (almost) any
# target container WITHOUT the `--hack`/derived-image glibc workaround.
#
# This script is a thin wrapper around `docker build`: it builds the
# image (mirage + rocjitsu compile in PARALLEL stages, with BuildKit
# cache mounts so re-runs are fast) and then extracts the assembled
# `/opt/mirage` prefix out of the image into a host directory with the
# standard layout:
#   <prefix>/bin/mirage
#   <prefix>/bin/rocjitsu
#   <prefix>/lib/librocjitsu.so            (combined rocjitsu library:
#                                          VM API + KMD interposer)
#   <prefix>/lib/librocjitsu_hooks.so      (DBT HSA hooks)
#   <prefix>/share/rocjitsu/configs/*.json
# `mirage` searches `../lib` relative to its own binary (see
# rocjitsu/src/lib.rs kmd_search_dirs), so `<prefix>/bin/mirage` finds
# its sibling `<prefix>/lib/librocjitsu*.so` automatically.
#
# Usage:
#   ./scripts/mirage-docker-build.sh [output-prefix]
#
# Examples:
#   ./scripts/mirage-docker-build.sh
#   ./scripts/mirage-docker-build.sh ./build/manylinux
#
# Environment variables:
#   MIRAGE_BUILD_IMAGE  - manylinux builder image passed as the Dockerfile
#                         BUILD_IMAGE arg
#                         (default: ghcr.io/rocm/therock_build_manylinux_x86_64:main)
#   MIRAGE_IMAGE_TAG    - tag for the built image (default: mirage:local)
#   CONTAINER_ENGINE    - docker or podman (default: docker)
#   CARGO_PROFILE       - cargo profile: release or debug (default: release)
#   RJ_LOG_GROUPS       - rocjitsu compile-time log groups passed as the
#                         Dockerfile RJ_LOG_GROUPS arg (default: VM; see
#                         rocjitsu/cmake/rj_log.cmake -- OFF, ALL, or
#                         comma-separated names like VM,CP,DBT_HOOKS)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MIRAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EMULATION_DIR="$(cd "$MIRAGE_DIR/.." && pwd)"
DOCKERFILE="$EMULATION_DIR/Dockerfile"

BUILD_IMAGE="${MIRAGE_BUILD_IMAGE:-ghcr.io/rocm/therock_build_manylinux_x86_64:main}"
IMAGE_TAG="${MIRAGE_IMAGE_TAG:-mirage:local}"
ENGINE="${CONTAINER_ENGINE:-docker}"
CARGO_PROFILE="${CARGO_PROFILE:-release}"
RJ_LOG_GROUPS="${RJ_LOG_GROUPS:-OFF}"

# Resolve the output prefix to an absolute host path and create it so we
# can copy artifacts into it.
OUTPUT_PREFIX="${1:-${MIRAGE_DIR}/build/manylinux}"
mkdir -p "$OUTPUT_PREFIX"
OUTPUT_PREFIX="$(cd "$OUTPUT_PREFIX" && pwd)"

echo "mirage: building mirage + rocjitsu via emulation/Dockerfile" >&2
echo "  dockerfile: $DOCKERFILE" >&2
echo "  context:    $EMULATION_DIR" >&2
echo "  builder:    $BUILD_IMAGE" >&2
echo "  image tag:  $IMAGE_TAG" >&2
echo "  engine:     $ENGINE" >&2
echo "  profile:    $CARGO_PROFILE" >&2
echo "  log groups: $RJ_LOG_GROUPS" >&2
echo "  install:    $OUTPUT_PREFIX" >&2

# --- Build the image (mirage + rocjitsu stages run in parallel) -------
# Enable BuildKit for docker so the cache mounts in the Dockerfile work.
# podman supports the same BuildKit features natively.
DOCKER_BUILDKIT=1 "$ENGINE" build \
    -f "$DOCKERFILE" \
    -t "$IMAGE_TAG" \
    --build-arg "BUILD_IMAGE=${BUILD_IMAGE}" \
    --build-arg "CARGO_PROFILE=${CARGO_PROFILE}" \
    --build-arg "RJ_LOG_GROUPS=${RJ_LOG_GROUPS}" \
    "$EMULATION_DIR"

# --- Extract /opt/mirage out of the image into the host prefix --------
# Create a throwaway container (never started) and copy the assembled
# prefix out of it. `docker cp <id>:/opt/mirage/.` copies the *contents*
# into OUTPUT_PREFIX so the layout becomes <prefix>/bin, <prefix>/lib, ...
echo "== Extracting artifacts into $OUTPUT_PREFIX ==" >&2
CID="$("$ENGINE" create "$IMAGE_TAG")"
trap '"$ENGINE" rm -f "$CID" >/dev/null 2>&1 || true' EXIT
"$ENGINE" cp "$CID:/opt/mirage/." "$OUTPUT_PREFIX/"

# --- Report -----------------------------------------------------------
echo "== Installed artifacts ==" >&2
ls -lh "$OUTPUT_PREFIX"/bin/mirage "$OUTPUT_PREFIX"/lib/librocjitsu*.so 2>&1 || true

if command -v objdump >/dev/null 2>&1; then
    echo "== glibc version requirements (max GLIBC_* symbol per binary) ==" >&2
    for f in "$OUTPUT_PREFIX"/bin/mirage "$OUTPUT_PREFIX"/lib/librocjitsu*.so; do
        [ -f "$f" ] || continue
        maxglibc=$(objdump -T "$f" 2>/dev/null \
            | grep -oE "GLIBC_[0-9]+\.[0-9]+" \
            | sort -V | tail -1)
        echo "  $(basename "$f"): ${maxglibc:-none}" >&2
    done
fi

echo "mirage: build complete; artifacts in $OUTPUT_PREFIX" >&2
echo "  run with: $OUTPUT_PREFIX/bin/mirage --help" >&2
