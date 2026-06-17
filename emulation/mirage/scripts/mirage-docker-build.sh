#!/bin/bash

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Build mirage AND rocjitsu inside the TheRock manylinux container and
# install both into a host output directory.
#
# WHY: mirage's per-session host bind-mounts its own `mirage` binary and
# the rocjitsu KMD interposer into the workload container. When those are
# built on a modern host they link a newer glibc than older images carry
# (e.g. vllm jammy = glibc 2.35), so the in-container binary fails with
# `version 'GLIBC_2.39' not found`. Building inside the manylinux image
# links against an old glibc with broad forward compatibility, so the
# resulting binaries run in (almost) any target container WITHOUT the
# `--hack`/derived-image glibc workaround.
#
# The build installs into a single prefix with the standard layout:
#   <prefix>/bin/mirage
#   <prefix>/lib/librocjitsu.so            (combined rocjitsu library)
#   <prefix>/lib/librocjitsu_kmd.so        (KMD interposer, if built)
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
#   MIRAGE_BUILD_IMAGE  - builder image
#                         (default: ghcr.io/rocm/therock_build_manylinux_x86_64:main)
#   CONTAINER_ENGINE    - docker or podman (default: docker)
#   CARGO_PROFILE       - cargo profile: release or debug (default: release)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MIRAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MONOREPO_DIR="$(cd "$MIRAGE_DIR/../.." && pwd)"

BUILD_IMAGE="${MIRAGE_BUILD_IMAGE:-ghcr.io/rocm/therock_build_manylinux_x86_64:main}"
ENGINE="${CONTAINER_ENGINE:-docker}"
CARGO_PROFILE="${CARGO_PROFILE:-release}"

# Resolve the output prefix to an absolute host path and create it so the
# container can write into the bind mount.
OUTPUT_PREFIX="${1:-${MIRAGE_DIR}/build/manylinux}"
mkdir -p "$OUTPUT_PREFIX"
OUTPUT_PREFIX="$(cd "$OUTPUT_PREFIX" && pwd)"

echo "mirage: building mirage + rocjitsu in manylinux container" >&2
echo "  image:   $BUILD_IMAGE" >&2
echo "  engine:  $ENGINE" >&2
echo "  source:  $MONOREPO_DIR" >&2
echo "  profile: $CARGO_PROFILE" >&2
echo "  install: $OUTPUT_PREFIX" >&2

"$ENGINE" run --rm \
    -v "${MONOREPO_DIR}:/src:ro" \
    -v "${OUTPUT_PREFIX}:/out" \
    --tmpfs /src/emulation/rocjitsu/third_party \
    -e "CARGO_PROFILE=${CARGO_PROFILE}" \
    --entrypoint bash \
    "$BUILD_IMAGE" \
    -c '
set -euo pipefail
PREFIX=/out

echo "== Toolchain =="
echo "gcc: $(cc --version 2>/dev/null | head -1 || echo missing)"
echo "cmake: $(cmake --version 2>/dev/null | head -1 || echo missing)"

# --- Rust toolchain ---------------------------------------------------
# The manylinux image is a build base and may not ship cargo; install a
# pinned-channel rustup toolchain into a writable location when absent.
if ! command -v cargo >/dev/null 2>&1; then
    echo "== Installing Rust toolchain (cargo not present) =="
    export RUSTUP_HOME=/tmp/rustup CARGO_HOME=/tmp/cargo
    curl -fsSL https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null
    export PATH="$CARGO_HOME/bin:$PATH"
fi
echo "cargo: $(cargo --version)"

# --- Build mirage (Rust) ----------------------------------------------
# Source is mounted read-only, so build out-of-tree.
echo "== Building mirage (cargo --$CARGO_PROFILE) =="
export CARGO_TARGET_DIR=/tmp/target
PROFILE_FLAG=""
PROFILE_OUTDIR="debug"
if [ "$CARGO_PROFILE" = "release" ]; then
    PROFILE_FLAG="--release"
    PROFILE_OUTDIR="release"
fi
cargo build $PROFILE_FLAG --manifest-path /src/emulation/mirage/Cargo.toml 2>&1 | tail -8

mkdir -p "$PREFIX/bin"
install -m 0755 "/tmp/target/$PROFILE_OUTDIR/mirage" "$PREFIX/bin/mirage"
echo "Installed: $PREFIX/bin/mirage"

# --- Build rocjitsu (C++) ---------------------------------------------
echo "== Building rocjitsu (cmake + ninja) =="
mkdir -p /tmp/rjbuild && cd /tmp/rjbuild
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_SCAN_FOR_MODULES=OFF \
    -DCMAKE_SHARED_LINKER_FLAGS="-static-libstdc++" \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DFETCHCONTENT_BASE_DIR=/tmp/deps \
    -DBUILD_TESTING=OFF \
    /src/emulation/rocjitsu 2>&1 | tail -5
ninja -j"$(nproc)" 2>&1 | tail -5
cmake --install . 2>&1 | tail -10

# --- Report -----------------------------------------------------------
echo "== Installed artifacts =="
ls -lh "$PREFIX"/bin/mirage "$PREFIX"/lib/librocjitsu*.so 2>&1 || true

echo "== glibc version requirements (max GLIBC_* symbol per binary) =="
for f in "$PREFIX"/bin/mirage "$PREFIX"/lib/librocjitsu*.so; do
    [ -f "$f" ] || continue
    maxglibc=$(objdump -T "$f" 2>/dev/null \
        | grep -oE "GLIBC_[0-9]+\.[0-9]+" \
        | sort -V | tail -1)
    echo "  $(basename "$f"): ${maxglibc:-none}"
done
'

echo "mirage: build complete; artifacts in $OUTPUT_PREFIX" >&2
echo "  run with: $OUTPUT_PREFIX/bin/mirage --help" >&2
