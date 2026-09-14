#!/usr/bin/env bash

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Convenience entry point: build the glibc-portable rocjitsu + rocjitsu
# prefix via `rocjitsu-docker-build.sh` (if not already built) and then run
# the freshly installed `rocjitsu` binary, forwarding all arguments.
#
# The installed `<prefix>/bin/rocjitsu` finds its sibling
# `<prefix>/lib/librocjitsu*.so` automatically, so no LD_LIBRARY_PATH or
# extra wiring is needed.
#
# Usage:
#   ./scripts/rocjitsu.sh [rocjitsu args...]
#
# Examples:
#   ./scripts/rocjitsu.sh --help
#   ./scripts/rocjitsu.sh run --profile rocjitsu-MI350X -- rocminfo
#
# Environment variables:
#   ROCJITSU_PREFIX  - install/run prefix (default: <rocjitsu>/build/manylinux)
#   ROCJITSU_SKIP_BUILD - set to 1 to skip the rebuild and run the existing binary
#   RJ_LOG_GROUPS  - rocjitsu compile-time log groups (default: VM; see
#                    rocjitsu/cmake/rj_log.cmake -- OFF, ALL, or names like
#                    VM,CP,DBT_HOOKS)
#   plus everything honoured by rocjitsu-docker-build.sh
#   (ROCJITSU_BUILD_IMAGE, ROCJITSU_IMAGE_TAG, CONTAINER_ENGINE, CARGO_PROFILE)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROCJITSU_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PREFIX="${ROCJITSU_PREFIX:-${ROCJITSU_DIR}/build/manylinux}"
ROCJITSU_BIN="$PREFIX/bin/rocjitsu"

# Always rebuild via the Docker image -- BuildKit layer + cache mounts make
# re-runs fast, and this guarantees source/log-group changes are picked up.
# Set ROCJITSU_SKIP_BUILD=1 to bypass and run the already-installed binary.
if [ "${ROCJITSU_SKIP_BUILD:-0}" != "1" ]; then
    echo "rocjitsu: building via rocjitsu-docker-build.sh ($PREFIX)" >&2
    "$SCRIPT_DIR/rocjitsu-docker-build.sh" "$PREFIX" >&2
fi

if [ ! -x "$ROCJITSU_BIN" ]; then
    echo "rocjitsu: build did not produce $ROCJITSU_BIN" >&2
    exit 1
fi

exec "$ROCJITSU_BIN" "$@"
