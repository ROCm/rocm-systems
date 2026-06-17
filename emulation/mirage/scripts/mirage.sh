#!/usr/bin/env bash

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Convenience entry point: build the glibc-portable mirage + rocjitsu
# prefix via `mirage-docker-build.sh` (if not already built) and then run
# the freshly installed `mirage` binary, forwarding all arguments.
#
# The installed `<prefix>/bin/mirage` finds its sibling
# `<prefix>/lib/librocjitsu*.so` automatically, so no LD_LIBRARY_PATH or
# extra wiring is needed.
#
# Usage:
#   ./scripts/mirage.sh [mirage args...]
#
# Examples:
#   ./scripts/mirage.sh --help
#   ./scripts/mirage.sh run --profile rocjitsu-MI350X -- rocminfo
#
# Environment variables:
#   MIRAGE_PREFIX  - install/run prefix (default: <mirage>/build/manylinux)
#   MIRAGE_SKIP_BUILD - set to 1 to skip the rebuild and run the existing binary
#   RJ_LOG_GROUPS  - rocjitsu compile-time log groups (default: VM; see
#                    rocjitsu/cmake/rj_log.cmake -- OFF, ALL, or names like
#                    VM,CP,DBT_HOOKS)
#   plus everything honoured by mirage-docker-build.sh
#   (MIRAGE_BUILD_IMAGE, MIRAGE_IMAGE_TAG, CONTAINER_ENGINE, CARGO_PROFILE)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MIRAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PREFIX="${MIRAGE_PREFIX:-${MIRAGE_DIR}/build/manylinux}"
MIRAGE_BIN="$PREFIX/bin/mirage"

# Always rebuild via the Docker image -- BuildKit layer + cache mounts make
# re-runs fast, and this guarantees source/log-group changes are picked up.
# Set MIRAGE_SKIP_BUILD=1 to bypass and run the already-installed binary.
if [ "${MIRAGE_SKIP_BUILD:-0}" != "1" ]; then
    echo "mirage: building via mirage-docker-build.sh ($PREFIX)" >&2
    "$SCRIPT_DIR/mirage-docker-build.sh" "$PREFIX" >&2
fi

if [ ! -x "$MIRAGE_BIN" ]; then
    echo "mirage: build did not produce $MIRAGE_BIN" >&2
    exit 1
fi

exec "$MIRAGE_BIN" "$@"
