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
#   MIRAGE_REBUILD - set to 1 to force a rebuild even if the binary exists
#   plus everything honoured by mirage-docker-build.sh
#   (MIRAGE_BUILD_IMAGE, MIRAGE_IMAGE_TAG, CONTAINER_ENGINE, CARGO_PROFILE)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MIRAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PREFIX="${MIRAGE_PREFIX:-${MIRAGE_DIR}/build/manylinux}"
MIRAGE_BIN="$PREFIX/bin/mirage"

# Build via the Docker image when the binary is missing or a rebuild is
# explicitly requested.
if [ "${MIRAGE_REBUILD:-0}" = "1" ] || [ ! -x "$MIRAGE_BIN" ]; then
    echo "mirage: building via mirage-docker-build.sh ($PREFIX)" >&2
    "$SCRIPT_DIR/mirage-docker-build.sh" "$PREFIX" >&2
fi

if [ ! -x "$MIRAGE_BIN" ]; then
    echo "mirage: build did not produce $MIRAGE_BIN" >&2
    exit 1
fi

exec "$MIRAGE_BIN" "$@"
