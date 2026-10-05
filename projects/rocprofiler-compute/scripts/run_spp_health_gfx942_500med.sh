#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Compatibility wrapper: 500-iter median health now lives in
# scripts/run_spp_health_gfx942.sh (ITERS=500 default).
#
#   TAG=261001-gfx942-500med THEROCK=... bash scripts/run_spp_health_gfx942_500med.sh

set -euo pipefail

TAG=${TAG:-$(date +%y%m%d)-gfx942-500med}
ITERS=${ITERS:-500}
NO_ROOF=${NO_ROOF:-1}
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

export TAG ITERS NO_ROOF ROOT
exec bash "${ROOT}/scripts/run_spp_health_gfx942.sh" "$@"
