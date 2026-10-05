#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Compatibility wrapper for the main 500-iter local driver.
#
#   TAG=261001-gfx942-500med ROCR_VISIBLE_DEVICES=17 \
#     ./scripts/run_spp_health_gfx942_500med_local_driver.sh --wait --fetch --report

set -euo pipefail

TAG=${TAG:-261001-gfx942-500med}
ITERS=${ITERS:-500}
NO_ROOF=${NO_ROOF:-1}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export TAG ITERS NO_ROOF
# Map legacy --compare to a no-op hint; main driver uses --report (median HTML).
args=()
for a in "$@"; do
  if [[ "$a" == "--compare" ]]; then
    echo "NOTE: --compare deprecated here; use tools/compare_spp_legacy_medians.py" >&2
    continue
  fi
  args+=("$a")
done
exec bash "${REPO_ROOT}/scripts/run_spp_health_gfx942_local_driver.sh" "${args[@]}"
