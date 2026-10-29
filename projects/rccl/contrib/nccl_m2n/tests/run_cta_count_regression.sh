#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NCCL_RESHARD_NUM_CTAS is read once per process, so exercise each CTA count in
# a fresh basic_api_test_local process.
#
# The counts below deliberately mix multiples of the default GIN context count
# with counts that are not.  A sweep of only {1,4,8,16,32} cannot see a broken
# CTA-to-context mapping: every one of those values either divides evenly or
# is below the context count, which is exactly the set that survived the
# out-of-range mapping this script was written for.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/bin/basic_api_test_local"
CTA_COUNTS=(1 4 5 6 8 9 16 32)

if [[ ! -x "$BIN" ]]; then
    echo "error: $BIN not found. Build with 'make tests' first." >&2
    exit 1
fi

for num_ctas in "${CTA_COUNTS[@]}"; do
    echo "=== NCCL_RESHARD_NUM_CTAS=$num_ctas ==="
    NCCL_RESHARD_NUM_CTAS="$num_ctas" "$BIN" "$@" --filter tiny_contribution
done
