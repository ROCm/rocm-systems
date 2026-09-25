#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
set -euo pipefail
# All invocations share this slice; limits apply to their aggregate usage.
systemctl --user set-property consan-validation.slice \
  MemoryHigh=36G MemoryMax=40G MemorySwapMax=1G
exec systemd-run --user --scope --slice=consan-validation.slice -- "$@"
