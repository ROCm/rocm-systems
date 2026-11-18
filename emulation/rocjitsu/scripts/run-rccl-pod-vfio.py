#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Launch the same-host VFIO RCCL pod gate."""

from rccl_pod_vfio import main

if __name__ == "__main__":
    raise SystemExit(main())
