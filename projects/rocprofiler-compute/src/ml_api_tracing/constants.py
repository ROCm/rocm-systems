# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Backend identities shared by ML profiling and analysis."""

from typing import Tuple

KNOWN_ML_API_BACKENDS: Tuple[str, ...] = ("torch", "triton")
