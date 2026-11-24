# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Backend identities shared by ML profiling and analysis."""

from typing import Tuple

KNOWN_ML_API_BACKENDS: Tuple[str, ...] = ("torch", "triton")
USER_REGION_BACKEND: str = "user"

# Backends accepted by --list-operators and --operator.
OPERATOR_BACKENDS: Tuple[str, ...] = (*KNOWN_ML_API_BACKENDS, USER_REGION_BACKEND)
