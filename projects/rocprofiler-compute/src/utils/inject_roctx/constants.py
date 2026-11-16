# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Backend-selection constants for inject_roctx."""

from typing import Tuple

# Backends recognized by install_global_wraps.
KNOWN_ML_API_BACKENDS: Tuple[str, ...] = ("torch", "triton")

# Backend name for user ROCTx regions, which carry no backend tag.
USER_REGION_BACKEND = "user"

# Backends that --list-operators and --operator accept.
OPERATOR_BACKENDS: Tuple[str, ...] = (*KNOWN_ML_API_BACKENDS, USER_REGION_BACKEND)
