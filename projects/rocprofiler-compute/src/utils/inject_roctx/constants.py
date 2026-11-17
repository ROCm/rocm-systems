# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Backend-selection constants for inject_roctx."""

from typing import Tuple

# Backends recognized by install_global_wraps.
KNOWN_ML_API_BACKENDS: Tuple[str, ...] = ("torch", "triton")
