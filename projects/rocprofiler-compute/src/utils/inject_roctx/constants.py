# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Backend-selection constants for inject_roctx."""

# Backends recognized by install_global_wraps.
KNOWN_ML_API_BACKENDS: tuple[str, ...] = ("torch", "triton")
