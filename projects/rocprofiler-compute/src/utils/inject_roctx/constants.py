# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Backend-selection constants for inject_roctx."""

# Backends recognized by install_global_wraps and the "api" alias.
KNOWN_BACKENDS: tuple[str, ...] = ("torch", "triton")

# The "api" alias selects every backend in KNOWN_BACKENDS.
API_ALIAS = "api"
