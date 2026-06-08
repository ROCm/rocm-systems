# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Concrete plugins (each will become a standalone .so in the future — see #6628).
rj_add_object_library(
    rocjitsu_plugin_logging
    logging/plugin.cpp
)
