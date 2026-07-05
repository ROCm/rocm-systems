# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_analysis
    def_use_chain.cpp
    indirect_branch_discovery.cpp
    liveness.cpp
)
