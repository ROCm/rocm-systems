# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_vm_amdgpu
    wavefront.cpp
    compute_unit.cpp
    memory_pipeline.cpp
    command_processor.cpp
    shader_engine.cpp
    xcd.cpp
    iod.cpp
    l1_scalar_cache.cpp
    l1_vector_cache.cpp
    l2_cache.cpp
    memory_side_cache.cpp
)
