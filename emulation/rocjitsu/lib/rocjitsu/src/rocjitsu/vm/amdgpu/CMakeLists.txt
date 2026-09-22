# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_vm_amdgpu
    buffer_format.cpp
    cluster_lds_multicast.cpp
    command_processor.cpp
    completion_tracker.cpp
    compute_unit.cpp
    device_cache_coherence.cpp
    gpu_queue_registry.cpp
    gpu_vm.cpp
    graphics_draw.cpp
    hwreg.cpp
    iod.cpp
    l1_scalar_cache.cpp
    l1_vector_cache.cpp
    l2_cache.cpp
    lds_stack.cpp
    memory_pipeline.cpp
    memory_side_cache.cpp
    shader_engine.cpp
    wavefront.cpp
    xcd.cpp
)

add_subdirectory(pci)
