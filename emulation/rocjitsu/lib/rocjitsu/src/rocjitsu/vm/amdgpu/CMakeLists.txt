# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_vm_amdgpu
    wavefront.cpp
    cluster_lds_multicast.cpp
    compute_unit.cpp
    hwreg.cpp
    memory_pipeline.cpp
    command_processor.cpp
    pm4_bootstrap_executor.cpp
    sdma_executor.cpp
    sdma_queue_runner.cpp
    gpu_vm.cpp
    queue_service.cpp
    completion_tracker.cpp
    shader_engine.cpp
    xcd.cpp
    iod.cpp
    device_cache_coherence.cpp
    l1_scalar_cache.cpp
    l1_vector_cache.cpp
    l2_cache.cpp
    memory_side_cache.cpp
)

add_subdirectory(pci)
