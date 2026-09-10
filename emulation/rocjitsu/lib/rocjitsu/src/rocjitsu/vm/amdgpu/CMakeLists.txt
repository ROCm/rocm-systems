# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_vm_amdgpu
    aql_packet_processor.cpp
    wavefront.cpp
    cluster_lds_multicast.cpp
    compute_unit.cpp
    hwreg.cpp
    memory_pipeline.cpp
    circular_ring_reader.cpp
    command_processor.cpp
    consumer_cursor_journal.cpp
    pm4_ring_consumer.cpp
    pm4_packet_processor.cpp
    gpu_vm.cpp
    gpu_queue_registry.cpp
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
