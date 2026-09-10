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
    aql_queue_binding_factory.cpp
    consumer_cursor_journal.cpp
    mes_engine.cpp
    pm4_ring_consumer.cpp
    pm4_packet_processor.cpp
    pm4_queue_controller.cpp
    pm4_queue_binding_factory.cpp
    sdma_packet_processor.cpp
    sdma_queue_scheduler.cpp
    sdma_queue_binding_factory.cpp
    sdma_ring_consumer.cpp
    gpu_memory_access.cpp
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
