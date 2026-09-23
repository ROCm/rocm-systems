# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_vm_amdgpu
    aql/aql_packet_processor.cpp
    aql/aql_queue_binding_factory.cpp
    circular_ring_reader.cpp
    cluster_lds_multicast.cpp
    command_processor.cpp
    completion_tracker.cpp
    compute_unit.cpp
    consumer_cursor_journal.cpp
    device_cache_coherence.cpp
    gpu_memory_access.cpp
    gpu_queue_registry.cpp
    gpu_vm.cpp
    hwreg.cpp
    iod.cpp
    l1_scalar_cache.cpp
    l1_vector_cache.cpp
    l2_cache.cpp
    memory_pipeline.cpp
    memory_side_cache.cpp
    mes_engine.cpp
    pm4/pm4_packet_processor.cpp
    pm4/pm4_queue_binding_factory.cpp
    pm4/pm4_queue_controller.cpp
    pm4/pm4_ring_consumer.cpp
    sdma_packet_processor.cpp
    sdma_queue_binding_factory.cpp
    sdma_queue_scheduler.cpp
    sdma_ring_consumer.cpp
    shader_engine.cpp
    wavefront.cpp
    xcd.cpp
)

add_subdirectory(pci)
