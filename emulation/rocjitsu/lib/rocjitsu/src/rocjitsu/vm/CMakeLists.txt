# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_vm
    soc.cpp
    virtual_machine.cpp
    topology_assembly.cpp
    fabric_backing_kfd.cpp
    canonical_backing_id.cpp
    canonical_line_freshness.cpp
    fabric_endpoint.cpp
    rj_vm.cpp
    amdgpu/partitioning.cpp
)

# rj_vm.cpp uses config and checkpoint APIs which depend on FlatBuffers.
target_link_libraries(rocjitsu_vm PRIVATE flatbuffers)
target_include_directories(rocjitsu_vm PRIVATE ${GENERATED_DIR})
add_dependencies(rocjitsu_vm flatbuffers_schemas)

add_subdirectory(amdgpu)
add_subdirectory(risc_v)
