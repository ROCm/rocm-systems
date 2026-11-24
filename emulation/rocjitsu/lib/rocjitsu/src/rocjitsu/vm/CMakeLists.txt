# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# The RISC-V and AMDGPU models construct default plugin groups independently.
add_library(rocjitsu_execution_plugin_group INTERFACE)
rj_add_host_policy_sources(
    rocjitsu_execution_plugin_group plugins/execution_plugin_group.cpp
)

rj_add_object_library(rocjitsu_vm
    soc.cpp
    topology_assembly.cpp
    fabric_backing_kfd.cpp
    canonical_backing_id.cpp
    canonical_line_freshness.cpp
    fabric_endpoint.cpp
    rj_vm_config.cpp
    amdgpu/partitioning.cpp
)

# rj_vm.cpp uses config and checkpoint APIs which depend on FlatBuffers.
target_link_libraries(
    rocjitsu_vm
    PUBLIC rocjitsu_execution_plugin_group
    PRIVATE flatbuffers
)
target_include_directories(rocjitsu_vm PRIVATE ${GENERATED_DIR})
add_dependencies(rocjitsu_vm flatbuffers_schemas)
rj_add_host_policy_sources(rocjitsu_vm rj_vm.cpp virtual_machine.cpp)

add_subdirectory(amdgpu)
add_subdirectory(risc_v)
