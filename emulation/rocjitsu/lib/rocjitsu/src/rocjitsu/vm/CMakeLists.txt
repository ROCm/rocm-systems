# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_vm
    soc.cpp
    virtual_machine.cpp
    rj_vm.cpp
)

# rj_vm.cpp uses config and checkpoint APIs which depend on FlatBuffers.
target_link_libraries(rocjitsu_vm PRIVATE flatbuffers)
target_include_directories(rocjitsu_vm PRIVATE ${GENERATED_DIR})
add_dependencies(rocjitsu_vm flatbuffers_schemas)

add_subdirectory(amdgpu)
add_subdirectory(risc_v)
