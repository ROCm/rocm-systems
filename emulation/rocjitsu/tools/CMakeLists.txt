# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

set(RJ_TOOL_OBJECT_LIBS
    rocjitsu_analysis
    rocjitsu_code
    rocjitsu_isa
    rocjitsu_amdgpu_isa_registry
    simdojo
    rocjitsu_vm_amdgpu
)

add_library(rocjitsu_tooling STATIC dbt_translate.cpp)
target_include_directories(rocjitsu_tooling PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(rocjitsu_tooling PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(
    rocjitsu_tooling
    PUBLIC ${RJ_TOOL_OBJECT_LIBS} util flatbuffers Threads::Threads
)
rj_configure_target(rocjitsu_tooling TOOL)

add_executable(rj_dbt_translate dbt_translate_main.cpp)
target_link_libraries(rj_dbt_translate PRIVATE rocjitsu_tooling)
rj_configure_target(rj_dbt_translate TOOL)
