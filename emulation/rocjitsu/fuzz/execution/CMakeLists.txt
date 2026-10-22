# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(rj_execution_fuzz_core STATIC execution_fuzz_core.cpp)
target_include_directories(
    rj_execution_fuzz_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
    rj_execution_fuzz_core
    PUBLIC
        rocjitsu_analysis
        rocjitsu_code
        rocjitsu_code_object
        rocjitsu_isa
        rocjitsu_isa_registry
        simdojo
        rocjitsu_kmd
        rocjitsu_vm
        rocjitsu_vm_amdgpu
        rocjitsu_plugin_loader
        rocjitsu_vm_risc_v
        rocjitsu_config
        rocjitsu_dbt_config
        rocjitsu_ip_discovery
        util
        flatbuffers
        Threads::Threads
)
rj_configure_target(rj_execution_fuzz_core GENERATED PRIVATE_INCLUDES WARNINGS)

add_executable(rj_execution_fuzz execution_fuzz_main.cpp)
target_link_libraries(rj_execution_fuzz PRIVATE rj_execution_fuzz_core)
rj_configure_target(rj_execution_fuzz PRIVATE_INCLUDES WARNINGS)
