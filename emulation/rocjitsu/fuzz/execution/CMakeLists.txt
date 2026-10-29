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
        rocjitsu_executable
        rocjitsu_isa
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

# Retain the ppc64le GCC 14+ -Wfree-nonheap-object demotion as a precaution
# pending verification of the pre-sized random_input() implementation there.
# The former reserve()/push_back() implementation triggered GCC PR 115016
# at -O3; the new implementation compiles cleanly on x86-64 GCC 14/15.
# Recheck: remove this block and build execution_fuzz_core.cpp on ppc64le
# GCC 14 at -O3 before removing the precaution.
if(
    CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
    AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "14"
    AND CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64"
)
    set_property(
        SOURCE execution_fuzz_core.cpp
        APPEND
        PROPERTY COMPILE_OPTIONS -Wno-error=free-nonheap-object
    )
endif()

add_executable(rj_execution_fuzz execution_fuzz_main.cpp)
target_link_libraries(
    rj_execution_fuzz
    PRIVATE rj_execution_fuzz_core rocjitsu_isa_registry
)
rj_configure_target(rj_execution_fuzz PRIVATE_INCLUDES WARNINGS)
