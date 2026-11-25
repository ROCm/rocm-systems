# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(
    rj_concurrent_dispatch_fuzz_core
    STATIC
    concurrent_dispatch_fuzz_core.cpp
)
target_include_directories(
    rj_concurrent_dispatch_fuzz_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
    rj_concurrent_dispatch_fuzz_core
    PUBLIC rj_execution_fuzz_core
)
rj_configure_target(rj_concurrent_dispatch_fuzz_core GENERATED PRIVATE_INCLUDES WARNINGS)

add_executable(rj_concurrent_dispatch_fuzz concurrent_dispatch_fuzz_main.cpp)
target_link_libraries(
    rj_concurrent_dispatch_fuzz
    PRIVATE rj_concurrent_dispatch_fuzz_core rocjitsu_isa_registry
)
rj_configure_target(rj_concurrent_dispatch_fuzz PRIVATE_INCLUDES WARNINGS)
