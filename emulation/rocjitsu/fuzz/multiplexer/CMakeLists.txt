# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(rj_multiplexer_fuzz_core STATIC multiplexer_fuzz_core.cpp)
target_include_directories(
    rj_multiplexer_fuzz_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
    rj_multiplexer_fuzz_core
    PUBLIC
        rj_decode_fuzz_core
        rj_execution_fuzz_core
        rj_code_object_fuzz_core
        rj_concurrent_dispatch_fuzz_core
)
rj_configure_target(rj_multiplexer_fuzz_core PRIVATE_INCLUDES WARNINGS)

add_executable(rj_fuzz multiplexer_fuzz_main.cpp)
target_link_libraries(
    rj_fuzz
    PRIVATE rj_multiplexer_fuzz_core rocjitsu_isa_registry
)
rj_configure_target(rj_fuzz PRIVATE_INCLUDES WARNINGS)
