# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build the arithmetic library with optional x86-64 SIMD backends.
# Public include paths and implementation-only compiler settings are defined here.

add_library(
    rj_arith
    STATIC
    common/cpu.cpp
    common/cpu_decode.cpp
    fp/scalar_repair.cpp
    fp/fma.cpp
    fp/half_fma.cpp
    dot/scalar_repair.cpp
    register/dpp.cpp
    wmma/swmmac16.cpp
    wmma/swmmac8.cpp
    wmma/swmmac_float.cpp
    wmma/swmmac_integer.cpp
    wmma/wmma.cpp
    wmma/wmma_fp8_integer.cpp
    wmma/wmma_replicated.cpp
)

target_include_directories(
    rj_arith
    PUBLIC ${rj_arith_source_dir}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
if(UNIX)
    target_link_libraries(rj_arith PRIVATE m)
endif()
rj_configure_target(rj_arith WARNINGS)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(rj_arith PRIVATE -ffp-contract=off)
endif()
# Reset the CPU model for baseline sources. Explicit ISA feature flags (such as
# -mavx2) remain the enclosing build's responsibility; -march does not cancel them.
get_target_property(rj_arith_baseline_sources rj_arith SOURCES)
set_source_files_properties(
    ${rj_arith_baseline_sources}
    PROPERTIES COMPILE_DEFINITIONS RJ_ARITH_IMPL_NAMESPACE=baseline
)
if(RJ_ARITH_ENABLE_X86_64)
    target_compile_options(rj_arith PRIVATE -march=x86-64)
    target_compile_definitions(rj_arith PRIVATE RJ_ARITH_X86_64_QUERY=1)
    include(${CMAKE_CURRENT_SOURCE_DIR}/x86_64.cmake)
endif()
