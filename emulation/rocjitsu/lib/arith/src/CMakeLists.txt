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
    dot/scalar_repair.cpp
    fp/fma.cpp
    fp/binary.cpp
    fp/div_fixup.cpp
    fp/div_fmas.cpp
    fp/div_scale.cpp
    fp/fp64.cpp
    fp/frexp_exp.cpp
    fp/half_binary.cpp
    fp/half_exponent.cpp
    fp/half_fma.cpp
    fp/half_minmax.cpp
    fp/ldexp.cpp
    fp/minmax.cpp
    fp/mixed_fma.cpp
    fp/packed_fma.cpp
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
if(RJ_ARITH_ENABLE_X86_64)
    target_compile_definitions(rj_arith PRIVATE RJ_ARITH_X86_64_QUERY=1)
    include(${CMAKE_CURRENT_SOURCE_DIR}/x86_64.cmake)
endif()
