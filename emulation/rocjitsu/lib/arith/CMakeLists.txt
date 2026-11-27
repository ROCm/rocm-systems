# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

set(rj_arith_source_dir ${CMAKE_CURRENT_SOURCE_DIR})

# Enable x86-64 CPU detection and SIMD backends with GCC/Clang.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" rj_arith_target_processor)

set(RJ_ARITH_ENABLE_X86_64 OFF)
if(
    CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$"
    AND rj_arith_target_processor MATCHES "^(x86_64|amd64|x64|x86-64)$"
    AND CMAKE_SIZEOF_VOID_P EQUAL 8
)
    set(RJ_ARITH_ENABLE_X86_64 ON)
endif()

# VM request records use the C operand types without linking arithmetic into
# analysis-only objects and plugins.
add_library(rj_arith_headers INTERFACE)
target_include_directories(
    rj_arith_headers
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_subdirectory(src)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
