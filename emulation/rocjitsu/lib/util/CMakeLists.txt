# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(util_headers INTERFACE)
target_include_directories(
    util_headers
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(util STATIC src/simd.cpp)
target_link_libraries(util PUBLIC util_headers)
set_target_properties(util PROPERTIES POSITION_INDEPENDENT_CODE ON)
