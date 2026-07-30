# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(util STATIC src/simd.cpp)
target_include_directories(util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(util PUBLIC rj_build_config)
set_target_properties(util PROPERTIES POSITION_INDEPENDENT_CODE ON)
